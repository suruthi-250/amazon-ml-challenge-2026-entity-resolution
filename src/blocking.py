"""Step 2: candidate generation (blocking).

Every record becomes a sparse, L2-normalised TF-IDF vector over hashed,
field-prefixed tokens (IDF from Source 1, per country label):
    n|word        name core words (legal forms / fillers removed)
    p|w1_w2       unordered pairs of the first five name words
    c|xxxxxx      first 6 chars of the space-less name (catches "acmecorp.com")
    a|tok         address tokens
    h|num_word    house number + following word ("1658_mississippi")
    x|word|tok    first two name words x every address token -- rare even when
                  both parts are common (e.g. "digital" x "gurgaon")

Two-pass retrieval for every Source-2/3 record, restricted to Source-1
records with the same country label:
  1. partial cosine over *rare* tokens only (df <= MAX_DF) through an inverted
     index = chunked sparse product; keep the top RETRIEVE_N.
  2. re-rank those N by the full cosine (adds the common-token part of the dot
     product) and keep the top RAW_TOPK.
Cost is O(sum of rare posting-list lengths), never O(|S1| x |S2+S3|).

Usage: python blocking.py train|test
"""
import sys
import time
from multiprocessing import Pool

import numpy as np
import polars as pl
import scipy.sparse as sp

import config as C

_WEIGHT = {"n": 1.0, "p": 0.8, "c": 0.6, "a": 0.7, "h": 0.9, "x": 0.6}
MAX_DF = 1000          # posting-list cap for the retrieval index
X_MAX_DF = 300         # tighter cap for the (numerous) cross tokens
PART_MIN = 0.02        # partial-cosine entries below this are dropped before ranking
RETRIEVE_N = 20        # partial-cosine neighbours re-ranked per right record
RAW_TOPK = 5           # neighbours kept after re-ranking
RAW_MIN_SCORE = 0.05   # full-cosine floor
R_CHUNK = 300_000      # right records per matrix build
STEP = 2000            # right records per sparse product
N_WORKERS = max(1, min(C.N_JOBS, 6))


def _pref(p, e):
    return pl.concat_str([pl.lit(p), e])


def tok_long(df):
    """df(li, n_core, a_toks) -> long frame (li, h, w) of hashed blocking tokens."""
    core = pl.col("n_core").list.unique(maintain_order=True)
    c5 = core.list.head(5).list.sort()
    pairs = [pl.concat_str([pl.lit("p|"), c5.list.get(i, null_on_oob=True), pl.lit("_"),
                            c5.list.get(j, null_on_oob=True)])
             for i in range(5) for j in range(i + 1, 5)]
    comp = pl.col("n_core").list.join("")
    pref = pl.when(comp.str.len_chars() >= 4).then(_pref("c|", comp.str.slice(0, 6)))
    e = pl.element()
    isnum = e.str.contains(r"^\d+$")
    hn = pl.col("a_toks").list.eval(
        pl.when(isnum & ~e.shift(-1).str.contains(r"^\d+$"))
          .then(pl.concat_str([pl.lit("h|"), e, pl.lit("_"), e.shift(-1)])))
    toks = pl.concat_list([
        core.list.eval(_pref("n|", e)),
        pl.concat_list(pairs),
        pref,
        pl.col("a_toks").list.unique(maintain_order=True).list.eval(_pref("a|", e)),
        hn,
    ]).list.drop_nulls().list.unique()
    base = df.select("li", toks.alias("tok")).explode("tok").drop_nulls("tok")
    # cross tokens: first two name words x address tokens
    nm = df.select("li", core.list.head(2).alias("nw")).explode("nw").drop_nulls("nw").filter(pl.col("nw") != "")
    ad = df.select("li", pl.col("a_toks").list.unique().alias("t")).explode("t").drop_nulls("t")
    x = nm.join(ad, on="li").select("li", pl.concat_str([pl.lit("x|"), pl.col("nw"), pl.lit("|"), pl.col("t")]).alias("tok"))
    return (pl.concat([base, x]).unique(["li", "tok"])
              .select("li", pl.col("tok").hash(seed=7).alias("h"),
                      pl.col("tok").str.slice(0, 1).replace_strict(_WEIGHT, return_dtype=pl.Float32).alias("w"),
                      (pl.col("tok").str.slice(0, 1) == "x").alias("isx")))


def _to_csr(L, stats, vocabs, n, unseen_idf):
    """Weighted, L2-normalised rows split over several disjoint vocabularies."""
    L = L.join(stats.select("h", "idf"), on="h", how="left").with_columns(
        wt=pl.col("idf").fill_null(unseen_idf).cast(pl.Float32) * pl.col("w"))
    norm = L.group_by("li").agg(nrm=(pl.col("wt") ** 2).sum().sqrt())
    L = L.join(norm, on="li").with_columns(v=(pl.col("wt") / pl.col("nrm")).cast(pl.Float32))
    out = []
    for vocab in vocabs:
        M = L.join(vocab, on="h", how="inner")
        out.append(sp.csr_matrix((M["v"].to_numpy(), (M["li"].to_numpy(), M["col"].to_numpy())),
                                 shape=(n, vocab.height)))
    return out


def vocabularies(L1, n1):
    """IDF table plus the rare (indexed) / common (re-rank only) vocabularies."""
    stats = L1.group_by("h").agg(df=pl.len(), isx=pl.col("isx").first()).with_columns(
        idf=(np.log(n1 + 1) - pl.col("df").cast(pl.Float64).log()))
    rare = pl.col("df") <= pl.when(pl.col("isx")).then(X_MAX_DF).otherwise(MAX_DF)
    v_rare = stats.filter(rare).select("h").with_row_index("col")
    v_common = stats.filter(~rare).select("h").with_row_index("col")
    return stats, v_rare, v_common


_S1RT = None   # rare-token part of S1, transposed  (V_rare x n1)
_S1C = None    # common-token part of S1             (n1 x V_common)


def _winit(p_rare, p_common):
    global _S1RT, _S1C
    _S1RT = sp.load_npz(p_rare).tocsr()
    _S1C = sp.load_npz(p_common).tocsr()


def _empty():
    return np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32), np.empty(0, np.float32), np.empty(0, np.int8)


def _rank_within(rows, score):
    order = np.lexsort((-score, rows))
    rows_s = rows[order]
    rank = np.arange(rows_s.size) - np.searchsorted(rows_s, rows_s, side="left")
    return order, rank


def _topk_chunk(args):
    Rr, Rc, off = args
    P = (Rr @ _S1RT).tocsr()
    rows = np.repeat(np.arange(P.shape[0], dtype=np.int32), np.diff(P.indptr))
    if rows.size == 0:
        return _empty()
    cols, part = P.indices, P.data
    if PART_MIN > 0:
        k = part >= PART_MIN
        rows, cols, part = rows[k], cols[k], part[k]
        if rows.size == 0:
            return _empty()
    order, rank = _rank_within(rows, part)
    m = rank < RETRIEVE_N
    sel = order[m]
    rows, cols, part = rows[sel], cols[sel], part[sel]
    common = np.asarray(Rc[rows].multiply(_S1C[cols]).sum(axis=1)).ravel().astype(np.float32)
    full = part + common
    keep = full >= RAW_MIN_SCORE
    rows, cols, part, full = rows[keep], cols[keep], part[keep], full[keep]
    if rows.size == 0:
        return _empty()
    order, rank = _rank_within(rows, full)
    m = rank < RAW_TOPK
    sel = order[m]
    return rows[sel].astype(np.int64) + off, cols[sel], full[sel], part[sel], rank[m].astype(np.int8)


def block(split):
    t0 = time.time()
    d = C.split_dir(split)
    cols = ["country", "n_core", "a_toks"]
    s1 = pl.read_parquet(d / "s1.parquet", columns=cols).with_row_index("row")
    rsrc = pl.concat([pl.scan_parquet(d / "s2.parquet").select(cols),
                      pl.scan_parquet(d / "s3.parquet").select(cols)]).with_row_index("row")
    out = []
    for country in s1["country"].unique().sort().to_list():
        s1c = s1.filter(pl.col("country") == country).with_row_index("li")
        n1 = s1c.height
        L1 = tok_long(s1c.select("li", "n_core", "a_toks"))
        stats, v_rare, v_common = vocabularies(L1, n1)
        stats = stats.select("h", "idf")
        unseen = float(np.log(n1 + 1))
        S1r, S1c = _to_csr(L1, stats, [v_rare, v_common], n1, unseen)
        del L1
        p_rare, p_common = d / f"_s1r_{country}.npz", d / f"_s1c_{country}.npz"
        sp.save_npz(p_rare, S1r.T.tocsr(), compressed=False)
        sp.save_npz(p_common, S1c, compressed=False)
        print(f"[{country}] s1={n1:,} V_rare={v_rare.height:,} V_common={v_common.height:,} "
              f"nnz rare={S1r.nnz:,} common={S1c.nnz:,}  {time.time() - t0:.0f}s", flush=True)
        del S1r, S1c
        rc = rsrc.filter(pl.col("country") == country).collect()
        s1rows = s1c["row"].to_numpy()
        with Pool(N_WORKERS, initializer=_winit, initargs=(str(p_rare), str(p_common))) as pool:
            for c0 in range(0, rc.height, R_CHUNK):
                part = rc.slice(c0, R_CHUNK).with_row_index("li")
                Rr, Rc = _to_csr(tok_long(part.select("li", "n_core", "a_toks")), stats,
                                 [v_rare, v_common], part.height, unseen)
                jobs = [(Rr[i:i + STEP], Rc[i:i + STEP], i) for i in range(0, part.height, STEP)]
                res = list(pool.imap(_topk_chunk, jobs, chunksize=2))
                ri = np.concatenate([x[0] for x in res]); ci = np.concatenate([x[1] for x in res])
                rrows = part["row"].to_numpy()
                out.append(pl.DataFrame({
                    "rrow": rrows[ri].astype(np.uint32), "s1row": s1rows[ci].astype(np.uint32),
                    "bscore": np.concatenate([x[2] for x in res]),
                    "pscore": np.concatenate([x[3] for x in res]),
                    "brank": np.concatenate([x[4] for x in res])}))
                print(f"[{country}] right {c0 + part.height:,}/{rc.height:,}  pairs so far "
                      f"{sum(o.height for o in out):,}  {time.time() - t0:.0f}s", flush=True)
                del Rr, Rc, jobs, res
        p_rare.unlink()
        p_common.unlink()
        del rc, stats, v_rare, v_common
    cands = pl.concat(out)
    cands.write_parquet(d / "cands_raw.parquet")
    print("total raw candidate pairs", f"{cands.height:,}", f"{time.time() - t0:.0f}s", flush=True)
    return cands


if __name__ == "__main__":
    block(sys.argv[1])

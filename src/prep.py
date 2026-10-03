"""Step 1: read raw TSVs, normalise names/addresses, learn the Indic->English
lexicon from the training ground truth, and write tokenised parquet files.

Usage:  python prep.py            (processes train and test)
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

import polars as pl

import config as C
import textnorm as T

_INDIC_SEG = re.compile(r"[ऀ-෿][ऀ-෿​-‏\s]*")
_LEX = None


def read_source(split, k):
    f = C.DATASET / split / f"{split}_source{k}.tsv"
    df = pl.read_csv(f, separator="\t", quote_char=None, infer_schema=False,
                     columns=["entity_id", "business_name", "business_address", "country"])
    return df.rename({"business_name": "name", "business_address": "addr"}).with_columns(
        pl.col("name").fill_null(""), pl.col("addr").fill_null(""),
        pl.col("country").fill_null("UNK"))


def _init(lex):
    global _LEX
    _LEX = lex


def _norm_chunk(args):
    names, addrs = args
    lex = _LEX
    out = []
    for n, a in zip(names, addrs):
        toks, core = T.name_tokens(n, lex)
        at = T.addr_tokens(a, lex)
        if lex:  # collapse "mp mp" produced by multi-word state names
            at = [t for i, t in enumerate(at) if i == 0 or t != at[i - 1]]
        out.append((toks, core, at, T.has_indic(n)))
    return pl.DataFrame(
        {"n_toks": [r[0] for r in out], "n_core": [r[1] for r in out],
         "a_toks": [r[2] for r in out], "indic": [r[3] for r in out]},
        schema={"n_toks": pl.List(pl.String), "n_core": pl.List(pl.String),
                "a_toks": pl.List(pl.String), "indic": pl.Boolean})


def _jobs(df, chunk):
    for i in range(0, df.height, chunk):
        part = df.slice(i, chunk)
        yield part["name"].to_list(), part["addr"].to_list()


def normalise(df, lex, pool_size=C.N_JOBS, chunk=20000):
    with Pool(pool_size, initializer=_init, initargs=(lex,)) as p:
        res = pl.concat(list(p.imap(_norm_chunk, _jobs(df, chunk), chunksize=1)))
    return pl.concat([df, res], how="horizontal")


def learn_lexicon(s1, rights, gt):
    """Co-occurrence lexicon: transliterated Indic token -> English token.

    For every matched (S1, right) pair whose right record contains Indic
    script, count (right-token, S1-token) co-occurrences; keep the S1 token
    with the best association score for each transliterated token.
    """
    pairs = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
               .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
               .rename({"source1_entity_id": "s1", "matched_entity_ids": "r"}))
    r = rights.filter(pl.col("name").str.contains(r"[ऀ-෿]") |
                      pl.col("addr").str.contains(r"[ऀ-෿]"))
    j = pairs.join(r.select("entity_id", "name", "addr"), left_on="r", right_on="entity_id") \
             .join(s1.select("entity_id", "n_toks", "a_toks"), left_on="s1", right_on="entity_id")
    print("lexicon pairs", j.height, flush=True)
    lex = {}
    for field, s1col in (("name", "n_toks"), ("addr", "a_toks")):
        co = defaultdict(Counter)
        ct, cw = Counter(), Counter()
        for raw, wtoks in zip(j[field].to_list(), j[s1col].to_list()):
            segs = _INDIC_SEG.findall(raw or "")
            if not segs:
                continue
            ttoks = set()
            for sgm in segs:
                ttoks.update(w for w in re.split(r"[^0-9a-z]+", T.strip_accents(T.transliterate(sgm)).lower()) if w)
            wset = set(wtoks)
            for w in wset:
                cw[w] += 1
            for t in ttoks:
                ct[t] += 1
                co[t].update(wset)
        for t, c in co.items():
            if ct[t] < 2:
                continue
            best, bs = None, 0.0
            for w, n in c.most_common(30):
                s = n / (ct[t] * cw[w]) ** 0.5
                if s > bs:
                    best, bs = w, s
            if best is not None and c[best] / ct[t] >= 0.3 and t not in lex:
                lex[t] = best
    print("lexicon size", len(lex), flush=True)
    return lex


def main():
    t0 = time.time()
    lex_path = C.WORK / "lexicon.json"
    if lex_path.exists():
        lex = json.loads(lex_path.read_text(encoding="utf-8"))
    else:
        gt = pl.read_csv(C.DATASET / "train" / "train_ground_truth.tsv", separator="\t",
                         quote_char=None, infer_schema=False)
        s1 = normalise(read_source("train", 1), None)
        ind = pl.col("name").str.contains(r"[ऀ-෿]") | pl.col("addr").str.contains(r"[ऀ-෿]")
        rights = pl.concat([read_source("train", k).filter(ind) for k in (2, 3)])
        lex = learn_lexicon(s1, rights, gt)
        s1 = rights = gt = None
        lex_path.parent.mkdir(parents=True, exist_ok=True)
        lex_path.write_text(json.dumps(lex, ensure_ascii=False), encoding="utf-8")
    print(f"lexicon ready {time.time() - t0:.0f}s", flush=True)
    splits = sys.argv[1:] or ["train", "test"]
    for split in splits:
        d = C.split_dir(split)
        if all((d / f"s{k}.parquet").exists() for k in (1, 2, 3)):
            continue
        for k in (1, 2, 3):
            tmp = d / f"_s{k}_raw.parquet"
            if tmp.exists():
                continue
            df = normalise(read_source(split, k), lex)
            df.write_parquet(tmp)
            print(f"{split} s{k} {df.height} rows  {time.time() - t0:.0f}s", flush=True)
            del df
        noise = noise_vocabulary([d / f"_s{k}_raw.parquet" for k in (1, 2, 3)])
        (d / "noise_tokens.json").write_text(json.dumps(
            {c: t for c, t in noise.group_by("country").agg(pl.col("tok").sort()).iter_rows()}, indent=1))
        print(f"{split}: noise tokens per country",
              noise.group_by("country").agg(pl.len()).rows(), flush=True)
        for k in (1, 2, 3):
            tmp = d / f"_s{k}_raw.parquet"
            remove_noise(pl.read_parquet(tmp), noise).write_parquet(d / f"s{k}.parquet")
            tmp.unlink()
        print(f"{split} done  {time.time() - t0:.0f}s", flush=True)


def noise_vocabulary(paths, min_ratio=3.5, min_frac=0.0005, min_count=300):
    """Name tokens injected by the Source-2/3 noise process: far more frequent in
    Source 2/3 names than in (clean) Source-1 names of the same country label.
    Unsupervised -- uses only the input records, never labels."""
    def doc_freq(path):
        return (pl.scan_parquet(path).select("country", pl.col("n_core").list.unique().alias("tok"))
                  .explode("tok").drop_nulls("tok").group_by("country", "tok").agg(c=pl.len()).collect())
    def sizes(path):
        return pl.scan_parquet(path).group_by("country").agg(n=pl.len()).collect()
    f1 = doc_freq(paths[0]).join(sizes(paths[0]), on="country")
    f23 = pl.concat([doc_freq(p) for p in paths[1:]]).group_by("country", "tok").agg(pl.col("c").sum())
    n23 = pl.concat([sizes(p) for p in paths[1:]]).group_by("country").agg(pl.col("n").sum())
    f23 = f23.join(n23, on="country")
    j = f23.join(f1, on=["country", "tok"], how="left", suffix="_1").with_columns(
        pl.col("c_1").fill_null(0), pl.col("n_1").fill_null(pl.col("n_1").max().over("country")))
    j = j.with_columns(ratio=(pl.col("c") / pl.col("n") + 1e-4) / (pl.col("c_1") / pl.col("n_1") + 1e-4))
    return j.filter((pl.col("ratio") >= min_ratio) & (pl.col("c") >= min_count)
                    & (pl.col("c") >= min_frac * pl.col("n"))).select("country", "tok")


def remove_noise(df, noise):
    """Drop noise tokens from n_core (keeping the original if nothing would remain)."""
    df = df.with_row_index("_i")
    ex = df.select("_i", "country", pl.col("n_core").alias("tok")).with_columns(
        pos=pl.int_ranges(pl.col("tok").list.len())).explode("tok", "pos").drop_nulls("tok")
    ex = ex.join(noise.with_columns(noise=pl.lit(True)), on=["country", "tok"], how="left") \
           .filter(pl.col("noise").is_null()).sort("_i", "pos") \
           .group_by("_i", maintain_order=True).agg(pl.col("tok").alias("core2"))
    df = df.join(ex, on="_i", how="left", maintain_order="left").with_columns(
        n_core=pl.when(pl.col("core2").list.len() > 0).then(pl.col("core2")).otherwise(pl.col("n_core")))
    return df.drop("_i", "core2")


if __name__ == "__main__":
    main()

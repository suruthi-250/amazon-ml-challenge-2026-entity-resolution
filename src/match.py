"""Step 3/4: final candidate selection, feature building, matching model,
one-to-one assignment and submission writing.

Stage A  gradient-boosted trees on pairwise similarity features, two models
         cross-fitted on disjoint halves of the (non-validation) Source-2/3
         records. On validation and test pairs pA is the average of both models.
Stage B  small gradient-boosted model on pA *relative to the other candidates of
         the same Source-2/3 record* (rank, best competitor, margin ...).
         Fitted on the held-out validation pairs, whose pA is distributed exactly
         like test pA (2-fold cross-fitting on validation for threshold tuning).
         Entity-level context (other records linked to the same Source-1 entity)
         helped on validation but did not transfer to test -- test has more
         look-alike "sibling" records per entity -- so it is deliberately not used.
Decision every Source-2/3 record goes to its best Source-1 candidate if
         pB >= threshold (threshold tuned for macro F0.5 on held-out entities).

Usage:
    python match.py train       # features, stage A, stage B, threshold (validation)
    python match.py test        # score test candidates, write output/*.tsv
    python match.py refit_b     # re-fit stage B + threshold from saved validation scores
    python match.py rescore     # re-apply stage B to saved test pA scores (no feature rebuild)
"""
import json
import pickle
import shutil
import sys
import time

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier

import config as C
from features import FEATURES, context_features, pair_features
from gt import gt_pairs

FINAL_K = 3        # keep at most this many S1 neighbours per right record ...
FINAL_MIN = 0.10   # ... whose blocking cosine is at least this ...
FINAL_REL = 0.5    # ... and at least this fraction of the right record's best cosine
VAL_MOD = 10       # 1/VAL_MOD of Source-1 entities form the validation split
A_ROWS_PER_FOLD = 3_000_000
CTX = ["r_ncand", "bscore", "brank"]
FEATURES_B = ["pA", "pa_other", "pa_margin", "pa_rank", "pa_sum_other", "pa_cnt5"] + CTX


def load_frames(split):
    d = C.split_dir(split)
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id", "country", "n_toks", "n_core", "a_toks"]) \
           .with_row_index("row")
    cols = ["entity_id", "n_toks", "n_core", "a_toks", "indic"]
    r = pl.concat([pl.read_parquet(d / "s2.parquet", columns=cols).with_columns(src=pl.lit(2, pl.Int8)),
                   pl.read_parquet(d / "s3.parquet", columns=cols).with_columns(src=pl.lit(3, pl.Int8))]) \
          .with_row_index("row")
    return s1, r


def select_candidates(raw):
    """Final blocking filter -> the exact candidate set the model scores."""
    return (raw.with_columns(rbest=pl.col("bscore").max().over("rrow"))
               .filter((pl.col("brank") < FINAL_K) & (pl.col("bscore") >= FINAL_MIN)
                       & (pl.col("bscore") >= FINAL_REL * pl.col("rbest")))
               .drop("rbest"))


def _chunks(c, chunk):
    """Slices of c (sorted by rrow) that never split a right record's candidates."""
    rr = c["rrow"].to_numpy()
    i = 0
    while i < c.height:
        j = min(i + chunk, c.height)
        while j < c.height and rr[j] == rr[j - 1]:
            j += 1
        yield c.slice(i, j - i), j
        i = j


def to_X(f, cols):
    return f.select(cols).to_numpy().astype(np.float32)


def new_model(iters, leaves, lr=0.1, min_leaf=80):
    return HistGradientBoostingClassifier(
        max_iter=iters, learning_rate=lr, max_leaf_nodes=leaves, min_samples_leaf=min_leaf,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.05,
        n_iter_no_change=40, random_state=C.SEED)


def build_features(c, s1, r, outdir, chunk=1_500_000, log=""):
    """All pair features, written chunk by chunk to parquet (keeps RAM flat)."""
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True)
    t0 = time.time()
    for i, (part, j) in enumerate(_chunks(c.sort("rrow"), chunk)):
        f = pair_features(part, s1, r)
        f.with_columns(pl.col(pl.Float64).cast(pl.Float32)).write_parquet(outdir / f"part{i:03d}.parquet")
        print(f"  {log} features {j:,}/{c.height:,}  {time.time() - t0:.0f}s", flush=True)


def stage2(p):
    """Right-record context of pA (every candidate of a right record is present)."""
    p = p.with_columns(
        pa_rank=pl.col("pA").rank("ordinal", descending=True).over("rrow").cast(pl.Int16),
        pa_max=pl.col("pA").max().over("rrow"),
        pa_2nd=pl.col("pA").top_k(2).min().over("rrow"),
        pa_sum=pl.col("pA").sum().over("rrow"),
        pa_cnt5=(pl.col("pA") > 0.5).sum().over("rrow").cast(pl.Int16),
        n=pl.len().over("rrow"),
    )
    other = pl.when(pl.col("n") == 1).then(0.0) \
              .when(pl.col("pa_rank") == 1).then(pl.col("pa_2nd")).otherwise(pl.col("pa_max"))
    return p.with_columns(pa_other=other).with_columns(
        pa_margin=pl.col("pA") - pl.col("pa_other"),
        pa_sum_other=pl.col("pa_sum") - pl.col("pA"),
    ).drop("pa_max", "pa_2nd", "pa_sum", "n")


def assign(p, thr, col="pB"):
    """Every right record goes to at most one S1: its best candidate, if score >= thr."""
    return (p.sort(col, descending=True).unique("rrow", keep="first")
             .filter(pl.col(col) >= thr).select("s1row", "rrow", col))


def macro_f05(pred, truth, s1rows):
    """pred/truth: (s1row, rrow) pairs; s1rows: evaluation universe of S1 rows."""
    base = pl.DataFrame({"s1row": np.asarray(s1rows, dtype=np.uint32)})
    tp = pred.join(truth, on=["s1row", "rrow"], how="inner").group_by("s1row").agg(tp=pl.len())
    npred = pred.group_by("s1row").agg(npred=pl.len())
    ntrue = truth.group_by("s1row").agg(ntrue=pl.len())
    m = (base.join(tp, on="s1row", how="left").join(npred, on="s1row", how="left")
             .join(ntrue, on="s1row", how="left").fill_null(0))
    P = pl.when(pl.col("npred") > 0).then(pl.col("tp") / pl.col("npred")).otherwise(0.0)
    R = pl.when(pl.col("ntrue") > 0).then(pl.col("tp") / pl.col("ntrue")).otherwise(0.0)
    f = pl.when((pl.col("ntrue") == 0) & (pl.col("npred") == 0)).then(1.0) \
          .when((pl.col("ntrue") == 0) | (pl.col("npred") == 0)).then(0.0) \
          .otherwise(pl.when(P + R > 0).then(1.25 * P * R / (0.25 * P + R)).otherwise(0.0))
    m = m.with_columns(f=f)
    return m["f"].mean(), m


def _is_val(col):
    return (pl.col(col).hash(seed=1) % VAL_MOD) == 0


def fit_stage_b(bv, split="train"):
    """Stage B on validation pairs: 2-fold cross-fit (by right record) gives
    out-of-fold pB for threshold tuning; the final model uses all of them."""
    d = C.split_dir(split)
    X, y = to_X(bv, FEATURES_B), bv["y"].to_numpy()
    fold = (bv["rrow"].hash(seed=11) % 2).to_numpy()
    oof = np.zeros(len(y), dtype=np.float32)
    for k in (0, 1):
        m = new_model(iters=300, leaves=31)
        m.fit(X[fold != k], y[fold != k])
        oof[fold == k] = m.predict_proba(X[fold == k])[:, 1]
    model_b = new_model(iters=300, leaves=31)
    model_b.fit(X, y)
    bv = bv.with_columns(pB=pl.Series(oof))

    s1v = pl.read_parquet(d / "s1.parquet", columns=["entity_id", "country"]).with_row_index("row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    truth = gt_pairs(s1v, r).filter(_is_val("s1row"))
    s1v = s1v.filter(_is_val("row"))
    val_rows = s1v["row"].to_numpy()
    vs = pl.Series(val_rows).cast(pl.UInt32).implode()
    res = {}
    for col in ("pA", "pB"):
        best = (0.0, 0.5)
        for thr in np.arange(0.30, 0.95, 0.02):
            pred = assign(bv, thr, col).filter(pl.col("s1row").is_in(vs))
            fs, _ = macro_f05(pred.select("s1row", "rrow"), truth, val_rows)
            if fs > best[0]:
                best = (fs, float(thr))
        res[col] = best
        print(f"  {col}: best val macro F0.5 = {best[0]:.5f} at thr={best[1]:.2f}", flush=True)
    thr = res["pB"][1]
    pred = assign(bv, thr, "pB").filter(pl.col("s1row").is_in(vs))
    _, m = macro_f05(pred.select("s1row", "rrow"), truth, val_rows)
    m = m.join(s1v.select(pl.col("row").alias("s1row"), "country"), on="s1row")
    print(m.group_by("country").agg(f05=pl.col("f").mean(), n=pl.len()), flush=True)
    bv.write_parquet(d / "pred_val.parquet")
    return model_b, thr, res


def save_models(models_a, model_b, thr, res):
    with open(C.WORK / "models.pkl", "wb") as fh:
        pickle.dump({"A": models_a, "B": model_b}, fh)
    (C.WORK / "model_meta.json").write_text(json.dumps(
        {"threshold": thr, "val_f05": res["pB"][0], "val_f05_stageA": res["pA"][0],
         "features": FEATURES, "features_b": FEATURES_B,
         "final_k": FINAL_K, "final_min": FINAL_MIN, "final_rel": FINAL_REL}, indent=2))


def train(split="train"):
    t0 = time.time()
    s1, r = load_frames(split)
    d = C.split_dir(split)
    c = context_features(select_candidates(pl.read_parquet(d / "cands_raw.parquet")))
    g = gt_pairs(s1, r)
    c = c.join(g.with_columns(y=pl.lit(1, pl.Int8)), on=["s1row", "rrow"], how="left") \
         .with_columns(pl.col("y").fill_null(0))
    print(f"candidates {c.height:,}  per S1 {c.height / s1.height:.2f}  "
          f"pair recall {c['y'].sum() / g.height:.4f}", flush=True)
    val_r = c.filter(_is_val("s1row")).select("rrow").unique().with_columns(val=pl.lit(True))
    c = c.join(val_r, on="rrow", how="left").with_columns(
        pl.col("val").fill_null(False), fold=(pl.col("rrow").hash(seed=3) % 2).cast(pl.Int8))
    fdir = d / "feat"
    build_features(c, s1, r, fdir, log=split)
    s1 = r = c = None          # free RAM before the feature matrix is loaded

    # ---- stage A: cross-fitted on the two halves of the non-validation right records ----
    f = pl.read_parquet(fdir / "*.parquet")
    print(f"feature frame {f.height:,} x {f.width}  {time.time() - t0:.0f}s", flush=True)
    meta = f.select(["s1row", "rrow", "y", "val", "fold"] + CTX)
    X = to_X(f, FEATURES)
    f = None
    y = meta["y"].to_numpy()
    val = meta["val"].to_numpy()
    fold = meta["fold"].to_numpy()
    rng = np.random.default_rng(C.SEED)
    pa_val = np.zeros(int(val.sum()), dtype=np.float32)
    models_a = []
    for k in (0, 1):
        idx = np.flatnonzero(~val & (fold != k))
        if idx.size > A_ROWS_PER_FOLD:
            idx = np.sort(rng.choice(idx, A_ROWS_PER_FOLD, replace=False))
        m = new_model(iters=1500, leaves=127)
        m.fit(X[idx], y[idx])
        pa_val += 0.5 * m.predict_proba(X[val])[:, 1]
        models_a.append(m)
        print(f"stage A fold {k}: {m.n_iter_} iters on {idx.size:,} rows  {time.time() - t0:.0f}s", flush=True)
    X = None
    shutil.rmtree(fdir)

    # ---- stage B on the validation pairs (their pA is distributed like test pA) ----
    bv = stage2(meta.filter(pl.col("val")).with_columns(pA=pl.Series(pa_val)))
    model_b, thr, res = fit_stage_b(bv, split)
    save_models(models_a, model_b, thr, res)
    print(f"done {time.time() - t0:.0f}s")


def refit_b(split="train"):
    """Re-fit stage B + threshold from the saved validation scores (pA)."""
    bv = pl.read_parquet(C.split_dir(split) / "pred_val.parquet", columns=["s1row", "rrow", "y", "pA"] + CTX)
    with open(C.WORK / "models.pkl", "rb") as fh:
        models = pickle.load(fh)
    model_b, thr, res = fit_stage_b(stage2(bv), split)
    save_models(models["A"], model_b, thr, res)


def write_lists(pairs, s1, rids, col, path):
    """pairs(s1row, rrow) -> one row per S1 entity with a comma-joined id list."""
    lists = (pairs.join(rids, left_on="rrow", right_on="row")
                  .group_by("s1row").agg(pl.col("entity_id").unique().sort().str.join(",").alias(col)))
    out = (s1.select(pl.col("row").alias("s1row"), pl.col("entity_id").alias("source1_entity_id"))
             .join(lists, on="s1row", how="left", maintain_order="left")
             .select("source1_entity_id", pl.col(col).fill_null("")))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def _finish(p, split, thr, models, meta):
    """stage B + assignment + matching_results.tsv"""
    d = C.split_dir(split)
    p = stage2(p)
    p = p.with_columns(pB=pl.Series(models["B"].predict_proba(to_X(p, meta["features_b"]))[:, 1]))
    p.select("s1row", "rrow", "pA", "pB").write_parquet(d / "pred.parquet")
    thr = float(thr) if thr is not None else meta["threshold"]
    m = assign(p, thr, "pB")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    rids = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    out = write_lists(m.select("s1row", "rrow"), s1, rids, "matched_entity_ids", C.OUTPUT / "matching_results.tsv")
    print(f"threshold {thr:.2f}: matches {m.height:,}; S1 with >=1 match "
          f"{(out['matched_entity_ids'] != '').sum():,}/{out.height:,}")


def _load_models():
    meta = json.loads((C.WORK / "model_meta.json").read_text())
    with open(C.WORK / "models.pkl", "rb") as fh:
        return pickle.load(fh), meta


def predict(split="test", thr=None):
    t0 = time.time()
    models, meta = _load_models()
    s1, r = load_frames(split)
    d = C.split_dir(split)
    c = context_features(select_candidates(pl.read_parquet(d / "cands_raw.parquet"))).sort("rrow")
    print(f"{split} candidates {c.height:,}  per S1 {c.height / s1.height:.2f}", flush=True)
    write_lists(c.select("s1row", "rrow"), s1, r.select("row", "entity_id"), "candidate_entity_ids",
                C.OUTPUT / "candidate_pairs.tsv")
    ps = []
    for part, j in _chunks(c, 1_500_000):
        f = pair_features(part, s1, r)
        X = to_X(f, meta["features"])
        pa = np.mean([m.predict_proba(X)[:, 1] for m in models["A"]], axis=0)
        ps.append(f.select(["s1row", "rrow"] + CTX).with_columns(pA=pl.Series(pa)))
        print(f"  scored {j:,}/{c.height:,}  {time.time() - t0:.0f}s", flush=True)
        f = X = None
    s1 = r = c = None
    _finish(pl.concat(ps), split, thr, models, meta)
    print(f"done {time.time() - t0:.0f}s")


def rescore(split="test", thr=None):
    """Re-apply stage B to saved test pA (work/<split>/pred.parquet) -- no feature rebuild."""
    models, meta = _load_models()
    d = C.split_dir(split)
    c = context_features(select_candidates(pl.read_parquet(d / "cands_raw.parquet"))).select(["s1row", "rrow"] + CTX)
    p = pl.read_parquet(d / "pred.parquet", columns=["s1row", "rrow", "pA"]).join(c, on=["s1row", "rrow"])
    _finish(p, split, thr, models, meta)


if __name__ == "__main__":
    {"train": train, "test": predict, "refit_b": refit_b, "rescore": rescore}[sys.argv[1]](*sys.argv[2:])

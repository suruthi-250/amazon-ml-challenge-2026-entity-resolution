"""v8: v5 + TWO cross-encoder scores (first run + second run) in the record-context stage B.
Same protocol as combine_transformer.py: stage B fitted on validation pairs (2-fold OOF for evaluation),
transformer used only for country labels seen in training, v5 kept for the rest.

Usage:
    python combine_ensemble.py <dir_with_second_run_scores> eval     # validation comparison vs v7T only
    python combine_ensemble.py <dir_with_second_run_scores> build    # + write test submission to work/v8_out
The second-run directory must contain val_scores_e1.parquet and test_scores_e1.parquet (or *_e2).
"""
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

import config as C
import match as M
from features import context_features
from gt import gt_pairs

W = C.WORK
OUT = Path(os.environ.get("ER_OUTPUT_DIR", W / "v8_out"))


def ctx(p, col, pre):
    p = p.with_columns(
        (pl.col(col).rank("ordinal", descending=True).over("rrow").cast(pl.Int16)).alias(pre + "_rank"),
        pl.col(col).max().over("rrow").alias("_mx"), pl.col(col).top_k(2).min().over("rrow").alias("_2"),
        pl.len().over("rrow").alias("_n"))
    other = pl.when(pl.col("_n") == 1).then(0.0).when(pl.col(pre + "_rank") == 1).then(pl.col("_2")).otherwise(pl.col("_mx"))
    return p.with_columns(other.alias(pre + "_other")).with_columns((pl.col(col) - pl.col(pre + "_other")).alias(pre + "_margin")) \
            .drop("_mx", "_2", "_n")


def scores(path, name):
    return pl.read_parquet(path).select("s1row", "rrow", pl.col("pT").cast(pl.Float32).alias(name))


def pick(dir2, split):
    for e in ("e2", "e1"):
        f = Path(dir2) / f"{split}_scores_{e}.parquet"
        if f.exists():
            return f
    raise FileNotFoundError(f"no {split}_scores_e1/e2 in {dir2}")


FB1 = M.FEATURES_B + ["pT", "pt_rank", "pt_other", "pt_margin"]
FB2 = FB1 + ["pT2", "pt2_rank", "pt2_other", "pt2_margin", "pT_mean"]


def prepare(p, dir2, split):
    p = p.join(scores(W / "gpu_score" / f"{split}_scores.parquet", "pT"), on=["s1row", "rrow"]) \
         .join(scores(pick(dir2, split), "pT2"), on=["s1row", "rrow"])
    p = ctx(ctx(p, "pT", "pt"), "pT2", "pt2")
    return p.with_columns(pT_mean=(pl.col("pT") + pl.col("pT2")) / 2)


def main():
    dir2, mode = sys.argv[1], sys.argv[2]
    d = C.split_dir("train")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    truth = gt_pairs(s1, r).filter(M._is_val("s1row"))
    rows = s1.filter(M._is_val("row"))["row"].to_numpy()
    bv = prepare(pl.read_parquet(W / "v5_out" / "pred_val.parquet"), dir2, "val")
    print(f"validation pairs with both transformer scores: {bv.height:,}")
    fold = (bv["rrow"].hash(seed=11) % 2).to_numpy(); y = bv["y"].to_numpy()
    for name, fb in (("pB7", FB1), ("pB8", FB2)):
        X = M.to_X(bv, fb); oof = np.zeros(len(y), dtype=np.float32)
        for k in (0, 1):
            m = M.new_model(iters=300, leaves=31); m.fit(X[fold != k], y[fold != k])
            oof[fold == k] = m.predict_proba(X[fold == k])[:, 1]
        bv = bv.with_columns(pl.Series(name, oof))

    def macro(col, t, rr):
        vv = pl.Series(rr).cast(pl.UInt32).implode()
        return M.macro_f05(M.assign(bv, t, col).filter(pl.col("s1row").is_in(vv)).select("s1row", "rrow"),
                           truth.filter(pl.col("s1row").is_in(vv)), rr)

    grid = np.arange(0.56, 0.92, 0.02)
    t7 = max(grid, key=lambda t: macro("pB7", t, rows)[0]); t8 = max(grid, key=lambda t: macro("pB8", t, rows)[0])
    f7, m7 = macro("pB7", t7, rows); f8, m8 = macro("pB8", t8, rows)
    print(f"v7T (1 transformer)  thr {t7:.2f}: {f7:.5f}")
    print(f"v8  (2 transformers) thr {t8:.2f}: {f8:.5f}   delta {(f8 - f7) * 100:+.3f} points")
    half = (pl.Series(rows).hash(seed=41) % 2).to_numpy()
    for a in (0, 1):
        ra, rb = rows[half == a], rows[half != a]
        a7 = max(grid, key=lambda t: macro("pB7", t, ra)[0]); a8 = max(grid, key=lambda t: macro("pB8", t, ra)[0])
        print(f"  split-half {a}: v7T {macro('pB7', a7, rb)[0]:.5f}  v8 {macro('pB8', a8, rb)[0]:.5f}")
    q = (pl.Series(rows).hash(seed=99) % 4).to_numpy()
    wins = 0
    for k in range(4):
        rk = rows[q == k]; x7 = macro("pB7", t7, rk)[0]; x8 = macro("pB8", t8, rk)[0]; wins += x8 > x7
        print(f"  holdout {k + 1}: v7T {x7:.5f}  v8 {x8:.5f}  delta {(x8 - x7) * 100:+.3f}")
    dd = m7.select("s1row", "f").join(m8.select("s1row", pl.col("f").alias("f8")), on="s1row")
    diff = (dd["f8"] - dd["f"]).to_numpy(); rng = np.random.default_rng(0)
    lo, hi = np.percentile([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)], [2.5, 97.5])
    ok = (f8 - f7 >= 0.001) and wins == 4 and lo > 0
    print(f"paired bootstrap v8 - v7T: {diff.mean() * 100:+.3f} points, 95% CI [{lo * 100:+.3f}, {hi * 100:+.3f}]  "
          f"-> GATE {'PASS' if ok else 'FAIL'} (needs >= +0.1 point, 4/4 holdouts, CI > 0)")
    if mode != "build":
        return
    # ---- test file
    OUT.mkdir(parents=True, exist_ok=True)
    model_b = M.new_model(iters=300, leaves=31); model_b.fit(M.to_X(bv, FB2), y)
    with open(OUT / "v8_stage_b.pkl", "wb") as fh:
        pickle.dump(model_b, fh)
    (OUT / "v8_meta.json").write_text(json.dumps({"threshold": float(t8), "features_b": FB2, "val_f05": f8, "gate": ok}, indent=2))
    dt = C.split_dir("test")
    cx = context_features(M.select_candidates(pl.read_parquet(dt / "cands_raw.parquet"))).select(["s1row", "rrow"] + M.CTX)
    p = pl.read_parquet(W / "v5_out" / "pred_test.parquet", columns=["s1row", "rrow", "pA"]).join(cx, on=["s1row", "rrow"])
    p = prepare(M.stage2(p), dir2, "test")
    assert p.height == cx.height, (p.height, cx.height)
    p = p.with_columns(pB=pl.Series(model_b.predict_proba(M.to_X(p, FB2))[:, 1]))
    seen = pl.read_parquet(d / "s1.parquet", columns=["country"])["country"].unique().to_list()
    cty = pl.read_parquet(dt / "s1.parquet", columns=["country"]).with_row_index("s1row")
    p = p.join(cty, on="s1row")
    v5p = pl.read_parquet(W / "v5_out" / "pred_test.parquet", columns=["s1row", "rrow", "pB"]).join(cty, on="s1row")
    thr5 = json.loads((W / "v5_out" / "model_meta.json").read_text())["threshold"]
    m = pl.concat([M.assign(p.filter(pl.col("country").is_in(seen)), t8, "pB").select("s1row", "rrow"),
                   M.assign(v5p.filter(~pl.col("country").is_in(seen)), thr5, "pB").select("s1row", "rrow")])
    s1t = pl.read_parquet(dt / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    rids = pl.concat([pl.read_parquet(dt / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    out = M.write_lists(m.select("s1row", "rrow"), s1t, rids, "matched_entity_ids", OUT / "matching_results.tsv")
    shutil.copy(W / "v5_out" / "candidate_pairs.tsv", OUT / "candidate_pairs.tsv")
    print(f"v8 test file: {m.height:,} links -> {OUT}")


if __name__ == "__main__":
    main()

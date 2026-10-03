"""Precision / recall / F0.5 on the held-out validation split of the training data
(the test set has no labels, so this is the only local estimate).

Uses the validation predictions saved by `python match.py train`.

Usage: python evaluate.py [threshold]
"""
import json
import sys

import polars as pl

import config as C
import match as M
from gt import gt_pairs


def main():
    meta = json.loads((C.WORK / "model_meta.json").read_text())
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else meta["threshold"]
    d = C.split_dir("train")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id", "country"]).with_row_index("row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    g = gt_pairs(s1, r)
    val = s1.filter(M._is_val("row"))
    val_rows = val["row"].to_numpy()
    vs = pl.Series(val_rows).cast(pl.UInt32).implode()
    truth = g.filter(M._is_val("s1row"))
    pred = M.assign(pl.read_parquet(d / "pred_val.parquet"), thr, "pB").filter(pl.col("s1row").is_in(vs))
    f, m = M.macro_f05(pred.select("s1row", "rrow"), truth, val_rows)
    m = m.with_columns(
        P=pl.when(pl.col("npred") > 0).then(pl.col("tp") / pl.col("npred")),
        R=pl.when(pl.col("ntrue") > 0).then(pl.col("tp") / pl.col("ntrue")))
    tp, npred, ntrue = m["tp"].sum(), m["npred"].sum(), m["ntrue"].sum()
    print(f"validation split: {len(val_rows):,} Source-1 entities, threshold {thr:.2f}")
    print(f"  macro F0.5            : {f:.4f}   (the leaderboard metric)")
    print(f"  macro precision       : {m['P'].mean():.4f}   (over entities with >=1 predicted match)")
    print(f"  macro recall          : {m['R'].mean():.4f}   (over entities with >=1 true match)")
    print(f"  pair-level precision  : {tp / npred:.4f}   ({tp:,} correct of {npred:,} predicted links)")
    print(f"  pair-level recall     : {tp / ntrue:.4f}   ({tp:,} found of {ntrue:,} true links)")
    print(f"  singletons correctly left empty: "
          f"{((m['ntrue'] == 0) & (m['npred'] == 0)).sum():,} / {(m['ntrue'] == 0).sum():,}")
    by = m.join(val.select(pl.col("row").alias("s1row"), "country"), on="s1row").group_by("country").agg(
        f05=pl.col("f").mean(), precision=pl.col("P").mean(), recall=pl.col("R").mean(), n=pl.len()).sort("country")
    print(by)


if __name__ == "__main__":
    main()

"""Segment-specific decision thresholds, tuned without fooling ourselves:
tune on one half of the validation entities, measure on the other half (and swap).

Segments are record-level properties of the chosen pair (stable between train and test):
  address situation of the right record (no address / no house number / same number / different number)
  x whether the core names are identical.

Usage: python segment_thresholds.py
"""
import json

import numpy as np
import polars as pl

import config as C
import match as M
from gt import gt_pairs

NUM = pl.element().filter(pl.element().str.contains(r"^\d+$"))
GRID = [round(x, 2) for x in np.arange(0.40, 0.97, 0.04)]


def segments(pairs, split):
    d = C.split_dir(split)
    s1 = pl.read_parquet(d / "s1.parquet", columns=["n_core", "a_toks"]).with_row_index("s1row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["n_core", "a_toks"]) for k in (2, 3)]).with_row_index("rrow")
    x = pairs.join(s1, on="s1row").join(r, on="rrow", suffix="_b")
    x = x.with_columns(ha=pl.col("a_toks").list.eval(NUM).list.first(), hb=pl.col("a_toks_b").list.eval(NUM).list.first())
    addr = (pl.when(pl.col("a_toks_b").list.len() == 0).then(0)
              .when(pl.col("ha").is_null() | pl.col("hb").is_null()).then(1)
              .when(pl.col("ha") == pl.col("hb")).then(2).otherwise(3))
    same = (pl.col("n_core").list.sort() == pl.col("n_core_b").list.sort()).cast(pl.Int8)
    return x.select(pairs.columns + [(addr * 2 + same).cast(pl.Int8).alias("seg")])


def decide(p, thr_by_seg):
    best = p.sort("pB", descending=True).unique("rrow", keep="first")
    t = pl.Series([thr_by_seg[s] for s in range(8)])
    return best.with_columns(t=pl.col("seg").map_elements(lambda s: thr_by_seg[s], return_dtype=pl.Float64)) \
               .filter(pl.col("pB") >= pl.col("t")).select("s1row", "rrow")


def score(p, truth, rows, thr):
    vs = pl.Series(rows).cast(pl.UInt32).implode()
    pred = decide(p, thr).filter(pl.col("s1row").is_in(vs))
    return M.macro_f05(pred, truth.filter(pl.col("s1row").is_in(vs)), rows)[0]


def tune(p, truth, rows, base):
    thr = {s: base for s in range(8)}
    for _ in range(2):
        for s in range(8):
            best = (score(p, truth, rows, thr), thr[s])
            for g in GRID:
                trial = dict(thr); trial[s] = g
                f = score(p, truth, rows, trial)
                if f > best[0] + 1e-6:
                    best = (f, g)
            thr[s] = best[1]
    return thr


def main():
    base = json.loads((C.WORK / "model_meta.json").read_text())["threshold"]
    d = C.split_dir("train")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    truth = gt_pairs(s1, r).filter(M._is_val("s1row"))
    rows = s1.filter(M._is_val("row"))["row"].to_numpy()
    half = (pl.Series(rows).hash(seed=21) % 2).to_numpy()
    p = segments(pl.read_parquet(d / "pred_val.parquet", columns=["s1row", "rrow", "pB"]), "train")
    names = ["no addr/name diff", "no addr/name same", "no number/name diff", "no number/name same",
             "same number/name diff", "same number/name same", "diff number/name diff", "diff number/name same"]
    print(p.group_by("seg").agg(pl.len()).sort("seg"))
    glob = {s: base for s in range(8)}
    for a, b in ((0, 1), (1, 0)):
        ra, rb = rows[half == a], rows[half == b]
        thr = tune(p, truth, ra, base)
        fb_glob, fb_seg = score(p, truth, rb, glob), score(p, truth, rb, thr)
        print(f"\ntuned on half {a}, measured on half {b}: global {fb_glob:.5f} -> segment {fb_seg:.5f} "
              f"({(fb_seg - fb_glob) * 100:+.3f} points)")
        for s in range(8):
            print(f"   {names[s]:24s} thr {thr[s]:.2f}")


if __name__ == "__main__":
    main()

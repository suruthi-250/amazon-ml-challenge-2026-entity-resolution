"""Second, larger training sample for the cross-encoder that over-represents hard cases.
Writes work/gpu_export2/train_pairs_v2.parquet (s1row, rrow, label) -- non-validation training candidates only.

  hard negatives : wrong candidates that blocking ranked first or scored high (sibling look-alikes)
  hard positives : true matches that blocking ranked low or scored weak
  plus a random sample of everything else
"""
import polars as pl

import config as C
import match as M
from gt import gt_pairs

OUT = C.WORK / "gpu_export2"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = C.split_dir("train")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    g = gt_pairs(s1, r).with_columns(label=pl.lit(1, pl.Int8))
    c = M.select_candidates(pl.read_parquet(d / "cands_raw.parquet")).select("s1row", "rrow", "bscore", "brank") \
         .join(g, on=["s1row", "rrow"], how="left").with_columns(pl.col("label").fill_null(0))
    val_r = c.filter(M._is_val("s1row")).select("rrow").unique()
    tr = c.join(val_r, on="rrow", how="anti")
    hard_neg = tr.filter((pl.col("label") == 0) & ((pl.col("brank") == 0) | (pl.col("bscore") >= 0.5)))
    hard_pos = tr.filter((pl.col("label") == 1) & ((pl.col("brank") > 0) | (pl.col("bscore") < 0.5)))
    print(f"available: hard negatives {hard_neg.height:,}, hard positives {hard_pos.height:,}, all {tr.height:,}")
    parts = [hard_neg.sample(min(1_200_000, hard_neg.height), seed=1),
             hard_pos.sample(min(800_000, hard_pos.height), seed=2),
             tr.filter(pl.col("label") == 1).sample(1_000_000, seed=3),
             tr.filter(pl.col("label") == 0).sample(1_000_000, seed=4)]
    out = pl.concat(parts).unique(["s1row", "rrow"]).sample(fraction=1.0, shuffle=True, seed=5).select("s1row", "rrow", "label")
    out.write_parquet(OUT / "train_pairs_v2.parquet")
    print(f"train_pairs_v2: {out.height:,} pairs, positives {int(out['label'].sum()):,} "
          f"({(OUT / 'train_pairs_v2.parquet').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

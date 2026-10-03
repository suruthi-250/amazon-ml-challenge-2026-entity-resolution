"""Ground-truth helpers (training split only)."""
import polars as pl

import config as C


def gt_pairs(s1, r):
    """Long-format ground truth as (s1row, rrow) row indices."""
    gt = pl.read_csv(C.DATASET / "train" / "train_ground_truth.tsv", separator="\t",
                     quote_char=None, infer_schema=False)
    g = (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
           .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != "")
           .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "r_id"}))
    return (g.join(s1.select(pl.col("entity_id").alias("s1_id"), pl.col("row").alias("s1row")), on="s1_id")
             .join(r.select(pl.col("entity_id").alias("r_id"), pl.col("row").alias("rrow")), on="r_id")
             .select("s1row", "rrow"))


def block_report(cands, s1, r):
    g = gt_pairs(s1, r).with_columns(y=pl.lit(1, pl.Int8))
    j = cands.join(g, on=["s1row", "rrow"], how="left")
    hit = j["y"].sum()
    print(f"pairs={cands.height:,}  per S1={cands.height / s1.height:.2f}  "
          f"pair recall={hit / g.height:.4f}  ({hit:,}/{g.height:,})")
    top1 = j.filter(pl.col("brank") == 0)["y"].sum()
    print(f"recall at rank0={top1 / g.height:.4f}")
    cty = g.join(s1.select("row", "country"), left_on="s1row", right_on="row") \
           .join(cands.select("s1row", "rrow", pl.lit(1).alias("hit")), on=["s1row", "rrow"], how="left")
    print(cty.group_by("country").agg(recall=pl.col("hit").fill_null(0).mean(), n=pl.len()))
    return j

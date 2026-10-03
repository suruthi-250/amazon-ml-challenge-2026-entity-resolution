"""Recall / error share per error archetype on the validation split.

Each true link (and each predicted link) is described by simple, interpretable
properties of the pair: is the address missing, do house numbers agree, is the
name identical / a typo / very different, is the name in an Indic script.
Shows where the remaining recall and precision losses concentrate.

Usage: python error_archetypes.py [threshold]
"""
import json
import sys

import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.process import cpdist

import config as C
import match as M
from gt import gt_pairs

NUM = pl.element().filter(pl.element().str.contains(r"^\d+$"))


def describe(pairs, s1, r):
    x = (pairs.join(s1, left_on="s1row", right_on="row").join(r, left_on="rrow", right_on="row", suffix="_b"))
    na = x["n_core"].list.join(" ").to_list(); nb = x["n_core_b"].list.join(" ").to_list()
    x = x.with_columns(nsim=pl.Series(cpdist(na, nb, scorer=fuzz.token_sort_ratio, workers=-1)))
    x = x.with_columns(ha=pl.col("a_toks").list.eval(NUM).list.first(), hb=pl.col("a_toks_b").list.eval(NUM).list.first())
    name = (pl.when(pl.col("nsim") >= 99.5).then(pl.lit("name identical"))
              .when(pl.col("nsim") >= 75).then(pl.lit("name typo/reorder"))
              .when(pl.col("nsim") >= 45).then(pl.lit("name partial"))
              .otherwise(pl.lit("name very different")))
    addr = (pl.when(pl.col("a_toks_b").list.len() == 0).then(pl.lit("no address"))
              .when(pl.col("ha").is_null() | pl.col("hb").is_null()).then(pl.lit("house no. missing"))
              .when(pl.col("ha") == pl.col("hb")).then(pl.lit("house no. same"))
              .otherwise(pl.lit("house no. differs")))
    return x.with_columns(name=name, addr=addr, script=pl.when(pl.col("indic")).then(pl.lit("indic")).otherwise(pl.lit("latin")))


def main():
    meta = json.loads((C.WORK / "model_meta.json").read_text())
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else meta["threshold"]
    d = C.split_dir("train")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id", "n_core", "a_toks"]).with_row_index("row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id", "n_core", "a_toks", "indic"]) for k in (2, 3)]).with_row_index("row")
    truth = gt_pairs(s1, r).filter(M._is_val("s1row"))
    vs = pl.Series(s1.filter(M._is_val("row"))["row"].to_numpy()).cast(pl.UInt32).implode()
    bv = pl.read_parquet(d / "pred_val.parquet", columns=["s1row", "rrow", "pB", "y"])
    pred = M.assign(bv, thr, "pB").filter(pl.col("s1row").is_in(vs)).select("s1row", "rrow")
    s1s, rs = s1.drop("entity_id"), r.drop("entity_id")
    t = describe(truth, s1s, rs).join(pred.with_columns(found=pl.lit(1)), on=["s1row", "rrow"], how="left") \
            .join(bv.select("s1row", "rrow", pl.lit(1).alias("cand")), on=["s1row", "rrow"], how="left") \
            .with_columns(pl.col("found").fill_null(0), pl.col("cand").fill_null(0))
    n = t.height
    print(f"threshold {thr:.2f}: {n:,} true links, recall {t['found'].mean():.4f}\n")
    pl.Config.set_tbl_rows(40)
    for keys in (["name"], ["addr"], ["script"], ["name", "addr"]):
        g = t.group_by(keys).agg(links=pl.len(), recall=pl.col("found").mean(), in_candidates=pl.col("cand").mean(),
                                 missed=(1 - pl.col("found")).sum())
        print(g.with_columns(share_of_all_misses=pl.col("missed") / (n - t["found"].sum())).sort("missed", descending=True))
    p = describe(pred, s1s, rs).join(truth.with_columns(ok=pl.lit(1)), on=["s1row", "rrow"], how="left") \
            .with_columns(pl.col("ok").fill_null(0))
    print(f"\npredicted links {p.height:,}, wrong {int((p['ok'] == 0).sum()):,}")
    g = p.group_by(["name", "addr"]).agg(links=pl.len(), precision=pl.col("ok").mean(), wrong=(1 - pl.col("ok")).sum())
    print(g.with_columns(share_of_wrong=pl.col("wrong") / pl.col("wrong").sum()).sort("wrong", descending=True).head(10))


if __name__ == "__main__":
    main()

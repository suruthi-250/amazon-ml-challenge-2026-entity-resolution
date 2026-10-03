"""v7T: combine v5 (stage A probability) with the GPU cross-encoder score in a record-context
stage B, fitted on the validation pairs, then write the test submission.

Inputs : v5 outputs of `python run_all.py` -- work/train/pred_val.parquet, work/test/pred.parquet,
         work/model_meta.json, output/candidate_pairs.tsv -- plus work/gpu_score/{val,test}_scores.parquet
         (cross-encoder scores from gpu_crossencoder.py) and work/test/cands_raw.parquet
Output : <ER_OUTPUT_DIR or work/v7t_out>/matching_results.tsv (+ candidate_pairs.tsv copy), v7t model files

Usage: python combine_transformer.py [threshold]
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

W = C.WORK
FB = M.FEATURES_B + ["pT", "pt_rank", "pt_other", "pt_margin"]
OUT = Path(os.environ.get("ER_OUTPUT_DIR", W / "v7t_out"))
THR_DEFAULT = 0.72   # tuned for macro F0.5 on the validation entities (cross-fitted stage B)


def v5(name, standard):
    """v5 artefact: standard pipeline location, or the work/v5_out backup if one was made."""
    backup = W / "v5_out" / name
    return backup if backup.exists() else standard


PRED_VAL = v5("pred_val.parquet", W / "train" / "pred_val.parquet")
PRED_TEST = v5("pred_test.parquet", W / "test" / "pred.parquet")
META_V5 = v5("model_meta.json", W / "model_meta.json")
CANDS = v5("candidate_pairs.tsv", C.OUTPUT / "candidate_pairs.tsv")


def pt_context(p):
    p = p.with_columns(
        pt_rank=pl.col("pT").rank("ordinal", descending=True).over("rrow").cast(pl.Int16),
        pt_max=pl.col("pT").max().over("rrow"), pt_2nd=pl.col("pT").top_k(2).min().over("rrow"), n=pl.len().over("rrow"))
    other = pl.when(pl.col("n") == 1).then(0.0).when(pl.col("pt_rank") == 1).then(pl.col("pt_2nd")).otherwise(pl.col("pt_max"))
    return p.with_columns(pt_other=other).with_columns(pt_margin=pl.col("pT") - pl.col("pt_other")).drop("pt_max", "pt_2nd", "n")


def load_pt(name):
    return pl.read_parquet(W / "gpu_score" / name).with_columns(pl.col("pT").cast(pl.Float32))


def main():
    saved = W / "gpu_score" / "v7t_meta.json"
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else (
        json.loads(saved.read_text())["threshold"] if saved.exists() else THR_DEFAULT)
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- fit the final stage B on all validation pairs
    bv = pt_context(pl.read_parquet(PRED_VAL).join(load_pt("val_scores.parquet"), on=["s1row", "rrow"]))
    model_b = M.new_model(iters=300, leaves=31)
    model_b.fit(M.to_X(bv, FB), bv["y"].to_numpy())
    with open(OUT / "v7t_stage_b.pkl", "wb") as fh:
        pickle.dump(model_b, fh)
    (OUT / "v7t_meta.json").write_text(json.dumps({"threshold": thr, "features_b": FB}, indent=2))
    # ---- test
    d = C.split_dir("test")
    ctx = context_features(M.select_candidates(pl.read_parquet(d / "cands_raw.parquet"))).select(["s1row", "rrow"] + M.CTX)
    p = (pl.read_parquet(PRED_TEST, columns=["s1row", "rrow", "pA"])
           .join(ctx, on=["s1row", "rrow"]).join(load_pt("test_scores.parquet"), on=["s1row", "rrow"]))
    assert p.height == ctx.height, (p.height, ctx.height)
    p = pt_context(M.stage2(p))
    p = p.with_columns(pB=pl.Series(model_b.predict_proba(M.to_X(p, FB))[:, 1]))
    p.select("s1row", "rrow", "pA", "pT", "pB").write_parquet(OUT / "pred_test.parquet")
    # The cross-encoder only saw the country labels of the training data. For any other label it is
    # uncalibrated (it over-links look-alike businesses), so those records keep v5's decisions unchanged.
    seen = pl.read_parquet(C.split_dir("train") / "s1.parquet", columns=["country"])["country"].unique().to_list()
    cty = pl.read_parquet(d / "s1.parquet", columns=["country"]).with_row_index("s1row")
    p = p.join(cty, on="s1row")
    v5p = pl.read_parquet(PRED_TEST, columns=["s1row", "rrow", "pB"]).join(cty, on="s1row")
    thr5 = json.loads(META_V5.read_text())["threshold"]
    m = pl.concat([M.assign(p.filter(pl.col("country").is_in(seen)), thr, "pB").select("s1row", "rrow"),
                   M.assign(v5p.filter(~pl.col("country").is_in(seen)), thr5, "pB").select("s1row", "rrow")])
    print(f"transformer used for country labels seen in training {sorted(seen)}; v5 kept for the rest")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    rids = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    out = M.write_lists(m.select("s1row", "rrow"), s1, rids, "matched_entity_ids", OUT / "matching_results.tsv")
    if CANDS.resolve() != (OUT / "candidate_pairs.tsv").resolve():
        shutil.copy(CANDS, OUT / "candidate_pairs.tsv")
    print(f"v7T threshold {thr:.2f}: {m.height:,} links; S1 with >=1 match {(out['matched_entity_ids'] != '').sum():,}/{out.height:,} -> {OUT}")


if __name__ == "__main__":
    main()

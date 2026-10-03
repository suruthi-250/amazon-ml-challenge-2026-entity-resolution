"""Pre-upload safety check for a matching_results.tsv.

1. Local score: macro F0.5 / precision / recall of the current model on the held-out
   validation entities (the test set has no labels, so this is the only local score).
2. Shift check: compares the candidate file with a reference file that already has a
   leaderboard score. Big changes in link counts per country were what gave v2 its
   drop (+4 % links -> 0.964 -> 0.949), so large deviations are flagged.

Usage: python check_score.py <candidate matching_results.tsv> [reference.tsv]
"""
import subprocess
import sys
from pathlib import Path

import polars as pl

import config as C

REF_DEFAULT = C.WORK / "v4_out" / "matching_results.tsv"   # leaderboard 0.966


def links(path):
    m = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False) \
          .with_columns(pl.col("matched_entity_ids").fill_null(""))
    return (m.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "")
             .rename({"matched_entity_ids": "rid", "source1_entity_id": "sid"}))


def main():
    cand = Path(sys.argv[1])
    ref = Path(sys.argv[2]) if len(sys.argv) > 2 else REF_DEFAULT
    print("== 1. validation score of the current model (python evaluate.py)")
    subprocess.run([sys.executable, "-W", "ignore", str(Path(__file__).with_name("evaluate.py"))], check=False)

    print(f"\n== 2. shift check vs reference {ref}")
    s1 = pl.read_csv(C.DATASET / "test" / "test_source1.tsv", separator="\t", quote_char=None,
                     infer_schema=False, columns=["entity_id", "country"]).rename({"entity_id": "sid"})
    a, b = links(cand).join(s1, on="sid"), links(ref).join(s1, on="sid")
    ca = a.group_by("country").agg(cand=pl.len())
    cb = b.group_by("country").agg(ref=pl.len())
    t = ca.join(cb, on="country", how="full", coalesce=True).fill_null(0).with_columns(
        change=(pl.col("cand") / pl.col("ref") - 1) * 100).sort("country")
    print(t)
    same = a.join(b, on=["sid", "rid"], how="inner").height
    print(f"links: candidate {a.height:,}  reference {b.height:,}  shared {same:,}  "
          f"only-candidate {a.height - same:,}  only-reference {b.height - same:,}")
    worst = t["change"].abs().max()
    churn = (a.height - same + b.height - same) / b.height * 100
    print(f"max per-country change {worst:.2f} %, churn {churn:.2f} %")
    if worst > 2.0 or churn > 4.0:
        print("WARNING: large shift vs the scored reference -- v2 (0.949) looked like this. Investigate before uploading.")
    else:
        print("OK: modest change vs the scored reference.")


if __name__ == "__main__":
    main()

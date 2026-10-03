"""Re-write matching_results.tsv at another threshold from the saved test scores
(work/test/pred.parquet written by `python match.py test`) -- takes about a minute.

Usage: python rethreshold.py <threshold> [output_dir]
"""
import sys
from pathlib import Path

import polars as pl

import config as C
from match import assign, write_lists


def main():
    thr = float(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else C.OUTPUT
    d = C.split_dir("test")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    rids = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    p = pl.read_parquet(d / "pred.parquet", columns=["s1row", "rrow", "pB"])
    m = assign(p, thr, "pB")
    res = write_lists(m.select("s1row", "rrow"), s1, rids, "matched_entity_ids", out / "matching_results.tsv")
    print(f"threshold {thr:.2f}: {m.height:,} links, {(res['matched_entity_ids'] != '').sum():,} S1 with >=1 match "
          f"-> {out / 'matching_results.tsv'}")


if __name__ == "__main__":
    main()

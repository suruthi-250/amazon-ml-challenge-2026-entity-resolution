"""Export candidate pairs as compact text tables for GPU cross-encoder fine-tuning / scoring.

Writes work/gpu_export/:
  {split}_s1_text.parquet     row, text          (Source-1 records)
  {split}_r_text.parquet      row, text          (Source-2/3 records)
  train_pairs.parquet         s1row, rrow, label (sample of NON-validation training candidates)
  val_pairs.parquet           s1row, rrow, label (all validation candidates, same set v5 was scored on)
  test_pairs.parquet          s1row, rrow        (all final test candidates = candidate_pairs.tsv)
Texts are the normalised token forms v5 uses (accents stripped, Indic transliterated, noise removed).

Usage: python export_pairs.py
"""
import time

import polars as pl

import config as C
import match as M
from gt import gt_pairs

OUT = C.WORK / "gpu_export"
N_POS, N_NEG = 1_200_000, 1_200_000


def text_tables(split):
    d = C.split_dir(split)
    fmt = lambda: pl.concat_str([pl.lit("name: "), pl.col("n_toks").list.join(" "),
                                 pl.lit(" | addr: "), pl.col("a_toks").list.join(" ")]).alias("text")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["n_toks", "a_toks"]).with_row_index("row").select("row", fmt())
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["n_toks", "a_toks"]) for k in (2, 3)]) \
          .with_row_index("row").select("row", fmt())
    return s1, r


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        s1, r = text_tables(split)
        s1.write_parquet(OUT / f"{split}_s1_text.parquet", compression="zstd")
        r.write_parquet(OUT / f"{split}_r_text.parquet", compression="zstd")
        print(f"{split} texts: {s1.height:,} S1, {r.height:,} right  {time.time() - t0:.0f}s", flush=True)
    d = C.split_dir("train")
    s1 = pl.read_parquet(d / "s1.parquet", columns=["entity_id"]).with_row_index("row")
    r = pl.concat([pl.read_parquet(d / f"s{k}.parquet", columns=["entity_id"]) for k in (2, 3)]).with_row_index("row")
    g = gt_pairs(s1, r).with_columns(label=pl.lit(1, pl.Int8))
    c = M.select_candidates(pl.read_parquet(d / "cands_raw.parquet")).select("s1row", "rrow") \
         .join(g, on=["s1row", "rrow"], how="left").with_columns(pl.col("label").fill_null(0))
    val_r = c.filter(M._is_val("s1row")).select("rrow").unique()
    val = c.join(val_r, on="rrow", how="semi")
    tr = c.join(val_r, on="rrow", how="anti")
    tr = pl.concat([tr.filter(pl.col("label") == 1).sample(N_POS, seed=C.SEED),
                    tr.filter(pl.col("label") == 0).sample(N_NEG, seed=C.SEED)]).sample(fraction=1.0, shuffle=True, seed=C.SEED)
    tr.write_parquet(OUT / "train_pairs.parquet")
    val.write_parquet(OUT / "val_pairs.parquet")
    test = M.select_candidates(pl.read_parquet(C.split_dir("test") / "cands_raw.parquet")).select("s1row", "rrow")
    test.write_parquet(OUT / "test_pairs.parquet")
    print(f"pairs: train {tr.height:,} (pos {int(tr['label'].sum()):,}), val {val.height:,}, test {test.height:,}  "
          f"{time.time() - t0:.0f}s")
    for f in sorted(OUT.glob("*.parquet")):
        print(f"  {f.name:28s} {f.stat().st_size / 1e6:8.1f} MB")


if __name__ == "__main__":
    main()

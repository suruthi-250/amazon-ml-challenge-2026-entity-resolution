"""Paths and global settings for the entity-resolution pipeline.

Default layout (relative to the repository root):
    data/dataset/{train,test}/*.tsv   challenge data (not included in this repository)
    work/                             intermediates: parquet files, candidates, models
    output/                           candidate_pairs.tsv, matching_results.tsv

Every path can be overridden with an environment variable.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Folder that contains dataset/ (and, optionally, utils/validate_submission.py).
DATA_ROOT = Path(os.environ.get("ER_DATA_ROOT", REPO / "data"))
DATASET = DATA_ROOT / "dataset"

# Scratch directory for intermediate parquet files / models, and the submission folder.
WORK = Path(os.environ.get("ER_WORK_DIR", REPO / "work"))
OUTPUT = Path(os.environ.get("ER_OUTPUT_DIR", REPO / "output"))

N_JOBS = int(os.environ.get("ER_N_JOBS", max(1, (os.cpu_count() or 2) - 1)))
SEED = 42


def split_dir(split: str) -> Path:
    d = WORK / split
    d.mkdir(parents=True, exist_ok=True)
    return d

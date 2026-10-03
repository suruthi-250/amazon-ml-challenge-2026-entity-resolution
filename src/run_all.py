"""End-to-end pipeline: raw TSVs -> output/candidate_pairs.tsv + output/matching_results.tsv

    python run_all.py

Each step skips work whose output already exists in the work directory, so
the pipeline can be resumed after an interruption (delete work/ to redo all).
"""
import subprocess
import sys
import time

import config as C

STEPS = [
    (["prep.py"], [C.WORK / "train" / "s3.parquet", C.WORK / "test" / "s3.parquet"]),
    (["blocking.py", "train"], [C.WORK / "train" / "cands_raw.parquet"]),
    (["match.py", "train"], [C.WORK / "models.pkl"]),
    (["blocking.py", "test"], [C.WORK / "test" / "cands_raw.parquet"]),
    (["match.py", "test"], [C.OUTPUT / "matching_results.tsv"]),
]


def main():
    here = C.Path(__file__).resolve().parent
    for args, outputs in STEPS:
        if all(o.exists() for o in outputs):
            print(f"skip {' '.join(args)} (outputs exist)")
            continue
        t0 = time.time()
        print(f"==> {' '.join(args)}", flush=True)
        subprocess.run([sys.executable, "-W", "ignore", str(here / args[0])] + args[1:], check=True, cwd=here)
        print(f"<== {' '.join(args)} done in {time.time() - t0:.0f}s", flush=True)
    validator = C.DATA_ROOT / "utils" / "validate_submission.py"
    if validator.exists():
        subprocess.run([sys.executable, str(validator),
                        "--matching", str(C.OUTPUT / "matching_results.tsv"),
                        "--candidate", str(C.OUTPUT / "candidate_pairs.tsv"),
                        "--test-dir", str(C.DATASET / "test")], check=False)


if __name__ == "__main__":
    main()

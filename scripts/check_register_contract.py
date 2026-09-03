#!/usr/bin/env python3
"""CI/local check: does register.py's output actually satisfy the Phase 2
output contract end-to-end?

`pytest -q` exercises the package's internals, but nothing in the existing
suite runs the graded command itself against real generated images. A
packaging problem (a broken import, a missing weights file, an argument
mismatch) can pass every unit test and still leave register.py unusable --
this catches that class of bug, and would have caught the fallback-
degradation and CLI-decoupling regressions the hard way if it had existed
before issue #36.

Generates a small Phase-2-style dataset, runs
`register.py --input manifest.csv --output predictions.csv` on it exactly as
the organizer's grading command would, then asserts the organizer's stated
contract (task deck slide 5):

  * one row per pair_id, in input order, each pair_id exactly once
  * found in {0, 1}
  * found == 0  =>  x, y, theta, scale are all written 0
  * found == 1  =>  x, y, theta, scale, score are real numbers
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_FIELDS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-pairs", type=int, default=6)
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        ds = os.path.join(tmp, "ds")
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "generate_dataset.py"),
             "--phase2", "--num-pairs", str(a.num_pairs),
             "--output-dir", ds, "--seed", "12345"],
            check=True, cwd=REPO_ROOT)

        manifest = os.path.join(ds, "manifest.csv")
        preds = os.path.join(ds, "predictions.csv")
        subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, "register.py"),
             "--input", manifest, "--output", preds, "--quiet"],
            check=True, cwd=REPO_ROOT)

        with open(manifest, newline="") as f:
            in_ids = [row["id"] for row in csv.DictReader(f)]
        with open(preds, newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            out_rows = list(reader)

    assert fields == OUT_FIELDS, (
        f"predictions.csv header {fields} != required {OUT_FIELDS}")

    out_ids = [row["pair_id"] for row in out_rows]
    assert out_ids == in_ids, (
        "predictions.csv rows must match the input pair_ids, in order, each "
        f"exactly once. input={in_ids} output={out_ids}")

    for row in out_rows:
        found = row["found"]
        assert found in ("0", "1"), (
            f"{row['pair_id']}: found must be 0 or 1, got {found!r}")
        if found == "0":
            for col in ("x", "y", "theta", "scale"):
                assert float(row[col]) == 0.0, (
                    f"{row['pair_id']}: found=0 but {col}={row[col]!r}, "
                    "spec requires the pose columns to be 0")
        else:
            for col in ("x", "y", "theta", "scale", "score"):
                float(row[col])  # raises ValueError if not a real number

    n_found = sum(r["found"] == "1" for r in out_rows)
    print(f"OK: {len(out_rows)} pairs, contract satisfied "
          f"({n_found} found, {len(out_rows) - n_found} declined)")


if __name__ == "__main__":
    main()

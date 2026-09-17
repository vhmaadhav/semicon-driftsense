#!/usr/bin/env python3
"""Run register.py over a large pairs.csv in parallel chunks (issue #85 tooling).

Decoding a 500-pair validation split one pair at a time takes ~17 minutes.
This splits pairs.csv into --jobs contiguous chunks (image paths made absolute),
runs the unmodified register.py on each with --threads threads, and merges the
parts back in input order -- so the predictions are exactly what the
submission command writes, only faster:

    python scripts/register_parallel.py --input data/phase2_v2_val/dev/pairs.csv \
        --output data/phase2_v2_val/dev/eval/predictions.csv --jobs 6 --threads 4
    # extra register.py flags go after --
    python scripts/register_parallel.py --input ... --output ... -- --label-convention edge

Per-pair timings from a parallel run are contended and NOT valid for the
efficiency component; measure runtime single-process at 4 threads.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--register", default=os.path.join(HERE, "register.py"))
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="flags passed to register.py after --")
    a = ap.parse_args(argv)
    extra = a.extra[1:] if a.extra[:1] == ["--"] else a.extra

    base = os.path.dirname(os.path.abspath(a.input))
    with open(a.input, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    path_cols = [c for c in fields if c.lower().endswith("path") or c.lower() in ("reference", "search")]
    for r in rows:
        for c in path_cols:
            if r[c] and not os.path.isabs(r[c]):
                r[c] = os.path.join(base, r[c])

    parts_dir = os.path.abspath(a.output) + ".parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    os.makedirs(parts_dir)
    jobs = max(1, min(a.jobs, len(rows)))
    size = -(-len(rows) // jobs)
    procs = []
    t0 = time.time()
    for k in range(jobs):
        chunk = rows[k * size:(k + 1) * size]
        if not chunk:
            continue
        cin = os.path.join(parts_dir, f"in_{k:02d}.csv")
        cout = os.path.join(parts_dir, f"out_{k:02d}.csv")
        with open(cin, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(chunk)
        log = open(os.path.join(parts_dir, f"log_{k:02d}.txt"), "w")
        cmd = [sys.executable, a.register, "--input", cin, "--output", cout,
               "--threads", str(a.threads), "--quiet", *extra]
        procs.append((cout, len(chunk), subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log))

    failed = []
    merged = []
    header = None
    for cout, n, p, log in procs:
        rc = p.wait()
        log.close()
        if rc != 0 or not os.path.exists(cout):
            failed.append((cout, rc))
            continue
        with open(cout, newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            part = list(reader)
        if len(part) != n:
            failed.append((cout, f"{len(part)} rows, expected {n}"))
        merged.extend(part)
    if failed:
        raise SystemExit(f"chunk failures (see logs in {parts_dir}): {failed}")
    ids_in = [r["pair_id"] for r in rows]
    ids_out = [r["pair_id"] for r in merged]
    if ids_in != ids_out:
        raise SystemExit("merged pair_id order does not match the input")
    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(merged)
    print(f"wrote {len(merged)} rows to {a.output} from {len(procs)} chunks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

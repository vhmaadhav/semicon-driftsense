#!/usr/bin/env python3
"""Parallel front-end for generate_dataset.py / generate_cad_dataset.py.

Both upstream CLIs are single-process: one sample at a time on one core. This
runs N copies at once, each writing its own shard split with its own seed
(base_seed + shard index), then merges the shard manifests into one
`<output-dir>/<split>/manifest.csv` whose paths are relative to that
directory. The upstream scripts are called unmodified.

Measured on an i7-14700HX: ~0.5 s/sample and ~0.36 GB RAM per process, ~1.4 MB
disk per pair -- so CPU cores, not RAM, set the limit.

Examples:
    python generate_parallel.py --num-samples 5000 --split train --seed 100
    python generate_parallel.py --cad --num-samples 2000 --split cad_train --seed 200
    # extra upstream flags go after --
    python generate_parallel.py --num-samples 1000 --split hard -- --no-match-prob 0.3
"""

import argparse
import csv
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default=os.path.join(HERE, "output"))
    p.add_argument("--seed", type=int, default=42, help="shard k uses seed + k")
    p.add_argument("--workers", type=int, default=16,
                   help="parallel processes (default 16: throughput plateaus beyond this on the 14700HX -- ~9 samples/s measured at 16, 20 and 28)")
    p.add_argument("--cad", action="store_true", help="use generate_cad_dataset.py (GDSII reference)")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="flags passed through to the generator after --")
    return p.parse_args()


def main():
    args = parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    script = os.path.join(HERE, "generate_cad_dataset.py" if args.cad else "generate_dataset.py")
    workers = max(1, min(args.workers, args.num_samples))
    split_dir = os.path.join(args.output_dir, args.split)
    shard_root = os.path.join(split_dir, "shards")
    os.makedirs(shard_root, exist_ok=True)

    # Even split; the first shards absorb the remainder.
    base, rem = divmod(args.num_samples, workers)
    counts = [base + (k < rem) for k in range(workers)]

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1")
    procs = []
    for k, n in enumerate(counts):
        log = open(os.path.join(shard_root, f"s{k:02d}.log"), "w")
        cmd = [sys.executable, script, "--num-samples", str(n), "--split", f"s{k:02d}",
               "--output-dir", shard_root, "--seed", str(args.seed + k), *extra]
        procs.append((k, subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT), log))

    print(f"{args.num_samples} samples, {workers} workers, "
          f"{'CAD' if args.cad else 'image'} generator -> {split_dir}")
    t0 = time.time()
    failed = []
    while procs:
        time.sleep(2)
        for item in list(procs):
            k, p, log = item
            if p.poll() is not None:
                log.close()
                procs.remove(item)
                if p.returncode:
                    failed.append(k)
        done = sum(_rows(os.path.join(shard_root, f"s{k:02d}", "manifest.csv")) for k in range(workers))
        rate = done / max(time.time() - t0, 1e-9)
        eta = (args.num_samples - done) / rate if rate else 0
        print(f"\r  {done}/{args.num_samples}  {rate:.1f} samples/s  ETA {eta/60:.1f} min   ", end="", flush=True)
    print()
    if failed:
        raise SystemExit(f"shards failed: {failed} -- see {shard_root}\\sNN.log")

    # Merge: prefix every *_path column with shards/sNN so paths resolve from split_dir.
    merged = os.path.join(split_dir, "manifest.csv")
    total = 0
    with open(merged, "w", newline="") as out:
        writer = None
        for k in range(workers):
            shard = f"s{k:02d}"
            with open(os.path.join(shard_root, shard, "manifest.csv"), newline="") as f:
                reader = csv.DictReader(f)
                if writer is None:
                    fields = ["id", "shard_id"] + [c for c in reader.fieldnames if c != "id"]
                    writer = csv.DictWriter(out, fieldnames=fields)
                    writer.writeheader()
                for row in reader:
                    row["shard_id"] = row["id"]
                    row["id"] = total
                    for c in row:
                        if c.endswith("_path") and row[c]:
                            row[c] = os.path.join("shards", shard, row[c]).replace("\\", "/")
                    writer.writerow(row)
                    total += 1
    dt = time.time() - t0
    print(f"Wrote {total} samples in {dt/60:.1f} min ({total/dt:.1f}/s) -> {merged}")


def _rows(path):
    try:
        with open(path, "rb") as f:
            return max(f.read().count(b"\n") - 1, 0)
    except OSError:
        return 0


if __name__ == "__main__":
    main()

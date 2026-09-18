#!/usr/bin/env python3
"""Parallel front-end for generate_dataset.py / generate_cad_dataset.py /
generate_cad_varied.py.

Both upstream CLIs are single-process: one sample at a time on one core. This
splits the job into fixed-size chunks, runs N copies at once, each writing its
own chunk split with its own seed (base_seed + chunk index), then merges the shard manifests into one
`<output-dir>/<split>/manifest.csv` whose paths are relative to that
directory. The upstream scripts are called unmodified.

Measured on an i7-14700HX: ~0.5 s/sample and ~0.36 GB RAM per process, ~1.4 MB
disk per pair -- so CPU cores, not RAM, set the limit.

Examples:
    python generate_parallel.py --num-samples 5000 --split train --seed 100
    python generate_parallel.py --cad --num-samples 2000 --split cad_train --seed 200
    python generate_parallel.py --cad-varied --num-samples 2000 --split cad_varied --seed 300
    # extra upstream flags go after --
    python generate_parallel.py --num-samples 1000 --split hard -- --no-match-prob 0.3

Resuming: if the run is interrupted (terminal closed, Ctrl+C, reboot), rerun
the exact same command. Finished chunks carry a .done marker and are skipped;
unfinished ones are regenerated from their seed, so the result is identical to
an uninterrupted run.
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
    p.add_argument("--chunk-size", type=int, default=25,
                   help="samples per resumable chunk (an interruption loses at most the chunks in flight)")
    kind = p.add_mutually_exclusive_group()
    kind.add_argument("--cad", action="store_true", help="use generate_cad_dataset.py (GDSII reference)")
    kind.add_argument("--cad-varied", action="store_true",
                      help="use generate_cad_varied.py (GDSII reference, per-sample rotation and artefacts)")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="flags passed through to the generator after --")
    return p.parse_args()


def main():
    args = parse_args()
    extra = args.extra[1:] if args.extra[:1] == ["--"] else args.extra
    script = os.path.join(HERE, "generate_cad_varied.py" if args.cad_varied
                          else "generate_cad_dataset.py" if args.cad else "generate_dataset.py")
    split_dir = os.path.join(args.output_dir, args.split)
    shard_root = os.path.join(split_dir, "shards")
    os.makedirs(shard_root, exist_ok=True)

    # Fixed-size chunks, chunk k seeded seed + k. The chunking depends only on
    # --num-samples/--chunk-size, never on --workers, so a rerun with the same
    # arguments reproduces the same chunks and can skip finished ones.
    counts = [min(args.chunk_size, args.num_samples - s)
              for s in range(0, args.num_samples, args.chunk_size)]
    names = [f"s{k:04d}" for k in range(len(counts))]
    done_marker = lambda k: os.path.join(shard_root, names[k] + ".done")  # noqa: E731
    todo = [k for k in range(len(counts)) if not os.path.exists(done_marker(k))]
    skipped = sum(counts) - sum(counts[k] for k in todo)
    workers = max(1, min(args.workers, len(todo) or 1))

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENCV_FOR_THREADS_NUM="1")

    print(f"{args.num_samples} samples in {len(counts)} chunks, {workers} workers, "
          f"{os.path.basename(script)} -> {split_dir}")
    if skipped:
        print(f"  resuming: {skipped} samples already done, {len(todo)} chunks left")
    t0 = time.time()
    failed, running, queue = [], [], list(todo)
    finished = skipped
    while queue or running:
        while queue and len(running) < workers:
            k = queue.pop(0)
            # A chunk without its .done marker was interrupted: regenerate it whole.
            log = open(os.path.join(shard_root, names[k] + ".log"), "w")
            cmd = [sys.executable, script, "--num-samples", str(counts[k]), "--split", names[k],
                   "--output-dir", shard_root, "--seed", str(args.seed + k), *extra]
            running.append((k, subprocess.Popen(cmd, cwd=HERE, env=env, stdout=log,
                                                stderr=subprocess.STDOUT), log))
        time.sleep(1)
        for item in list(running):
            k, p, log = item
            if p.poll() is None:
                continue
            log.close()
            running.remove(item)
            if p.returncode:
                failed.append(names[k])
            else:
                open(done_marker(k), "w").close()
                finished += counts[k]
        in_flight = sum(_rows(os.path.join(shard_root, names[k], "manifest.csv")) for k, _, _ in running)
        made = finished - skipped + in_flight
        rate = made / max(time.time() - t0, 1e-9)
        eta = (args.num_samples - finished - in_flight) / rate if rate else 0
        print(f"\r  {finished + in_flight}/{args.num_samples}  {rate:.1f} samples/s  "
              f"ETA {eta/60:.1f} min   ", end="", flush=True)
    print()
    if failed:
        raise SystemExit(f"chunks failed: {failed} -- see logs in {shard_root}; rerun the same command to retry")

    # Merge: prefix every *_path column with shards/<chunk> so paths resolve from split_dir.
    merged = os.path.join(split_dir, "manifest.csv")
    total = 0
    with open(merged, "w", newline="") as out:
        writer = None
        for shard in names:
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

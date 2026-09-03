#!/usr/bin/env python3
"""Grow a training pool shard by shard, unattended, until told to stop.

Why shards
----------
Measured on this machine (scripts/profile_rtx.py): scene generation saturates
at ~8.8 pairs/s -- it is memory-bandwidth bound, so 6 workers and 14 workers
give the same rate -- while the GPU training step runs at ~57 img/s. Generation
is therefore the critical path by 6.5x, and a pool big enough to matter takes
hours.

Serialising that (generate everything, then train) would leave the GPU idle for
most of the run. Instead each shard is a self-contained split directory that
gets a COMPLETE marker written last, and `train.py --refresh-pool` picks up
finished shards between epochs. Generation and training then overlap, and later
epochs simply see a larger pool.

Every shard draws from one continuous seed stream (`--seed`, advanced by
`--start-index`), so canvas i is the same sample no matter how the run was
chunked and no canvas is ever generated twice.

    python scripts/build_pool.py --out data/pool --shards 12 --canvases 3000
    python scripts/build_pool.py --out data/pool --shards 4 --resume   # add more later
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def existing_shards(out_dir: str) -> list[str]:
    if not os.path.isdir(out_dir):
        return []
    return sorted(d for d in os.listdir(out_dir)
                  if os.path.exists(os.path.join(out_dir, d, "COMPLETE")))


def dir_size_gb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 2 ** 30


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/pool")
    p.add_argument("--shards", type=int, default=12, help="shards to add in this run")
    p.add_argument("--canvases", type=int, default=3000, help="canvases per shard")
    p.add_argument("--crops-per-canvas", type=int, default=8,
                   help="8 is the validated setting. Measured: 16 buys only +17%% "
                        "pairs/s and 32 only +32%%, while halving/quartering the "
                        "number of distinct canvases behind each pair -- and canvas "
                        "diversity is what the streaming experiment showed matters.")
    p.add_argument("--seed", type=int, default=31337,
                   help="disjoint from every split seed (42, 555, 1234, 7777, 20001, "
                        "20002) and from stream_eval's 999983 namespace")
    p.add_argument("--workers", type=int, default=6,
                   help="6 saturates generation on this machine; more does not help")
    p.add_argument("--max-gb", type=float, default=120.0, help="stop if the pool exceeds this")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    out_dir = os.path.join(HERE, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(out_dir, exist_ok=True)

    done = existing_shards(out_dir)
    first = len(done)
    print(f"pool: {out_dir}")
    print(f"already complete: {len(done)} shards, {dir_size_gb(out_dir):.1f} GB")
    print(f"adding {args.shards} shards x {args.canvases} canvases x "
          f"{args.crops_per_canvas} crops = "
          f"{args.shards * args.canvases * args.crops_per_canvas:,} pairs\n", flush=True)

    for k in range(first, first + args.shards):
        shard = f"s{k:03d}"
        start = k * args.canvases
        cmd = [sys.executable, os.path.join(HERE, "scripts", "gen_data.py"),
               "--split", shard,
               "--num-samples", str(args.canvases),
               "--seed", str(args.seed),
               "--start-index", str(start),
               "--workers", str(args.workers),
               "--crops-per-canvas", str(args.crops_per_canvas),
               "--store-templates",
               "--output-dir", out_dir,
               "--progress-every", "500"]
        if args.dry_run:
            print(" ".join(cmd))
            continue

        size = dir_size_gb(out_dir)
        if size > args.max_gb:
            print(f"stopping: pool is {size:.1f} GB, over --max-gb {args.max_gb}")
            break

        t0 = time.time()
        print(f"=== shard {shard}  canvases [{start}, {start + args.canvases})", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            print(f"shard {shard} FAILED (exit {r.returncode}); stopping so the "
                  f"pool is not left with a partial shard un-marked", flush=True)
            break
        dt = time.time() - t0
        pairs = args.canvases * args.crops_per_canvas
        print(f"    {shard} done in {dt/60:.1f} min  ({pairs/dt:.1f} pairs/s)  "
              f"pool now {dir_size_gb(out_dir):.1f} GB, "
              f"{len(existing_shards(out_dir)) * pairs:,} pairs\n", flush=True)

    print(f"pool: {len(existing_shards(out_dir))} shards, {dir_size_gb(out_dir):.1f} GB")


if __name__ == "__main__":
    main()

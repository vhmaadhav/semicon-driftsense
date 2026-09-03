#!/usr/bin/env python3
"""Development CLI for generating dataset splits.

Thin wrapper over driftsense.generate (the same core the deliverable
generate_dataset.py uses, so both emit byte-identical data). This one is
split-oriented -- it takes a split name and the full list of node presets --
whereas generate_dataset.py takes an architecture family and a pair count.

    python scripts/gen_data.py --split train_mc --num-samples 1500 --seed 555 \
        --noise randomized --workers 5 --crops-per-canvas 8

Sample i is seeded from its own SeedSequence child, so a given (seed, i)
reproduces regardless of --workers.
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.generate import (  # noqa: E402
    NOISE_PRESETS, PRESETS, PoseParams, PoseSpec, write_split)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", required=True)
    p.add_argument("--num-samples", type=int, required=True,
                   help="number of canvases (pairs = this x --crops-per-canvas)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--noise", default="randomized", choices=list(NOISE_PRESETS) + ["randomized"])
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()),
                   choices=list(PRESETS.keys()))
    p.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "data"))
    p.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) // 2, 1))
    p.add_argument("--crops-per-canvas", type=int, default=1,
                   help="reference crops per canvas; >1 multiplies training pairs cheaply "
                        "but they share a search image (keep at 1 for eval splits)")
    p.add_argument("--store-templates", action="store_true",
                   help="write 100x100 templates instead of 1000x1000 references. "
                        "Lossless for training -- build_sample's only use of the "
                        "reference is exactly this downsample, and it commutes with "
                        "the dihedral augmentation -- and ~7x less disk per pair. "
                        "TRAINING POOLS ONLY: evaluate.py and infer.py need full "
                        "resolution references.")
    p.add_argument("--start-index", type=int, default=0,
                   help="append to an existing split starting at this canvas index, "
                        "keeping the same seed stream. Lets a pool grow in shards.")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--edge-brightening", type=float, default=0.0)
    p.add_argument("--rotation-deg", type=float, default=0.0)
    p.add_argument("--magnification", type=float, default=10.0)
    p.add_argument("--phase2", action="store_true",
                   help="disclosed Phase 2 operating point: magnification 8-12x, "
                        "rotation +/-5 deg, absent pairs at --absent-frac")
    p.add_argument("--absent-frac", type=float, default=0.2,
                   help="absent-pair fraction under --phase2")
    args = p.parse_args()

    split_dir = os.path.join(args.output_dir, args.split)
    pairs = write_split(
        split_dir=split_dir,
        num_canvases=args.num_samples,
        seed=args.seed,
        noise=args.noise,
        architectures=args.architectures,
        workers=args.workers,
        crops_per_canvas=args.crops_per_canvas,
        store_templates=args.store_templates,
        start_index=args.start_index,
        progress_every=args.progress_every,
        pose=(PoseSpec(rotation_deg=(-5.0, 5.0), magnification=(8.0, 12.0),
                       edge_brightening=(args.edge_brightening,) * 2,
                       absent_frac=args.absent_frac)
              if args.phase2 else
              PoseParams(edge_brightening=args.edge_brightening,
                         rotation_deg=args.rotation_deg,
                         magnification=args.magnification)),
    )
    # Marker written last, so a shard is only ever picked up complete. train.py
    # --refresh-pool rescans for these between epochs, which is what lets the
    # GPU start training on the first shards while later ones are still being
    # generated -- and generation is 6.5x slower than the GPU here, so that
    # overlap is most of the wall clock.
    with open(os.path.join(split_dir, "COMPLETE"), "w") as f:
        f.write(f"{pairs} pairs from {args.num_samples} canvases, seed {args.seed}, "
                f"start_index {args.start_index}, crops {args.crops_per_canvas}, "
                f"templates {args.store_templates}\n")
    print(f"Wrote {pairs} pairs from {args.num_samples} canvases ({args.noise}) to {split_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate synthetic Drift-Sense Reference/Search image pairs with ground truth.

Physical setup (fixed by the problem statement):

    Reference : 1000 x 1000 px @ 1 nm/px   -> 1 um field of view
    Search    : 1000 x 1000 px @ 10 nm/px  -> 10 um field of view

The 10x pixel-size ratio means the reference pattern occupies exactly a
100 x 100 px box inside the search image.

Each sample builds a 10000 x 10000 px "fine canvas" of a repeating layout
(DRAM 6F^2 cell array or FinFET fin/gate array), composed into discrete
array mats separated by flatter routing strips. A 1000 x 1000 window of that
canvas is imaged as the Reference; the whole canvas is blurred, downsampled
10x and imaged as the Search. Both go through an SEM acquisition model
(beam PSF, shot noise, detector noise, raster drift, and optional
astigmatism / vignetting / gamma / barrel / charging / speckle / impulse
noise).

Ground truth
------------
Two conventions are recorded per pair:

    gt_x,      gt_y       centre from the crop origin, pre-imaging
    gt_x_corr, gt_y_corr  centre after the search frame's geometric warps

Use gt_x_corr / gt_y_corr. The raster-drift and barrel-distortion steps move
the pattern *after* the crop coordinates are fixed, so the uncorrected
columns can be several pixels off -- more than the evaluation tolerance. See
driftsense/generate.py:correct_gt for the inversion, and
scripts/verify_gt_correction.py for the empirical check.

Examples
--------
    # 50 mixed DRAM+FinFET pairs
    python generate_dataset.py --num-pairs 50 --output-dir ./output

    # DRAM only, harsh acquisition conditions
    python generate_dataset.py --architecture dram --num-pairs 100 \
        --noise severe --output-dir ./output/dram_severe

Outputs `<output-dir>/reference/`, `<output-dir>/search/` and
`<output-dir>/manifest.csv`.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from driftsense.generate import NOISE_PRESETS, write_split  # noqa: E402
from driftsense.presets import architecture_presets  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture", default="mixed", choices=["dram", "finfet", "mixed"],
                   help="structure family to sample from (default: mixed)")
    p.add_argument("--num-pairs", type=int, default=20,
                   help="number of Reference/Search pairs to generate")
    p.add_argument("--output-dir", default="./output",
                   help="directory to write reference/, search/ and manifest.csv into")
    p.add_argument("--noise", default="randomized",
                   choices=list(NOISE_PRESETS) + ["randomized"],
                   help="acquisition conditions: a fixed operating point, or "
                        "'randomized' to draw them per pair (default: randomized)")
    p.add_argument("--seed", type=int, default=42, help="reproducibility seed")
    p.add_argument("--workers", type=int, default=max((os.cpu_count() or 2) // 2, 1))
    p.add_argument("--crops-per-canvas", type=int, default=1,
                   help="reference crops per canvas. >1 is much cheaper per pair but "
                        "the pairs share a search image -- use only for training data")
    return p.parse_args()


def main():
    args = parse_args()
    presets = architecture_presets(args.architecture)

    if args.crops_per_canvas < 1:
        raise SystemExit("--crops-per-canvas must be >= 1")
    canvases = -(-args.num_pairs // args.crops_per_canvas)  # ceil

    print(f"architecture : {args.architecture} ({len(presets)} presets)")
    print(f"pairs        : {args.num_pairs} from {canvases} canvas(es)")
    print(f"conditions   : {args.noise}")
    print(f"output       : {args.output_dir}")

    pairs = write_split(
        split_dir=args.output_dir,
        num_canvases=canvases,
        seed=args.seed,
        noise=args.noise,
        architectures=presets,
        workers=args.workers,
        crops_per_canvas=args.crops_per_canvas,
    )

    print(f"\nWrote {pairs} pairs to {args.output_dir}")
    print(f"  {os.path.join(args.output_dir, 'reference')}/  1000x1000 @ 1 nm/px")
    print(f"  {os.path.join(args.output_dir, 'search')}/     1000x1000 @ 10 nm/px")
    print(f"  {os.path.join(args.output_dir, 'manifest.csv')}  "
          f"(ground truth: use gt_x_corr, gt_y_corr)")


if __name__ == "__main__":
    main()

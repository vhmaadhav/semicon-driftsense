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

from driftsense.generate import (  # noqa: E402
    NOISE_PRESETS, PoseSpec, write_split)
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

    pose = p.add_argument_group(
        "pose and edge response",
        "Degradations named by the problem statement that the upstream "
        "generator does not model. All default to the nominal no-op, which "
        "reproduces the upstream imaging path exactly; the shipped weights "
        "were trained without them.")
    pose.add_argument("--edge-brightening", type=float, default=0.0, metavar="F",
                      help="secondary-electron edge brightening, 0-1 "
                           "(0 = off; 0.15-0.35 is a visible, realistic range)")
    pose.add_argument("--rotation-deg", type=float, default=0.0, metavar="DEG",
                      help="relative rotation between Reference and Search "
                           "(0 = off; the problem statement implies about +/-2)")
    pose.add_argument("--magnification", type=float, default=10.0, metavar="X",
                      help="effective magnification ratio (default 10.0, the "
                           "nominal 1 nm/px : 10 nm/px; 9-11 spans the range "
                           "the problem statement implies)")

    ph2 = p.add_argument_group(
        "phase 2: unknown pose and absent pairs",
        "Phase 2 draws the pose independently for every pair and leaves a "
        "fraction of pairs with no true instance. These ranges override the "
        "fixed --rotation-deg / --magnification values above.")
    ph2.add_argument("--phase2", action="store_true",
                     help="shorthand for the disclosed Phase 2 operating point: "
                          "--magnification-range 8 12 --rotation-range -5 5 "
                          "--absent-frac 0.2")
    ph2.add_argument("--rotation-range", type=float, nargs=2, metavar=("LO", "HI"),
                     help="sample rotation uniformly in [LO, HI] degrees, "
                          "CCW positive (Phase 2 bound: -5 5)")
    ph2.add_argument("--magnification-range", type=float, nargs=2, metavar=("LO", "HI"),
                     help="sample the magnification ratio uniformly in [LO, HI] "
                          "(Phase 2 bound: 8 12)")
    ph2.add_argument("--absent-frac", type=float, default=0.0, metavar="F",
                     help="fraction of pairs whose reference has no true "
                          "instance in the search frame, cropped instead from "
                          "another die region of the same architecture "
                          "(Phase 2 blind set is about 0.2)")
    ph2.add_argument("--polygon-scale-range", type=float, nargs=2, metavar=("LO", "HI"),
                     help="Set B polygon scaling: multiply every drawn feature's "
                          "CD by 1+f with f uniform in [LO, HI], pitch unchanged "
                          "(Phase 2 Set B bound: -0.2 0.2). Disabled by default, "
                          "and when disabled no random draw is made, so the "
                          "Phase 1 splits reproduce byte-for-byte.")
    ph2.add_argument("--severity-range", type=float, nargs=2, metavar=("LO", "HI"),
                     help="Set B severity ladder: draw one latent severity in "
                          "[LO, HI] (0=nominal, 1=level 4) and move charging, "
                          "scan distortion, defocus and shot noise together "
                          "along it. Disabled by default; --phase2 turns it on "
                          "over the full 0 1 range.")
    return p.parse_args()


def build_pose_spec(args) -> PoseSpec:
    """Fold the fixed --rotation-deg/--magnification flags and the Phase 2
    range flags into one spec. A range wins over the corresponding fixed
    value; absent both, the fixed value is pinned and the result is the
    nominal no-op that reproduces the upstream imaging path."""
    rot = tuple(args.rotation_range) if args.rotation_range else (args.rotation_deg,) * 2
    mag = tuple(args.magnification_range) if args.magnification_range else (args.magnification,) * 2
    absent = args.absent_frac
    poly = tuple(args.polygon_scale_range) if args.polygon_scale_range else (0.0, 0.0)
    sev = tuple(args.severity_range) if args.severity_range else (0.0, 0.0)
    if args.phase2:
        rot = tuple(args.rotation_range) if args.rotation_range else (-5.0, 5.0)
        mag = tuple(args.magnification_range) if args.magnification_range else (8.0, 12.0)
        absent = args.absent_frac if args.absent_frac else 0.2
        # Set B names polygon scaling +/-20% as a degradation category, so the
        # Phase 2 shorthand turns it on. Pass --polygon-scale-range 0 0 to
        # generate a Set A-style split with the pose ranges but nominal CD.
        poly = tuple(args.polygon_scale_range) if args.polygon_scale_range else (-0.2, 0.2)
        # The shipped weights had never seen severity 4; the full range is the
        # whole point of regenerating for Phase 2.
        sev = tuple(args.severity_range) if args.severity_range else (0.0, 1.0)
    if not 0.0 <= absent <= 1.0:
        raise SystemExit("--absent-frac must be in [0, 1]")
    for name, (lo, hi) in (("--rotation-range", rot), ("--magnification-range", mag),
                           ("--polygon-scale-range", poly)):
        if hi < lo:
            raise SystemExit(f"{name}: LO must not exceed HI (got {lo} {hi})")
    if mag[0] <= 0:
        raise SystemExit("--magnification-range: magnification must be positive")
    if poly[0] <= -1.0:
        raise SystemExit("--polygon-scale-range: LO must exceed -1 (features cannot vanish)")
    return PoseSpec(rotation_deg=rot, magnification=mag,
                    edge_brightening=(args.edge_brightening,) * 2,
                    absent_frac=absent, polygon_scale=poly, severity=sev)


def main():
    args = parse_args()
    presets = architecture_presets(args.architecture)

    if args.crops_per_canvas < 1:
        raise SystemExit("--crops-per-canvas must be >= 1")
    canvases = -(-args.num_pairs // args.crops_per_canvas)  # ceil

    print(f"architecture : {args.architecture} ({len(presets)} presets)")
    print(f"pairs        : {args.num_pairs} from {canvases} canvas(es)")
    print(f"conditions   : {args.noise}")
    spec = build_pose_spec(args)
    if spec != PoseSpec():
        def _fmt(lohi, unit):
            lo, hi = lohi
            return f"{lo:g}{unit}" if hi <= lo else f"[{lo:g}, {hi:g}]{unit}"
        print(f"pose/edge    : magnification {_fmt(spec.magnification, 'x')}, "
              f"rotation {_fmt(spec.rotation_deg, ' deg')}, "
              f"edge brightening {_fmt(spec.edge_brightening, '')}")
    if spec.absent_frac:
        print(f"absent pairs : {spec.absent_frac:.0%} (found=0, reference from "
              f"another die region)")
    print(f"output       : {args.output_dir}")

    pairs = write_split(
        split_dir=args.output_dir,
        num_canvases=canvases,
        seed=args.seed,
        noise=args.noise,
        architectures=presets,
        workers=args.workers,
        crops_per_canvas=args.crops_per_canvas,
        pose=spec,
    )

    print(f"\nWrote {pairs} pairs to {args.output_dir}")
    print(f"  {os.path.join(args.output_dir, 'reference')}/  1000x1000 @ 1 nm/px")
    print(f"  {os.path.join(args.output_dir, 'search')}/     1000x1000 @ 10 nm/px")
    print(f"  {os.path.join(args.output_dir, 'manifest.csv')}  "
          f"(ground truth: use gt_x_corr, gt_y_corr)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Empirically check that gt_x_corr/gt_y_corr beat the upstream gt_x/gt_y.

Method: generate samples under near-noiseless photometric conditions but
with the geometric warps (shear/jitter/barrel) active, then let ZNCC find
the pattern. Under clean photometrics ZNCC's peak is a good proxy for the
true visual location, so whichever label it sits closer to is the label that
actually describes where the pattern is.

If the correction in driftsense.generate.correct_gt is right, the distance to
gt_*_corr should be substantially smaller than to gt_*.

Usage:  python scripts/verify_gt_correction.py --n 24
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))
sys.path.insert(0, REPO_ROOT)

from src import sem_imaging  # noqa: E402
from src.pipeline import (  # noqa: E402
    PIXEL_SIZE_REF_NM, REFERENCE_SIZE_PX, SCALE_FACTOR,
    GenerationParams, generate_fine_canvas_zoned, _pick_crop_origin,
)
from baseline_solution.zncc import zncc_match  # noqa: E402
from driftsense.generate import BOX_PX, correct_gt, image_search_traced  # noqa: E402

ARCHS = ["dram_1x", "dram_dense", "dram_wide", "finfet_10nm", "finfet_14nm", "finfet_22nm"]


def run_case(label: str, overrides: dict, n: int, seed: int) -> None:
    d_raw, d_corr = [], []
    for i in range(n):
        rng = np.random.default_rng(seed + i)
        arch = ARCHS[i % len(ARCHS)]
        # Clean photometrics so the ZNCC peak reflects geometry, not noise.
        base = dict(
            dose_reference=4000.0, dose_search=4000.0,
            detector_noise_sigma_ref=0.5, detector_noise_sigma_search=0.5,
            shear_amplitude_px=0.0, drift_jitter_px=0.0,
        )
        base.update(overrides)
        params = GenerationParams(**base)
        zone = generate_fine_canvas_zoned(arch, rng, params)
        canvas = zone["canvas"]
        x0, y0 = _pick_crop_origin(zone, params, rng)
        crop = canvas[y0:y0 + REFERENCE_SIZE_PX, x0:x0 + REFERENCE_SIZE_PX]

        reference = sem_imaging.image_reference(
            crop, pixel_size_nm=PIXEL_SIZE_REF_NM,
            spot_size_nm=params.beam_spot_size_nm, dose=params.dose_reference,
            rng=rng, detector_noise_sigma=params.detector_noise_sigma_ref,
            drift_jitter_px=0.0, barrel_distortion_k=0.0,
        )
        search, row_shift, k = image_search_traced(canvas, params, rng)

        gt_x = x0 / SCALE_FACTOR + BOX_PX / 2.0
        gt_y = y0 / SCALE_FACTOR + BOX_PX / 2.0
        cx, cy = correct_gt(gt_x, gt_y, row_shift, k)

        m = zncc_match(reference, search)
        d_raw.append(np.hypot(m["x"] - gt_x, m["y"] - gt_y))
        d_corr.append(np.hypot(m["x"] - cx, m["y"] - cy))

    d_raw, d_corr = np.array(d_raw), np.array(d_corr)

    # ZNCC frequently locks onto the WRONG repeat in these periodic layouts
    # (that failure mode is the problem statement, not a labelling issue).
    # Those samples land ~50px away and say nothing about label quality, so
    # judge the correction only where the matcher found the right region.
    locked = np.minimum(d_raw, d_corr) < 15.0
    if locked.sum() == 0:
        print(f"{label:28s} n={n:3d}  no lock-ons; inconclusive")
        return
    lr, lc = d_raw[locked], d_corr[locked]
    print(f"{label:28s} n={n:3d}  lock-on {locked.mean():4.0%}  |  "
          f"residual on lock-ons: gt {lr.mean():5.2f}px -> corr {lc.mean():5.2f}px   "
          f"acc@5px {np.mean(lr <= 5):.2f} -> {np.mean(lc <= 5):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--seed", type=int, default=31337)
    args = p.parse_args()

    print("Clean photometrics; only the geometric warp varies.\n")
    run_case("shear=4.0 only", {"shear_amplitude_px": 4.0}, args.n, args.seed)
    run_case("barrel k=+0.02 only", {"barrel_distortion_k": 0.02}, args.n, args.seed)
    run_case("barrel k=-0.02 only", {"barrel_distortion_k": -0.02}, args.n, args.seed)
    run_case("shear=4.0 + barrel=0.02", {"shear_amplitude_px": 4.0, "barrel_distortion_k": 0.02},
             args.n, args.seed)
    run_case("shear=3 + jitter=1 + barrel", {"shear_amplitude_px": 3.0, "drift_jitter_px": 1.0,
                                             "barrel_distortion_k": 0.015}, args.n, args.seed)


if __name__ == "__main__":
    main()

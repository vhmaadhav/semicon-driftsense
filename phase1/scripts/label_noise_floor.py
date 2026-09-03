#!/usr/bin/env python3
"""Estimate the irreducible label error in the Drift-Sense ground truth.

Why this exists: pipeline.generate_sample computes ground truth purely from
the crop origin on the *pre-imaging* fine canvas --

    gt_x = x0 / 10 + 50,   gt_y = y0 / 10 + 50

-- but sem_imaging.image_search then applies apply_raster_drift() and
apply_barrel_distortion() to the downsampled canvas. Those warps move the
pattern relative to the coordinate frame the label was written in, and the
label is never corrected for them. So the visually-correct answer and the
recorded label differ by a known-but-unmodelled offset.

Consequences for a learned matcher:
  * the shear term is deterministic given the row -- a model CAN learn it
  * the per-row jitter is i.i.d. noise -- nothing can learn it; it sets a
    hard floor on achievable accuracy
  * at tolerance=5px (the baseline's metric) this floor is a real fraction
    of the error budget, so it should be measured, not assumed negligible

Usage:  python scripts/label_noise_floor.py --manifest data/train/manifest.csv
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

SEARCH_SIZE_PX = 1000


def displacement_for_row(shear_amp: float, jitter_std: float, row: float,
                         rng: np.random.Generator, n: int = 4000) -> np.ndarray:
    """X-displacement (px) between where the label says a feature is and
    where apply_raster_drift actually renders it, at search-image row `row`.

    apply_raster_drift builds map_x = x + row_shift[row], and cv2.remap
    reads output(y, x) = input(y, map_x) -- so content at input column c
    lands at output column c - row_shift[row]. The label tracks input
    coordinates, hence the observed feature sits at (label - row_shift).
    """
    shear = shear_amp * (row / (SEARCH_SIZE_PX - 1))
    jitter = rng.normal(0.0, jitter_std, size=n) if jitter_std > 0 else np.zeros(n)
    return -(shear + jitter)


def barrel_displacement(k: float, x: float, y: float) -> float:
    """Radial displacement magnitude (px) from apply_barrel_distortion at
    (x, y), using the same normalisation as the imaging code.
    """
    if k == 0.0:
        return 0.0
    cx = cy = (SEARCH_SIZE_PX - 1) / 2.0
    nx, ny = (x - cx) / cx, (y - cy) / cy
    r2 = nx * nx + ny * ny
    # source = n * (1 + k*r2); displacement is that minus the identity map
    dx = (nx * (1 + k * r2) - nx) * cx
    dy = (ny * (1 + k * r2) - ny) * cy
    return float(np.hypot(dx, dy))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True)
    p.add_argument("--tolerance-px", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    shear_off, jitter_only, barrel_off = [], [], []
    for r in rows:
        gy, gx = float(r["gt_y"]), float(r["gt_x"])
        shear_amp = float(r["shear_amplitude_px"])
        jitter_std = float(r["drift_jitter_px"])
        k = float(r["barrel_distortion_k"])

        d = displacement_for_row(shear_amp, jitter_std, gy, rng, n=64)
        shear_off.append(np.abs(d).mean())
        # jitter alone = the part no model can predict
        jitter_only.append(jitter_std)
        barrel_off.append(barrel_displacement(k, gx, gy))

    shear_off = np.array(shear_off)
    jitter_only = np.array(jitter_only)
    barrel_off = np.array(barrel_off)

    print(f"samples: {len(rows)}   tolerance: {args.tolerance_px}px\n")
    print("total label offset |shear + jitter| (px), learnable + not:")
    print(f"  mean {shear_off.mean():.2f}   median {np.median(shear_off):.2f}   "
          f"p95 {np.percentile(shear_off, 95):.2f}   max {shear_off.max():.2f}")
    print("\nunlearnable component -- per-row jitter sigma (px):")
    print(f"  mean {jitter_only.mean():.2f}   median {np.median(jitter_only):.2f}   "
          f"p95 {np.percentile(jitter_only, 95):.2f}   max {jitter_only.max():.2f}")
    print("\nbarrel-distortion offset at the gt point (px):")
    print(f"  mean {barrel_off.mean():.2f}   p95 {np.percentile(barrel_off, 95):.2f}   "
          f"max {barrel_off.max():.2f}")

    # Combined worst-case: a perfect matcher finds the *visual* location, so
    # its residual against the stored label is the sum of these offsets.
    combined = shear_off + barrel_off
    frac = float((combined > args.tolerance_px).mean())
    print(f"\nfraction of samples where a visually-perfect match would still "
          f"miss the stored label by >{args.tolerance_px}px: {frac:.3%}")
    print(f"mean residual of a visually-perfect matcher: {combined.mean():.2f}px "
          f"({combined.mean() / args.tolerance_px:.0%} of the error budget)")


if __name__ == "__main__":
    main()

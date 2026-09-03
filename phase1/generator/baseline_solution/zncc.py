#!/usr/bin/env python3
"""Explicit ZNCC (zero-mean normalized cross-correlation) matcher.

`cv2.matchTemplate(..., cv2.TM_CCOEFF_NORMED)` *is* ZNCC -- for each window
position it subtracts the local mean from both template and window before
correlating, then normalizes by their standard deviations. `infer.py`
already uses it under the hood; this module exposes it under its proper
name, returns the peak confidence score alongside the match location, and
reports the shift between predicted and ground-truth location when the
ground truth is known (as it always is for this synthetic generator).

Example:
    python baseline_solution/zncc.py --reference ref.png --search search.png \\
        --gt-x 746.6 --gt-y 318.8
"""

import argparse

import cv2
import numpy as np


def zncc_match(reference: np.ndarray, search: np.ndarray, scales=(9.0, 9.5, 10.0, 10.5, 11.0)) -> dict:
    """Multi-scale ZNCC template match. Returns the best-scoring location,
    its ZNCC confidence score (in [-1, 1], higher is a stronger match), and
    the scale/template size that produced it.
    """
    best = None
    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)  # ZNCC
        _, score, _, max_loc = cv2.minMaxLoc(result)
        if best is None or score > best["score"]:
            best = {
                "x": max_loc[0] + tw / 2.0,
                "y": max_loc[1] + th / 2.0,
                "score": float(score),
                "scale": scale,
                "template_w": tw,
                "template_h": th,
            }
    return best


def shift_report(pred_x: float, pred_y: float, gt_x: float, gt_y: float) -> dict:
    """Shift of the predicted match relative to the true (ground-truth)
    location: dx/dy are predicted-minus-actual, in search-image pixels.
    """
    dx = pred_x - gt_x
    dy = pred_y - gt_y
    return {
        "gt_x": gt_x, "gt_y": gt_y,
        "pred_x": pred_x, "pred_y": pred_y,
        "dx": dx, "dy": dy,
        "distance_px": float(np.hypot(dx, dy)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True)
    p.add_argument("--search", required=True)
    p.add_argument("--gt-x", type=float, default=None, help="ground-truth x, to report shift error")
    p.add_argument("--gt-y", type=float, default=None, help="ground-truth y, to report shift error")
    args = p.parse_args()

    reference = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(args.search, cv2.IMREAD_GRAYSCALE)
    if reference is None or search is None:
        raise ValueError("Could not read reference or search image")

    match = zncc_match(reference, search)
    print(f"predicted: x={match['x']:.2f} y={match['y']:.2f} zncc_score={match['score']:.4f} scale={match['scale']}")

    if args.gt_x is not None and args.gt_y is not None:
        shift = shift_report(match["x"], match["y"], args.gt_x, args.gt_y)
        print(f"shift (predicted - actual): dx={shift['dx']:+.2f} dy={shift['dy']:+.2f} "
              f"distance={shift['distance_px']:.2f} px")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Classical template-matching baseline for Drift-Sense.

Mirrors the exact submission interface Applied Materials requires: a
standalone script that accepts a reference and a search image path and
prints the predicted (x, y) center in the search image, in pixels.

Approach: the reference is captured at 10x finer resolution than the search
image, so we downsample it by ~10x and slide it over the search image with
normalized cross-correlation (cv2.matchTemplate). We search a small range of
scales around 10x rather than assuming the ratio is exactly known, since a
real solution shouldn't hardcode ground truth it isn't given.

Example:
    python baseline_solution/infer.py --reference ref.png --search search.png
"""

import argparse

import cv2
import numpy as np


def predict(reference_path: str, search_path: str, scales=(9.0, 9.5, 10.0, 10.5, 11.0)):
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if reference is None or search is None:
        raise ValueError("Could not read reference or search image")

    best_score = -np.inf
    best_center = (search.shape[1] / 2.0, search.shape[0] / 2.0)

    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, max_loc = cv2.minMaxLoc(result)
        if score > best_score:
            best_score = score
            best_center = (max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0)

    return best_center


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--search", required=True)
    args = p.parse_args()

    x, y = predict(args.reference, args.search)
    print(f"{x:.2f},{y:.2f}")


if __name__ == "__main__":
    main()

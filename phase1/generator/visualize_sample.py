#!/usr/bin/env python3
"""Draw the ground-truth box on a generated sample's search image, next to
its reference thumbnail, and save a side-by-side comparison PNG.

Example:
    python visualize_sample.py --output-dir ./output --split train --id 0
"""

import argparse
import csv
import os

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--split", default="train")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--save-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    split_dir = os.path.join(args.output_dir, args.split)
    manifest_path = os.path.join(split_dir, "manifest.csv")

    with open(manifest_path) as f:
        rows = list(csv.DictReader(f))
    row = next((r for r in rows if int(r["id"]) == args.id), None)
    if row is None:
        raise ValueError(f"id {args.id} not found in {manifest_path}")

    reference = cv2.imread(row["reference_path"], cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(row["search_path"], cv2.IMREAD_GRAYSCALE)

    search_annotated = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
    x0, y0, w, h = (float(row["gt_box_x"]), float(row["gt_box_y"]), float(row["gt_box_w"]), float(row["gt_box_h"]))
    cv2.rectangle(
        search_annotated,
        (int(round(x0)), int(round(y0))),
        (int(round(x0 + w)), int(round(y0 + h))),
        (0, 0, 255),
        2,
    )

    ref_resized = cv2.resize(
        cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR),
        (search_annotated.shape[1], search_annotated.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    combined = np.hstack([ref_resized, search_annotated])

    save_path = args.save_path or os.path.join(split_dir, f"visualize_{args.id:05d}.png")
    cv2.imwrite(save_path, combined)
    print(f"architecture={row['architecture']}  gt=({row['gt_x']}, {row['gt_y']})")
    print(f"Saved comparison to {save_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Integrity + summary report for a generated Drift-Sense split.

Checks every manifest row against the files on disk and the geometry the
problem statement fixes (1000x1000 frames, 100x100 target box fully inside
the search image), then prints the composition of the split.

Usage:  python scripts/verify_dataset.py data/train [data/val ...]
        python scripts/verify_dataset.py --sample 2000 data/train_pool

Compact training pools (gen_data.py --store-templates) store the reference at
100x100 instead of 1000x1000; both are accepted, and the manifest's
`reference_px` column must agree with what is actually on disk.

`--sample N` checks the geometry of every row but only opens the images of N
randomly chosen rows. A 300k-pair pool takes hours to read in full and the
per-image check is the same check 300k times; the row-level checks stay
exhaustive.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter

import cv2
import numpy as np

EXPECTED_HW = (1000, 1000)
EXPECTED_BOX = 100
TEMPLATE_HW = (100, 100)


def check_split(split_dir: str, sample: int = 0, seed: int = 0) -> bool:
    manifest = os.path.join(split_dir, "manifest.csv")
    if not os.path.exists(manifest):
        print(f"FAIL {split_dir}: no manifest.csv")
        return False

    with open(manifest) as f:
        rows = list(csv.DictReader(f))

    errors: list[str] = []
    ids = Counter()
    archs = Counter()
    shifts, gtc = [], []

    if sample and sample < len(rows):
        pick = set(np.random.default_rng(seed).choice(len(rows), sample, replace=False).tolist())
    else:
        pick = set(range(len(rows)))

    for n, r in enumerate(rows):
        i = r["id"]
        ids[i] += 1
        archs[r["architecture"]] += 1

        for key in ("reference_path", "search_path"):
            path = os.path.join(split_dir, r[key])
            if n not in pick:
                continue
            if not os.path.exists(path):
                errors.append(f"id={i}: missing {key} -> {r[key]}")
                continue
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # A compact pool stores the 100x100 template in reference/; the
            # search frame is always full resolution.
            want = (TEMPLATE_HW if (key == "reference_path"
                                    and r.get("reference_px") == str(TEMPLATE_HW[0]))
                    else EXPECTED_HW)
            if img is None:
                errors.append(f"id={i}: unreadable {r[key]}")
            elif img.shape != want:
                errors.append(f"id={i}: {key} shape {img.shape} != {want}")

        # Target box must sit fully inside the search frame.
        bx, by = float(r["gt_box_x"]), float(r["gt_box_y"])
        bw, bh = float(r["gt_box_w"]), float(r["gt_box_h"])
        if (bw, bh) != (EXPECTED_BOX, EXPECTED_BOX):
            errors.append(f"id={i}: box {bw}x{bh} != {EXPECTED_BOX}^2")
        if not (0 <= bx <= EXPECTED_HW[1] - bw and 0 <= by <= EXPECTED_HW[0] - bh):
            errors.append(f"id={i}: box origin ({bx},{by}) out of frame")

        cx, cy = float(r["gt_x_corr"]), float(r["gt_y_corr"])
        if not (0 <= cx < EXPECTED_HW[1] and 0 <= cy < EXPECTED_HW[0]):
            errors.append(f"id={i}: corrected centre ({cx:.1f},{cy:.1f}) outside frame")
        gtc.append((cx, cy))
        shifts.append(float(r["label_shift_px"]))

    dupes = [k for k, v in ids.items() if v > 1]
    if dupes:
        errors.append(f"duplicate ids: {dupes[:10]}")

    # Multi-crop splits intentionally share one search frame across several
    # reference crops, so compare against distinct paths rather than row count.
    n_ref = len({r["reference_path"] for r in rows})
    n_src = len({r["search_path"] for r in rows})
    stray_ref = len(os.listdir(os.path.join(split_dir, "reference"))) - n_ref
    stray_src = len(os.listdir(os.path.join(split_dir, "search"))) - n_src
    if stray_ref or stray_src:
        errors.append(f"orphaned files: reference {stray_ref:+d}, search {stray_src:+d}")
    if n_ref != len(rows):
        errors.append(f"duplicate reference paths: {len(rows)} rows, {n_ref} distinct")

    shifts = np.array(shifts)
    gtc = np.array(gtc)
    name = os.path.basename(split_dir.rstrip("/"))
    status = "OK  " if not errors else "FAIL"
    if len(pick) < len(rows):
        print(f"     (images checked on {len(pick)} of {len(rows)} rows; "
              f"geometry checked on all)")
    print(f"{status} {name:12s} {len(rows):5d} samples  "
          f"| label shift (gt -> corrected): mean {shifts.mean():.2f}px  p95 {np.percentile(shifts, 95):.2f}px  max {shifts.max():.2f}px")
    print(f"     centre spread: x {gtc[:, 0].min():.0f}-{gtc[:, 0].max():.0f}  "
          f"y {gtc[:, 1].min():.0f}-{gtc[:, 1].max():.0f}   "
          f"architectures: {len(archs)} ({min(archs.values())}-{max(archs.values())} each)")

    for e in errors[:15]:
        print(f"     ! {e}")
    if len(errors) > 15:
        print(f"     ... and {len(errors) - 15} more")
    return not errors


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("splits", nargs="+")
    p.add_argument("--sample", type=int, default=0,
                   help="open the images of only this many randomly chosen rows "
                        "(0 = all). Row-level geometry checks stay exhaustive.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    ok = all([check_split(t, args.sample, args.seed) for t in args.splits])
    print("\nall splits passed" if ok else "\nSOME SPLITS FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

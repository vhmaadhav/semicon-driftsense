#!/usr/bin/env python3
"""Honest arm comparison: threshold calibrated on one half, scored on the other.

Why this file exists
--------------------
The first two attempts at scoring the Phase 3 arms both had a methodology
problem, and the second one only became visible once the shipped rubric was
vendored in:

  1. The original script reimplemented the rubric inline and scored every arm
     with a *swept, oracle* rejection threshold. That flatters whichever arm has
     the best achievable peak, not the one that is better at a real operating
     point. It also credited declined present pairs with their localisation,
     which the real scorer does not (register.py zero-fills a declined row).

  2. With the real rubric at a FIXED threshold, the answer flips depending on
     which threshold you pick -- because each representation produces a
     different score *distribution*, so a single numeric threshold is not
     comparable across arms. That is the same unit-system mismatch the repo
     already documents for LEGACY_FALLBACK_THRESHOLD vs SHIPPED_THRESHOLD.

The defensible protocol, which this script implements:

  * Split the pairs in half (stratified so both halves keep the present/absent
    ratio).
  * For each arm, pick the threshold that maximises the shipped rubric's
    SUBTOTAL on the CALIBRATION half only.
  * Score that fixed threshold on the held-out half.
  * Report the held-out subtotal. No arm ever sees the half it is judged on.

Every arm is now compared at its OWN calibrated operating point, which is how a
real submission would ship, and the held-out number cannot be gamed by picking
a lucky threshold.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase3_ab import edges, cad_to_sem, zncc_peak  # noqa: E402
import vendor_rubric  # noqa: E402

ARMS = ("A_intensity", "B_edges", "C_cad2sem", "D_cad2sem_edges")
GRID = np.round(np.arange(0.02, 0.98, 0.02), 4)


def subtotal(df, thr) -> float:
    res, _ = vendor_rubric.score(df, thr, quiet=True)
    loc = res["localisation"][1]
    sc = res["scale"][1]
    rc = res["rotation"][1]
    f1 = res["rejection"][1]
    cal = res["calibration"][1]
    vals = [loc, sc, rc, f1, cal]
    if any(not np.isfinite(v) for v in [f1, cal]):   # f1/auc need both classes
        return float("nan")
    return float(sum(vals))


def build(root: str):
    refs = {k: v for k, v in np.load(os.path.join(root, "cad_raster_cache.npz")).items()}
    rows = list(csv.DictReader(open(os.path.join(root, "manifest.csv"))))
    out = {a: [] for a in ARMS}
    for r in rows:
        ref = refs.get(r["id"])
        if ref is None:
            continue
        sea = cv2.imread(os.path.join(root, r["search_path"]), cv2.IMREAD_GRAYSCALE)
        is_p = r["match_found"] == "True"
        sem = cad_to_sem(ref, seed=int(r["id"]), edge_strength=0.30)
        inputs = {
            "A_intensity":     (ref, sea),
            "B_edges":         (edges(ref), edges(sea)),
            "C_cad2sem":       (sem, sea),
            "D_cad2sem_edges": (edges(sem), edges(sea)),
        }
        for a, (t, s) in inputs.items():
            sc, x, y, _ = zncc_peak(s, t)
            out[a].append({
                "pair_id": r["id"], "gt_found": 1 if is_p else 0, "score": float(sc),
                "x": x, "y": y,
                "gt_x": float(r["gt_x"]) if is_p else 0.0,
                "gt_y": float(r["gt_y"]) if is_p else 0.0,
                "scale": 10.0, "gt_scale": 10.0, "theta": 0.0, "gt_rot": 0.0,
            })
    return {a: pd.DataFrame(v) for a, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/phase3_200")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=5,
                    help="repeat the split to average out split luck")
    args = ap.parse_args()
    cv2.setNumThreads(args.threads)

    fr = build(args.root)
    n = len(fr["A_intensity"])
    print(f"n = {n} pairs ({int(fr['A_intensity'].gt_found.sum())} present / "
          f"{int((fr['A_intensity'].gt_found == 0).sum())} absent)")
    print(f"protocol: {args.seeds} stratified 50/50 splits; threshold chosen on "
          f"the calibration half, scored on the held-out half\n")

    rows_out = {a: [] for a in ARMS}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        idx = np.arange(n)
        pos = idx[fr["A_intensity"].gt_found.values == 1]
        neg = idx[fr["A_intensity"].gt_found.values == 0]
        rng.shuffle(pos); rng.shuffle(neg)
        cal_idx = np.concatenate([pos[:len(pos) // 2], neg[:len(neg) // 2]])
        ho_idx = np.setdiff1d(idx, cal_idx)

        for a in ARMS:
            d = fr[a]
            cal_df, ho_df = d.iloc[cal_idx], d.iloc[ho_idx]
            # calibrate on the calibration half only
            best_t, best_v = 0.5, -np.inf
            for t in GRID:
                v = subtotal(cal_df, float(t))
                if np.isfinite(v) and v > best_v:
                    best_v, best_t = v, float(t)
            ho_v = subtotal(ho_df, best_t)
            rows_out[a].append((best_t, ho_v))

    print(f"{'arm':>16} {'thr(med)':>9} {'held-out subtotal':>18} {'std':>7} {'vs A':>8}")
    print("-" * 62)
    base = np.array([v for _, v in rows_out["A_intensity"]], dtype=float)
    for a in ARMS:
        vals = np.array([v for _, v in rows_out[a]], dtype=float)
        thrs = np.array([t for t, _ in rows_out[a]], dtype=float)
        delta = "" if a == "A_intensity" else f"{vals.mean() - base.mean():+8.2f}"
        print(f"{a:>16} {np.median(thrs):>9.2f} {vals.mean():>18.2f} "
              f"{vals.std():>7.2f} {delta:>8}")

    print(f"\n(85-point measurable subtotal: loc 40 + pose 20 + rejection 15 + "
          f"calibration 10)")
    print(f"pose is a constant here -- rotation is pinned at 0 and scale at 10,")
    print(f"so all four arms score the same 20 points and it cancels in 'vs A'.")


if __name__ == "__main__":
    main()

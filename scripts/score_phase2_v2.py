#!/usr/bin/env python3
"""Score register.py predictions on the mentor's Phase 2 v2 dataset.

The dataset (both zips combined) lives in phase2_v2/, git-ignored: it is
confidential and carries the scoring key. Rubric is scripts/grade_emulation.py
-> rubric() unchanged, so the numbers are comparable with earlier Phase 2
results. Set D (optical, bonus only) is reported per pair but kept out of the
grayscale total, as in the blind grade.

    python register.py --input phase2_v2/pairs.csv --output phase2_v2/eval/predictions.csv
    python scripts/score_phase2_v2.py
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.grade_emulation import LOC_TIERS, rubric, tier  # noqa: E402
from driftsense.config import SHIPPED_THRESHOLD  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="phase2_v2")
    ap.add_argument("--pred", default=None, help="default: <data>/eval/predictions.csv")
    ap.add_argument("--threshold", type=float, default=SHIPPED_THRESHOLD)
    a = ap.parse_args()

    pred = pd.read_csv(a.pred or os.path.join(a.data, "eval", "predictions.csv"), comment="#")
    gt = pd.read_csv(os.path.join(a.data, "ground_truth.csv"))
    jury = pd.read_csv(os.path.join(a.data, "manifest_jury.csv"))[["pair_id", "set", "architecture", "severity"]]
    df = (gt.rename(columns={"present": "gt_found", "x": "gt_x", "y": "gt_y",
                             "theta": "gt_rot", "scale": "gt_scale"})
          .merge(jury, on="pair_id").merge(pred, on="pair_id", how="left"))
    if df["score"].isna().any():
        raise SystemExit(f"missing predictions for: {df.loc[df.score.isna(), 'pair_id'].tolist()}")
    # Absent pairs have no pose; give them the prediction so pose errors stay finite.
    for c, g in (("gt_x", "x"), ("gt_y", "y"), ("gt_rot", "theta"), ("gt_scale", "scale")):
        df[c] = df[c].fillna(df[g])
    df.loc[df.gt_scale == 0, "gt_scale"] = 1.0

    pd.set_option("display.width", 200)
    view = df.assign(
        err_px=lambda d: ((d.x - d.gt_x) ** 2 + (d.y - d.gt_y) ** 2) ** 0.5,
        loc_credit=lambda d: [tier(e, LOC_TIERS) if g == 1 else float("nan") for e, g in zip(d.err_px, d.gt_found)],
        scale_err_pct=lambda d: 100 * (d.scale - d.gt_scale).abs() / d.gt_scale,
        rot_err=lambda d: (d.theta - d.gt_rot).abs(),
    )
    view.loc[view.gt_found == 0, ["err_px", "scale_err_pct", "rot_err"]] = float("nan")
    cols = ["pair_id", "set", "severity", "architecture", "gt_found", "found", "score",
            "err_px", "loc_credit", "scale_err_pct", "rot_err"]
    print(view[cols].round(3).to_string(index=False))

    gray = df[df["set"].isin(["A", "B", "C"])].reset_index(drop=True)
    r = rubric(gray, a.threshold)
    print(f"\nthreshold {a.threshold:g}  (A/B/C grayscale, n={r['n']}, present={r['n_present']})")
    print(f"  localisation  loc_A {r['loc_A']:.3f}  loc_B {r['loc_B']:.3f}  -> {40 * r['loc']:.2f} / 40")
    print(f"  scale         {r['scale']:.3f}  -> {10 * r['scale']:.2f} / 10")
    print(f"  rotation      {r['rot']:.3f}  -> {10 * r['rot']:.2f} / 10")
    print(f"  rejection F1  {r['f1_reject']:.3f}  -> {15 * r['f1_reject']:.2f} / 15")
    print(f"  calib AUC     {r['auc']:.3f}  -> {10 * r['auc']:.2f} / 10")
    print(f"  TOTAL         {r['total']:.2f} / 85")


if __name__ == "__main__":
    main()

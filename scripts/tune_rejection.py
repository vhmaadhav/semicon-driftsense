#!/usr/bin/env python3
"""Choose the confidence statistic and threshold for the `found` decision.

Rejection is 15 points and calibration a further 10, and both are decided by
one scalar per pair. The shipped pipeline reports the *network* confidence,
but it also computes a full-resolution ZNCC against the winning hypothesis and
throws that number away. This script asks which statistic -- or which
combination -- actually separates present from absent, using results that have
already been computed, so it costs no inference.

Threshold selection is done under an explicit split: the threshold is chosen on
one half of the pairs and scored on the other. Picking a threshold on the same
pairs you report is not a result, and the gap between the two numbers is itself
worth seeing.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def f1_reject(pred_found, gt_found):
    """F1 with *rejection* as the positive class -- the convention the rubric
    implies by stating that a system which never rejects scores zero."""
    tp = int(((pred_found == 0) & (gt_found == 0)).sum())
    fp = int(((pred_found == 0) & (gt_found == 1)).sum())
    fn = int(((pred_found == 1) & (gt_found == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def auc(stat, correct):
    a, b = stat[correct], stat[~correct]
    if not len(a) or not len(b):
        return float("nan")
    return float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())


def best_threshold(stat, gt):
    grid = np.unique(stat)
    if len(grid) > 400:
        grid = np.quantile(stat, np.linspace(0, 1, 400))
    best = (-1.0, grid[0])
    for t in grid:
        f = f1_reject((stat >= t).astype(int), gt)
        if f > best[0]:
            best = (f, float(t))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--shipped-threshold", type=float, default=0.25)
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d = d[d["set"].isin(["A", "B", "C"])].reset_index(drop=True)   # the 180-pair analogue
    d["err"] = np.where(d.gt_found == 1, np.hypot(d.x - d.gt_x, d.y - d.gt_y), np.nan)
    correct = np.where(d.gt_found == 1, (d.err <= 5).fillna(False), False)
    gt = d.gt_found.values

    sc, zn = d.score.values, d.zncc.values
    stats = {
        "score (shipped)": sc,
        "zncc": zn,
        "min(score, zncc)": np.minimum(sc, zn),
        "score * zncc": sc * zn,
        "sqrt(score*zncc)": np.sqrt(np.clip(sc * zn, 0, None)),
        "0.5*(score+zncc)": 0.5 * (sc + zn),
    }
    if "peak_ratio" in d.columns and d.peak_ratio.notna().any():
        pr = np.nan_to_num(d.peak_ratio.values, nan=0.0)
        # A contested peak (runner-up close to the winner) is weak evidence
        # regardless of how high the winner scored, so it belongs as a
        # multiplicative discount rather than a separate threshold.
        stats["zncc * (1-peak_ratio)"] = zn * (1.0 - pr)
        stats["min(score,zncc)*(1-pr)"] = np.minimum(sc, zn) * (1.0 - pr)
    if "pose_peak" in d.columns and d.pose_peak.notna().any():
        stats["min(zncc, pose_peak)"] = np.minimum(zn, np.nan_to_num(d.pose_peak.values, nan=0.0))

    print(f"{len(d)} grayscale pairs  ({int((gt==1).sum())} present, {int((gt==0).sum())} absent)\n")
    print(f"{'statistic':<20}{'AUC':>8}{'best F1':>10}{'@thr':>9}"
          f"{'held-out F1':>13}{'ship F1':>9}")
    print("-" * 69)

    # Alternate rows into two folds so both halves keep the set mix.
    fold = np.arange(len(d)) % 2
    for name, s in stats.items():
        s = np.nan_to_num(s, nan=-9.0)
        bf, bt = best_threshold(s, gt)
        # honest: pick on fold 0, score on fold 1, and vice versa, then average
        held = []
        for f in (0, 1):
            _, t = best_threshold(s[fold == f], gt[fold == f])
            held.append(f1_reject((s[fold != f] >= t).astype(int), gt[fold != f]))
        ship = f1_reject((s >= a.shipped_threshold).astype(int), gt) \
            if name == "score (shipped)" else float("nan")
        print(f"{name:<20}{auc(s, correct):>8.4f}{bf:>10.4f}{bt:>9.4f}"
              f"{np.mean(held):>13.4f}{ship:>9.4f}")

    print("\n'best F1' is an upper bound (threshold chosen on the same pairs it is\n"
          "scored on). 'held-out F1' is the honest figure: threshold picked on one\n"
          "half, scored on the other, averaged both ways.")

    # ---- calibration: two readings of "per-pair correctness" ---------------
    # The rubric says the score column is scored by "AUC against per-pair
    # correctness". For a *present* pair, correct plainly means localised
    # within 5 px. For an *absent* pair it is ambiguous:
    #   (a) the pair is simply not localisable, so it counts as incorrect --
    #       a low score there helps the AUC;
    #   (b) correctness means the whole row is right, so an absent pair we
    #       correctly rejected counts as CORRECT -- and our low score there
    #       now works against us.
    # These are not close together, and we cannot satisfy both with one
    # monotonic column, so the exposure is worth stating rather than assuming
    # the favourable reading.
    print("\nCalibration AUC under both readings of 'per-pair correctness':")
    pred_found = (np.nan_to_num(d.score.values, nan=-9.0) >= a.shipped_threshold).astype(int)
    err_ok = (d.err <= 5).fillna(False).values
    read_a = np.where(gt == 1, err_ok, False)
    read_b = np.where(gt == 1, err_ok, pred_found == 0)
    for name, s in stats.items():
        s = np.nan_to_num(s, nan=-9.0)
        print(f"  {name:<20} (a) absent=incorrect {auc(s, read_a):.4f}   "
              f"(b) correct-reject=correct {auc(s, read_b):.4f}")
    print("  Reading (b) is the downside case; if it is much worse, that is a\n"
          "  10-point exposure created by an ambiguity in the brief, not by the model.")

    # What the shipped threshold costs on the shipped statistic.
    s = np.nan_to_num(d.score.values, nan=-9.0)
    bf, bt = best_threshold(s, gt)
    ship = f1_reject((s >= a.shipped_threshold).astype(int), gt)
    print(f"\nShipped threshold {a.shipped_threshold} on the shipped statistic: "
          f"F1 {ship:.4f}  ({15*ship:.2f} pts)")
    print(f"Best threshold {bt:.4f} on the shipped statistic:            "
          f"F1 {bf:.4f}  ({15*bf:.2f} pts)")
    print(f"  -> mis-set threshold is worth {15*(bf-ship):+.2f} points")


if __name__ == "__main__":
    main()

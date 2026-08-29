#!/usr/bin/env python3
"""Pick the `found` threshold against the *total* rubric, not against F1 alone.

The threshold is usually treated as a rejection-only decision, but it is not:
`register.py` writes `x=y=theta=scale=0` whenever it reports `found=0`, so
rejecting a pair that really did contain the target throws away that pair's
localisation credit (40 pts) and its pose credit (20 pts) as well as hurting
rejection F1 (15 pts). A threshold tuned to maximise F1 alone will sit too high.

This script therefore scores the *sum*:

    40*loc(t) + 10*scale(t) + 10*rot(t) + 15*F1(t) + 10*AUC

where loc and pose are evaluated the way the grader will see them -- zero for
any present pair we declined to answer.

Threshold selection is reported both in-sample (an upper bound) and under a
two-fold split (the honest number).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ext import LOC_TIERS, ROT_TIERS, SCALE_TIERS, W_A, W_B, tier  # noqa: E402

SCALE_BOUNDS, ROT_BOUNDS = (8.0, 12.0), (-5.0, 5.0)


def prep(df, clamp=True):
    df = df.copy()
    if clamp:
        df["scale"] = df.scale.clip(*SCALE_BOUNDS)
        df["theta"] = df.theta.clip(*ROT_BOUNDS)
    df["err"] = np.where(df.gt_found == 1, np.hypot(df.x - df.gt_x, df.y - df.gt_y), np.nan)
    df["raw_loc"] = df.err.map(lambda e: tier(e, LOC_TIERS) if np.isfinite(e) else 0.0)
    df["raw_s"] = np.abs(df.scale - df.gt_scale) / df.gt_scale
    df["raw_r"] = np.abs(df.theta - df.gt_rot)
    return df


def points(d, stat, t, breakdown=False):
    """Total measurable points at threshold `t` on confidence `stat`."""
    gray = d[d["set"].isin(["A", "B", "C"])]
    s = stat[d["set"].isin(["A", "B", "C"]).values]
    said_found = s >= t

    pres = gray.gt_found.values == 1
    # A declined answer scores zero on localisation and therefore on pose too.
    loc = np.where(said_found, gray.raw_loc.values, 0.0)

    parts = {}
    for name in ("A", "B"):
        m = pres & (gray["set"].values == name)
        parts[name] = loc[m].mean() if m.any() else np.nan
    L = W_A * parts["A"] + W_B * parts["B"]

    scored = pres & said_found & (gray.raw_loc.values > 0)
    S = np.mean([tier(v, SCALE_TIERS) for v in gray.raw_s.values[scored]]) if scored.any() else 0.0
    R = np.mean([tier(v, ROT_TIERS) for v in gray.raw_r.values[scored]]) if scored.any() else 0.0

    tp = int((~said_found & ~pres).sum())
    fp = int((~said_found & pres).sum())
    fn = int((said_found & ~pres).sum())
    F1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    correct = pres & (gray.err.values <= 5)
    a, b = s[correct], s[~correct]
    AUC = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean()) \
        if len(a) and len(b) else 0.0

    total = 40 * L + 10 * S + 10 * R + 15 * F1 + 10 * AUC
    if breakdown:
        return dict(total=total, loc=L, locA=parts["A"], locB=parts["B"],
                    scale=S, rot=R, f1=F1, auc=AUC, tp=tp, fp=fp, fn=fn)
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--shipped", type=float, default=0.25)
    ap.add_argument("--no-clamp", action="store_true")
    a = ap.parse_args()

    d = prep(pd.read_csv(a.csv), clamp=not a.no_clamp)
    sc, zn = d.score.values, np.nan_to_num(d.zncc.values, nan=0.0)
    stats = {"score (shipped)": sc, "zncc": zn,
             "min(score,zncc)": np.minimum(sc, zn),
             "sqrt(score*zncc)": np.sqrt(np.clip(sc * zn, 0, None))}
    if "peak_ratio" in d.columns and d.peak_ratio.notna().any():
        pr = np.nan_to_num(d.peak_ratio.values, nan=0.0)
        stats["min(score,zncc)*(1-pr)"] = np.minimum(sc, zn) * (1.0 - pr)

    print(f"{len(d)} pairs.  Localisation/pose credit is zeroed on any present pair we decline,")
    print("which is what the grader sees when register.py writes found=0.\n")
    print(f"{'statistic':<26}{'best t':>9}{'in-sample':>11}{'held-out':>10}"
          f"{'loc':>8}{'F1':>8}{'AUC':>8}")
    print("-" * 78)

    fold = np.arange(len(d)) % 2
    best_overall = None
    for name, s in stats.items():
        grid = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 300))
        vals = [points(d, s, t) for t in grid]
        i = int(np.argmax(vals))
        t_best, p_best = float(grid[i]), float(vals[i])

        held = []
        for f in (0, 1):
            sub, other = d.iloc[fold == f], d.iloc[fold != f]
            g = np.quantile(s[fold == f], np.linspace(0.001, 0.6, 200))
            tt = g[int(np.argmax([points(sub, s[fold == f], t) for t in g]))]
            held.append(points(other, s[fold != f], float(tt)))
        b = points(d, s, t_best, breakdown=True)
        print(f"{name:<26}{t_best:>9.4f}{p_best:>11.2f}{np.mean(held):>10.2f}"
              f"{b['loc']:>8.4f}{b['f1']:>8.4f}{b['auc']:>8.4f}")
        if best_overall is None or np.mean(held) > best_overall[1]:
            best_overall = (name, float(np.mean(held)), t_best, s)

    print()
    b0 = points(d, sc, a.shipped, breakdown=True)
    print(f"Shipped: score >= {a.shipped}")
    print(f"  total {b0['total']:.2f}   loc {b0['loc']:.4f} (A {b0['locA']:.4f} B {b0['locB']:.4f})"
          f"   scale {b0['scale']:.4f}  rot {b0['rot']:.4f}  F1 {b0['f1']:.4f}  AUC {b0['auc']:.4f}")
    print(f"  correct-rejections {b0['tp']}   real instances declined {b0['fp']}"
          f"   absent accepted {b0['fn']}")

    name, heldpts, t, s = best_overall
    b1 = points(d, s, t, breakdown=True)
    print(f"\nBest: {name} >= {t:.4f}")
    print(f"  total {b1['total']:.2f}   loc {b1['loc']:.4f} (A {b1['locA']:.4f} B {b1['locB']:.4f})"
          f"   scale {b1['scale']:.4f}  rot {b1['rot']:.4f}  F1 {b1['f1']:.4f}  AUC {b1['auc']:.4f}")
    print(f"  correct-rejections {b1['tp']}   real instances declined {b1['fp']}"
          f"   absent accepted {b1['fn']}")
    print(f"\n  gain vs shipped: {b1['total']-b0['total']:+.2f} points "
          f"(held-out estimate {heldpts:.2f})")


if __name__ == "__main__":
    main()

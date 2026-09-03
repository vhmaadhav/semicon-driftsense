#!/usr/bin/env python3
"""Side-by-side rubric comparison of several `eval_ext.py` result CSVs.

Also prints the diagnostics that decide what to do next rather than only the
headline: where localisation credit is being lost, whether the scale estimate
is biased or merely noisy, and whether the *fixed* rejection threshold is
costing points that a better-placed one would keep.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_ext import LOC_TIERS, ROT_TIERS, SCALE_TIERS, W_A, W_B, tier  # noqa: E402


# The problem statement guarantees the true pose lies in these boxes, and the
# rules explicitly permit hard-coding them. Clipping a reported value into the
# feasible set can therefore only reduce the error -- it is not a heuristic.
SCALE_BOUNDS = (8.0, 12.0)
ROT_BOUNDS = (-5.0, 5.0)


def prep(df, clamp=False):
    df = df.copy()
    if clamp:
        df["scale"] = df.scale.clip(*SCALE_BOUNDS)
        df["theta"] = df.theta.clip(*ROT_BOUNDS)
    df["err"] = np.where(df.gt_found == 1, np.hypot(df.x - df.gt_x, df.y - df.gt_y), np.nan)
    df["loc_credit"] = df.err.map(lambda e: tier(e, LOC_TIERS) if np.isfinite(e) else np.nan)
    df["s_rel"] = (df.scale - df.gt_scale) / df.gt_scale          # signed
    df["s_err"] = df.s_rel.abs()
    df["r_err"] = (df.theta - df.gt_rot).abs()
    return df


def summary(df, threshold):
    gray = df[df["set"].isin(["A", "B", "C"])]
    present = gray[gray.gt_found == 1]
    parts = {s: present[present["set"] == s].loc_credit.mean() for s in ("A", "B")}
    loc = W_A * parts["A"] + W_B * parts["B"]
    ok = present[present.loc_credit > 0]
    sc = ok.s_err.map(lambda v: tier(v, SCALE_TIERS)).mean()
    rc = ok.r_err.map(lambda v: tier(v, ROT_TIERS)).mean()

    def f1_at(t, positive="reject"):
        pf = gray.score >= t
        if positive == "reject":
            tp = int((~pf & (gray.gt_found == 0)).sum())
            fp = int((~pf & (gray.gt_found == 1)).sum())
            fn = int((pf & (gray.gt_found == 0)).sum())
        else:
            tp = int((pf & (gray.gt_found == 1)).sum())
            fp = int((pf & (gray.gt_found == 0)).sum())
            fn = int((~pf & (gray.gt_found == 1)).sum())
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    f1 = f1_at(threshold)
    f1len = f1_at(threshold, "present")
    bf1 = max(f1_at(t) for t in np.unique(gray.score.values))
    correct = np.where(gray.gt_found == 1, (gray.err <= 5).fillna(False), False)
    a, b = gray.score.values[correct], gray.score.values[~correct]
    auc = float((a[:, None] > b[None, :]).mean() + 0.5 * (a[:, None] == b[None, :]).mean())
    return dict(locA=parts["A"], locB=parts["B"], loc=loc, scale=sc, rot=rc,
                f1=f1, f1len=f1len, bf1=bf1, auc=auc,
                pts=40 * loc + 10 * sc + 10 * rc + 15 * f1 + 10 * auc,
                a5=100 * (present.err <= 5).mean(), a1=100 * (present.err <= 1).mean(),
                med=present.err.median(),
                s_med=100 * ok.s_err.median(), s_bias=100 * ok.s_rel.median(),
                r_med=ok.r_err.median(), secs=df.secs.median())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="label=path.csv")
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--clamp", action="store_true",
                    help="clip reported scale/theta into the disclosed bounds")
    ap.add_argument("--clamp-compare", action="store_true",
                    help="show every run both with and without clamping")
    a = ap.parse_args()

    runs = {}
    for spec in a.csvs:
        label, path = spec.split("=", 1)
        raw = pd.read_csv(path)
        if a.clamp_compare:
            runs[label] = prep(raw)
            runs[label + "+clamp"] = prep(raw, clamp=True)
        else:
            runs[label] = prep(raw, clamp=a.clamp)

    rows = {k: summary(v, a.threshold) for k, v in runs.items()}
    keys = [("loc", "loc credit (.45A+.55B)"), ("locA", "  set A credit"),
            ("locB", "  set B credit"), ("a5", "  %<=5px"), ("a1", "  %<=1px"),
            ("med", "  median err px"),
            ("scale", "scale credit"), ("s_med", "  median |err| %"),
            ("s_bias", "  median signed err %"),
            ("rot", "rot credit"), ("r_med", "  median |err| deg"),
            ("f1", f"rejection F1 @{a.threshold}"), ("f1len", "  lenient F1(present)"),
            ("bf1", "  best-possible F1"),
            ("auc", "calibration AUC"),
            ("pts", "POINTS (of 85 measurable)")]
    w = max(len(v) for _, v in keys) + 2
    print(f"{'':<{w}}" + "".join(f"{k:>16}" for k in runs))
    print("-" * (w + 16 * len(runs)))
    for key, label in keys:
        line = f"{label:<{w}}"
        for k in runs:
            v = rows[k][key]
            line += f"{v:>16.4f}" if abs(v) < 100 else f"{v:>16.2f}"
        print(line)
        if key in ("med", "s_bias", "r_med", "bf1"):
            print()

    # ---- where localisation credit is actually lost ------------------------
    print("\nLocalisation credit lost, by tier (worst run first):")
    for k, df in runs.items():
        p = df[(df.gt_found == 1) & df["set"].isin(["A", "B"])]
        n = len(p)
        band = {"<=1px": (p.err <= 1).sum(), "1-2px": ((p.err > 1) & (p.err <= 2)).sum(),
                "2-3px": ((p.err > 2) & (p.err <= 3)).sum(),
                "3-5px": ((p.err > 3) & (p.err <= 5)).sum(), ">5px": (p.err > 5).sum()}
        print(f"  {k:<12}" + "  ".join(f"{b} {100*c/n:5.1f}%" for b, c in band.items()))

    # ---- per-set / per-severity localisation ------------------------------
    print("\n%<=5px by set and severity level:")
    for k, df in runs.items():
        p = df[df.gt_found == 1]
        print(f"  {k}")
        for s in ("A", "B", "D"):
            q = p[p["set"] == s]
            if not len(q):
                continue
            per = "  ".join(f"sev{int(lv)} {100*(g.err<=5).mean():5.1f}%"
                            for lv, g in q.groupby("severity") if lv > 0)
            print(f"    set {s} ({len(q):4d}): overall {100*(q.err<=5).mean():5.1f}%   {per}")


if __name__ == "__main__":
    main()

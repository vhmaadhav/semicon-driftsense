#!/usr/bin/env python3
"""Score a fitted rejector against the full rubric on the evaluation set.

`fit_rejector.py` reports held-out F1 and AUC on the *training* shards it was
fitted on. That is the right way to choose the model, but it is not the number
that matters: the rubric pays 40 points for localisation and 20 for pose, and
`register.py` zeroes both whenever it writes `found=0`. A statistic can raise
rejection F1 and still lose points overall by declining pairs we had located
correctly.

This applies the fitted logistic to an `eval_ext.py` CSV -- a set the fit never
saw -- and reports the total under the same accounting the grader uses, against
the shipped `min(score, zncc)` baseline at its own best threshold. Both
statistics are given their best threshold so the comparison is not rigged by
holding one at a stale operating point.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_threshold import points, prep  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--rejector", default="weights/rejector.json")
    a = ap.parse_args()

    art = json.load(open(a.rejector))
    feats = art["features"]
    d = prep(pd.read_csv(a.csv))

    missing = [f for f in feats if f not in d.columns]
    if missing:
        sys.exit(f"CSV lacks {missing} -- re-run eval_ext.py so the features are recorded")

    X = d[feats].values.astype(float)
    bad = ~np.isfinite(X).all(1)
    if bad.any():
        # A pair whose features could not be computed must not be silently
        # dropped: it still gets a row in the submission. Fall back to the
        # shipped statistic for those rather than deleting them.
        print(f"note: {int(bad.sum())} rows have non-finite features; "
              f"they fall back to min(score, zncc)")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    Z = (X - np.asarray(art["mu"])) / np.asarray(art["sd"])
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    p_absent = 1.0 / (1.0 + np.exp(-Z @ np.asarray(art["w"])))
    fitted = -p_absent

    sc = d.score.values
    zn = np.nan_to_num(d.zncc.values, nan=0.0)
    shipped = np.minimum(sc, zn)
    if bad.any():
        # Rank-align the fallback into the fitted statistic's range.
        lo, hi = np.nanmin(fitted), np.nanmax(fitted)
        # np.ptp, not ndarray.ptp: the method was removed in NumPy 2.0 and the
        # pin is numpy==2.4.6 — this path runs exactly when features are
        # malformed, so it must not crash on its own fallback.
        s_norm = (shipped - shipped.min()) / (np.ptp(shipped) + 1e-9)
        fitted = np.where(bad, lo + s_norm * (hi - lo), fitted)

    print(f"{len(d)} pairs from {os.path.basename(a.csv)}   "
          f"rejector fitted on {art['fit_pairs']} training pairs\n")
    print(f"{'statistic':<26}{'best t':>10}{'total':>9}{'loc':>8}"
          f"{'locB':>8}{'F1':>8}{'AUC':>8}")
    print("-" * 77)

    best = {}
    for name, s in (("shipped min(score,zncc)", shipped), ("fitted rejector", fitted)):
        grid = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 300))
        vals = [points(d, s, t) for t in grid]
        i = int(np.argmax(vals))
        b = points(d, s, float(grid[i]), breakdown=True)
        best[name] = b
        print(f"{name:<26}{grid[i]:>10.4f}{b['total']:>9.2f}{b['loc']:>8.4f}"
              f"{b['locB']:>8.4f}{b['f1']:>8.4f}{b['auc']:>8.4f}")

    g = best["fitted rejector"]["total"] - best["shipped min(score,zncc)"]["total"]
    print(f"\ndelta on the evaluation set: {g:+.2f} points")
    print("(rejection component is 15 pts, calibration 10; the rest of any move "
          "is localisation/pose credit won or lost by answering more or fewer pairs)")
    bonus = "YES" if best["fitted rejector"]["f1"] >= 0.90 else "no"
    print(f"rejection F1 >= 0.90 bonus reached: {bonus} "
          f"(F1 {best['fitted rejector']['f1']:.4f})")


if __name__ == "__main__":
    main()

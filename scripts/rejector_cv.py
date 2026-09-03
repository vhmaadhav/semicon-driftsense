#!/usr/bin/env python3
"""Does a fitted rejector beat min(score, zncc) on the real distribution?

Two earlier attempts to answer this failed for data reasons, not modelling ones:
the first fitted on 200 rows, the second on `data/ext_train` shards whose
reference_px is 100 -- pre-cropped templates that locate_phase2 cannot consume,
so every feature was noise (median error 600 px, zero pairs within 5 px).

This asks the question without needing new data. `eval_ext.py` records all six
features on the 2250-pair evaluation set, so the logistic can be cross-validated
*inside* it: fit on one fold, pick the threshold on that same fold, and score
the other. Nothing about the held-out fold touches the fit, so the number is
out-of-sample even though no extra data was downloaded.

Scored on the total rubric rather than F1, because `register.py` zeroes the 40
localisation and 20 pose points on any pair it declines: a statistic can raise
rejection F1 and still lose points overall.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_rejector import apply_logistic, fit_logistic  # noqa: E402
from optimize_threshold import points, prep  # noqa: E402

ALL = ["score", "zncc", "peak_ratio", "pose_peak", "psr", "apce"]
# Issue #6 features: peak-quality at the winner (rank/band, recorded under
# --features) and the winner's min(score, zncc) margin over the runner-up
# (recorded on every decode path). Older CSVs lack these columns, so feature
# lists are filtered through available() before fitting.
EXTENDED = ALL + ["rank", "band", "margin"]


def available(d, feats, min_finite: float = 0.5):
    """Features this DataFrame can actually be fitted on.

    Existence of the column is not enough: eval_ext always emits rank/band,
    but on a run without --features those columns are all-NaN, and fitting on
    nan_to_num'd zeros would print plausible-looking extended-trial results
    from data that was never computed. A feature is available only if the
    column exists AND at least `min_finite` of its values are finite.
    """
    ok, missing = [], []
    for f in feats:
        if f not in d.columns:
            missing.append(f)
            continue
        finite = np.isfinite(pd.to_numeric(d[f], errors="coerce").values).mean()
        if finite >= min_finite:
            ok.append(f)
        else:
            missing.append(f)
    return ok, missing


def cv(d, build, folds=4, seed=0):
    """Mean held-out rubric total. `build` returns a statistic given (train, test)."""
    rng = np.random.RandomState(seed)
    f = rng.permutation(len(d)) % folds
    tot, f1s, aucs = [], [], []
    for k in range(folds):
        tr, te = f != k, f == k
        s = build(d, tr)
        grid = np.quantile(s[tr][np.isfinite(s[tr])], np.linspace(0.001, 0.6, 200))
        t = float(grid[int(np.argmax([points(d.iloc[tr], s[tr], g) for g in grid]))])
        b = points(d.iloc[te], s[te], t, breakdown=True)
        # Rescale: points() returns the total for whatever subset it is given,
        # and the per-fold totals are already on the same 85-point scale because
        # every component is a mean or a ratio.
        tot.append(b["total"]); f1s.append(b["f1"]); aucs.append(b["auc"])
    return float(np.mean(tot)), float(np.mean(f1s)), float(np.mean(aucs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--folds", type=int, default=4)
    a = ap.parse_args()

    d = prep(pd.read_csv(a.csv))
    sc = d.score.values
    zn = np.nan_to_num(d.zncc.values, nan=0.0)
    feats, missing = available(d, EXTENDED)
    if missing:
        print(f"note: CSV lacks {', '.join(missing)} -- fitted trials restricted to "
              f"{', '.join(feats)}")
    X_all = {f: np.nan_to_num(d[f].values.astype(float), nan=0.0) for f in feats}
    y = (d.gt_found.values == 0).astype(float)      # 1 = should reject

    def shipped(dd, tr):
        return np.minimum(sc, zn)

    def make_logistic(feats):
        def build(dd, tr):
            X = np.column_stack([X_all[f] for f in feats])
            w, mu, sd = fit_logistic(X[tr], y[tr])
            return -apply_logistic(X, w, mu, sd)
        return build

    print(f"{len(d)} pairs, {a.folds}-fold CV. Threshold chosen on the training fold only.\n")
    print(f"{'statistic':<38}{'held-out total':>16}{'F1':>9}{'AUC':>9}")
    print("-" * 72)

    base = cv(d, shipped, a.folds)
    print(f"{'shipped min(score,zncc)':<38}{base[0]:>16.2f}{base[1]:>9.4f}{base[2]:>9.4f}")

    trials = [
        ("logistic: all shipped features", ALL),
        ("logistic: score,zncc", ["score", "zncc"]),
        ("logistic: score,zncc,psr,apce", ["score", "zncc", "psr", "apce"]),
        ("logistic: score,zncc,peak_ratio,pose_peak", ["score", "zncc", "peak_ratio", "pose_peak"]),
    ]
    if {"rank", "band", "margin"} <= set(X_all):
        # The issue-#6 question: do peak-quality features move the
        # present/absent decision out-of-sample?
        trials += [
            ("logistic: shipped + rank,band,margin", feats),
            ("logistic: score,zncc,rank,band,margin",
             ["score", "zncc", "rank", "band", "margin"]),
            ("logistic: score,zncc,margin", ["score", "zncc", "margin"]),
        ]
    best = ("shipped min(score,zncc)", base[0])
    for name, wanted in trials:
        use, missing = available(d, wanted)
        if missing:
            print(f"{name:<38}{'-- skipped (CSV lacks ' + ', '.join(missing) + ')':>40}")
            continue
        r = cv(d, make_logistic(use), a.folds)
        flag = "  <-- best" if r[0] > best[1] else ""
        print(f"{name:<38}{r[0]:>16.2f}{r[1]:>9.4f}{r[2]:>9.4f}{flag}")
        if r[0] > best[1]:
            best = (name, r[0])

    print(f"\nbest: {best[0]}   {best[1] - base[0]:+.2f} points vs shipped")
    if best[0] == "shipped min(score,zncc)":
        print("No fitted combination beats the hand-chosen rule out-of-sample.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Workstream B: calibration/AUC — does a richer feature vector beat the
shipped min(score, zncc) on held-out AUC?

Protocol is IDENTICAL to scripts/rejector_cv.py so the numbers are directly
comparable to its 0.9878 / 0.9917 baselines (see .agents/REJECTOR_FINDINGS.md):
4-fold, RandomState(0).permutation(n) % 4, threshold grid = train-fold
quantiles linspace(0.001, 0.6, 200) chosen on the train fold's TOTAL rubric
(optimize_threshold.points), AUC per optimize_threshold (correct = present
pair within 5 px, vs everything else, ties count 0.5). Logistic = the plain
GD of scripts/fit_rejector.py via driftsense.calibration.fit (same optimiser,
same hyperparameters). Fit on train folds only; every reported number is
held-out unless the table says in-sample.

Variants (d) are new statistics derived from EXISTING columns — no re-running
of inference:
  score_zncc_min      shipped scalar itself (baseline (a))
  score_zncc_prod     score * zncc
  score_zncc_gap      |score - zncc|  (disagreement between the two signals)
  zncc_over_score     zncc / score    (ratio)
  peak_pose_prod      peak_ratio * pose_peak
  n_hyp               number of surviving pose hypotheses

Modes:
  (default)          the CV table
  --convergence-check  optimiser convergence evidence for the report
  --oracle           in-sample oracle F1 ceiling (REJECTOR_FINDINGS Result 2
                     analogue): fit in-sample on all data, sweep every
                     threshold, report max F1
  --retune           threshold re-tune on the FULL 2250 for the frozen fused
                     statistic, against total-rubric semantics, with downward
                     bias (Subtlety 1), reported under BOTH F1 conventions
                     (Subtlety 2)
  --freeze           fit the final constants on the FULL 2250 with IRLS and
                     print the constants to paste into driftsense/calibration.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from driftsense.calibration import FEATURES, fit, fit_irls  # noqa: E402
from fit_rejector import apply_logistic  # noqa: E402
from optimize_threshold import points, prep  # noqa: E402

EXTENDED = FEATURES  # the 9-feature artefact in driftsense/calibration.py
DERIVED = ["score_zncc_min", "score_zncc_prod", "score_zncc_gap",
           "zncc_over_score", "peak_pose_prod", "n_hyp"]

# Shippable feature sets (integrator request). Only sets whose members are
# available at inference on the SHIPPED default decode path
# (verification="zncc", no return_hypotheses) can ship:
#   "6"   the six locate() features (matching.py lines 1164-1166)
#   "7m"  the six + winner_margin (line ~1032, computed on every path)
#   "9"   the 6 + rank/band/margin — NOT shippable on the default path
#         (rank/band need verification != "zncc" or return_hypotheses=True,
#         lines 874-886); kept for comparability with rejector_cv.py and
#         because driftsense/calibration.py is the 9-feature artefact.
SHIP6 = ["score", "zncc", "peak_ratio", "pose_peak", "psr", "apce"]
FEATURE_SETS = {
    "6": SHIP6,
    "7m": SHIP6 + ["margin"],
    "9": EXTENDED,
}


def add_derived(d):
    """New statistics computable from the recorded columns alone."""
    d = d.copy()
    sc, zn = d.score.values, np.nan_to_num(d.zncc.values, nan=0.0)
    d["score_zncc_min"] = np.minimum(sc, zn)
    d["score_zncc_prod"] = sc * zn
    d["score_zncc_gap"] = np.abs(sc - zn)
    d["zncc_over_score"] = np.where(sc > 0, zn / np.maximum(sc, 1e-9), 0.0)
    d["peak_pose_prod"] = (np.nan_to_num(d.peak_ratio.values, nan=0.0)
                           * np.nan_to_num(d.pose_peak.values, nan=0.0))
    if "n_hyp" in d.columns:
        d["n_hyp"] = pd.to_numeric(d.n_hyp, errors="coerce").fillna(0.0)
    else:
        d["n_hyp"] = 0.0
    return d


def available(d, feats, min_finite: float = 0.5):
    """Same availability guard as rejector_cv.available()."""
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
    """rejector_cv.cv, verbatim structure."""
    rng = np.random.RandomState(seed)
    f = rng.permutation(len(d)) % folds
    tot, f1s, aucs = [], [], []
    for k in range(folds):
        tr, te = f != k, f == k
        s = build(d, tr)
        grid = np.quantile(s[tr][np.isfinite(s[tr])], np.linspace(0.001, 0.6, 200))
        t = float(grid[int(np.argmax([points(d.iloc[tr], s[tr], g) for g in grid]))])
        b = points(d.iloc[te], s[te], t, breakdown=True)
        tot.append(b["total"]); f1s.append(b["f1"]); aucs.append(b["auc"])
    return float(np.mean(tot)), float(np.mean(f1s)), float(np.mean(aucs))


def make_logistic(X_all, feats, y):
    def build(dd, tr):
        X = np.column_stack([X_all[f] for f in feats])
        w, mu, sd = fit(X[tr], y[tr])
        return -apply_logistic(X, w, mu, sd)
    return build


def convergence_check(d, y):
    """Document, don't assume, convergence of the GD fitter on the real data."""
    X = np.column_stack([np.nan_to_num(d[f].values.astype(float), nan=0.0)
                         for f in EXTENDED])
    Z, mu, sd = fit.__globals__["standardize"](X)  # same transform fit() uses
    from driftsense.calibration import design
    D = design(Z)
    w = np.zeros(D.shape[1])
    prev = None
    print("GD convergence (full 2250, iters=4000, lr=0.5, l2=1e-3):")
    print(f"{'iter':>6}{'grad-norm':>14}{'max |dw|/iter':>15}{'logloss':>12}")
    for it in range(1, 4001):
        p = 1.0 / (1.0 + np.exp(-D @ w))
        g = D.T @ (p - y) / len(y)
        g[:-1] += 1e-3 * w[:-1]
        step = 0.5 * g
        w = w - step
        if it in (1, 100, 500, 1000, 2000, 3000, 4000):
            ll = -np.mean(y * np.log(np.clip(p, 1e-12, 1))
                          + (1 - y) * np.log(1 - np.clip(p, 1e-12, 1)))
            print(f"{it:>6}{np.linalg.norm(g):>14.3e}{np.max(np.abs(step)):>15.3e}{ll:>12.5f}")
        prev = w.copy()
    # IRLS cross-check — and an honest discrepancy note. IRLS (ridge 1e-6)
    # reaches a LOWER unregularised logloss than GD: the 9 features nearly
    # separate the 2,250 pairs, so the unregularised MLE diverges (norm grows,
    # logloss keeps creeping down) and IRLS's weaker ridge rides that
    # divergence further than GD's l2=1e-3 allows. The two are different
    # objectives, not optimiser noise. The FROZEN constants use GD because
    # that is the optimiser every CV number in this campaign was computed
    # with — the constants correspond to the evidence.
    wi, mui, sdi = fit_irls(X, y)
    Z2 = (X - mui) / sdi
    pi = 1.0 / (1.0 + np.exp(-(np.hstack([Z2, np.ones((len(Z2), 1))]) @ wi)))
    ll_i = -np.mean(y * np.log(np.clip(pi, 1e-12, 1))
                    + (1 - y) * np.log(1 - np.clip(pi, 1e-12, 1)))
    pg = 1.0 / (1.0 + np.exp(-(D @ w)))
    ll_g = -np.mean(y * np.log(np.clip(pg, 1e-12, 1))
                    + (1 - y) * np.log(1 - np.clip(pg, 1e-12, 1)))
    print(f"\nIRLS(ridge 1e-6) logloss {ll_i:.6f} vs GD(l2 1e-3) {ll_g:.6f}: "
          f"NOT the same optimum — near-separable features make the unregularised "
          f"MLE diverge; IRLS's weaker ridge follows it further. Frozen constants "
          f"use GD (the CV protocol's optimiser).")


def oracle(d, y, feats=None):
    """REJECTOR_FINDINGS Result 2 analogue: fit the logistic in-sample on ALL
    data (cheating freely), sweep every threshold, report the max-F1 ceiling."""
    feats = feats or EXTENDED
    X = np.column_stack([np.nan_to_num(d[f].values.astype(float), nan=0.0)
                         for f in feats])
    w, mu, sd = fit(X, y)
    s = -apply_logistic(X, w, mu, sd)
    grid = np.quantile(s, np.linspace(0.001, 0.99, 2000))
    best = max(
        (points(d, s, float(t), breakdown=True)["f1"], float(t)) for t in grid)
    print(f"in-sample oracle F1(reject-positive) ceiling, {len(feats)}-feature "
          f"logistic ({','.join(feats)}): {best[0]:.4f} at t={best[1]:.4f}")
    return best[0]


def retune(d, feats=None):
    """Threshold re-tune for the frozen fused statistic on the FULL 2250,
    against total-rubric semantics with a DOWNWARD bias (Subtlety 1: a
    declined present pair forfeits loc+pose), reported under BOTH F1
    conventions (Subtlety 2)."""
    import json
    feats = feats or EXTENDED
    X = np.column_stack([np.nan_to_num(d[f].values.astype(float), nan=0.0)
                         for f in feats])
    y = (d.gt_found.values == 0).astype(float)
    w, mu, sd = fit(X, y)
    s = -apply_logistic(X, w, mu, sd)   # calibrated confidence, higher = present

    # Raw-scale equivalent, so the threshold can be expressed in calibrate()
    # probability units: s == -P(absent) == calibrate(features).
    grid = np.quantile(s[np.isfinite(s)], np.linspace(0.001, 0.6, 200))

    def f1_present(dd_stat_found, pres):
        tp = int((dd_stat_found & pres).sum())
        fp = int((dd_stat_found & ~pres).sum())
        fn = int((~dd_stat_found & pres).sum())
        return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0

    gray = d["set"].isin(["A", "B", "C"]).values
    pres = d.gt_found.values == 1

    # Total-rubric optimum, then bias DOWNWARD: walk to the next grid points
    # whose held-style total is within 0.05 of the optimum and take the
    # LOWEST threshold (fewest present pairs declined).
    totals = [(float(t), points(d, s, float(t)) ) for t in grid]
    t_opt, p_opt = max(totals, key=lambda x: x[1])
    margin = 0.05
    cands = [t for t, p in totals if p >= p_opt - margin]
    t_ship = float(min(cands))

    def report(t, label):
        b = points(d, s, t, breakdown=True)
        said = (s >= t) & gray
        print(f"\n{label}: threshold {t:.4f}")
        print(f"  total {b['total']:.2f}  loc {b['loc']:.4f}  scale {b['scale']:.4f}  "
              f"rot {b['rot']:.4f}  AUC {b['auc']:.4f}")
        # Both F1 conventions (PHASE2_STATE.md Subtlety 2): reject-positive is
        # what points() scores; present-positive is the briefing-call reading.
        print(f"  F1 reject-positive {b['f1']:.4f}   "
              f"F1 present-positive {f1_present(said, pres & gray):.4f}")
        print(f"  confusion (gray A/B/C): declined-present {b['fp']}  "
              f"accepted-present {int((said & pres).sum())}  "
              f"declined-absent {b['tp']}  accepted-absent {int((~said & ~pres & gray).sum())}")
        return b

    b0 = report(0.18, "shipped threshold 0.18 on min(score,zncc), applied to fused score")
    bo = report(t_opt, f"total-rubric optimum")
    b1 = report(t_ship, f"shipped-choice (downward-biased, >= opt - {margin})")
    art = {"features": feats,
           "threshold_minus_preject_scale": t_ship,
           "threshold_calibrate_probability_units": 1.0 + t_ship,
           "threshold_opt_minus_preject_scale": t_opt,
           "total_opt": p_opt,
           "note": "s = -P(reject) = P(present) - 1; calibrate() returns "
                   "P(present), so reject when calibrate(features) < 1 + t"}
    print("\npaste-ready threshold JSON (both scales):")
    print(json.dumps(art, indent=2))
    print(f"i.e. reject when calibrate(features) < {1.0 + t_ship:.4f}")
    return b0, bo, b1, art


def freeze(d, feats=None):
    """Final constants: GD fit (the CV protocol's optimiser, l2=1e-3) on the
    FULL 2250, printed for pasting into driftsense/calibration.py. Called
    ONLY after the CV numbers are recorded.

    IRLS is deliberately NOT used for the shipped artefact: the features
    nearly separate the data, the unregularised MLE diverges, and IRLS's
    ridge (1e-6) is a different objective than the l2=1e-3 every CV number
    in this campaign was computed under. Constants must correspond to the
    evidence. (See --convergence-check output and B_CALIBRATION_REPORT.md.)
    """
    feats = feats or EXTENDED
    X = np.column_stack([np.nan_to_num(d[f].values.astype(float), nan=0.0)
                         for f in feats])
    y = (d.gt_found.values == 0).astype(float)
    w, mu, sd = fit(X, y)
    # Raw-scale conversion. fit() models P(reject) = sigmoid(a_std·z + b_std);
    # the shipped statistic is P(present) = 1 - P(reject) = sigmoid(-z), so:
    #   P(present) = sigmoid(-w·(x-mu)/sd - w[-1])
    #              = sigmoid(a·x + b) with a = -w[:-1]/sd,
    #                b = -(w[-1] - sum(w[:-1]*mu/sd)).
    a = -w[:-1] / sd
    b = float(-(w[-1] - np.sum(w[:-1] * mu / sd)))
    print(f"# paste-ready constants — GD(l2=1e-3) fit on FULL 2250, post-CV, "
          f"feature set [{', '.join(feats)}]")
    print(f"INTERCEPT = {b!r}")
    print("COEFS = {")
    for f, ai in zip(feats, a):
        print(f"    {f!r}: {ai!r},")
    print("}")
    # Self-check: the raw-scale constants must reproduce the std-space fit.
    # p_std here is P(present) = 1 - sigmoid(Z_std @ w), matching the shipped
    # statistic's convention.
    Z = (X - mu) / sd
    p_std = 1.0 - 1.0 / (1.0 + np.exp(-(np.hstack([Z, np.ones((len(Z), 1))]) @ w)))
    p_raw = 1.0 / (1.0 + np.exp(-(X @ a + b)))
    print(f"\nself-check max |P_std - P_raw| = {np.max(np.abs(p_std - p_raw)):.2e} "
          f"(must be ~1e-15)")
    s = p_raw  # == calibrate(features) per pair
    grid = np.quantile(s, np.linspace(0.001, 0.6, 200))
    tots = [(float(t), points(d, s, float(t))) for t in grid]
    t_opt, p_opt = max(tots, key=lambda x: x[1])
    print(f"raw-scale (probability-unit) total-rubric optimum threshold: "
          f"{t_opt:.4f} (total {p_opt:.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=".agents/ext_features_full.csv")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default=None,
                    help="feature subset to run the protocol on: "
                         "'6' = six locate() features (shippable), "
                         "'7m' = six + margin (shippable), "
                         "'9' = artefact set incl. rank/band (NOT shippable "
                         "on the default decode path; comparability only). "
                         "Default: run the full CV comparison table.")
    ap.add_argument("--convergence-check", action="store_true")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--retune", action="store_true")
    ap.add_argument("--freeze", action="store_true")
    a = ap.parse_args()

    d = add_derived(prep(pd.read_csv(a.csv)))
    y = (d.gt_found.values == 0).astype(float)      # 1 = should reject
    feats_all = EXTENDED + DERIVED
    use, missing = available(d, feats_all)
    if missing:
        print(f"note: CSV lacks {', '.join(missing)}")
    X_all = {f: np.nan_to_num(d[f].values.astype(float), nan=0.0) for f in use}
    sc, zn = d.score.values, np.nan_to_num(d.zncc.values, nan=0.0)

    feats = FEATURE_SETS[a.feature_set] if a.feature_set else EXTENDED
    if a.feature_set:
        # --feature-set <S> [--oracle|--retune|--freeze]: run the SAME
        # protocol end-to-end on that set: CV row, oracle ceiling, threshold
        # re-tune (downward-biased, both conventions), frozen constants.
        print(f"=== feature set '{a.feature_set}' ({len(feats)} features): "
              f"{', '.join(feats)} ===\n")
        r = cv(d, make_logistic(X_all, feats, y), a.folds)
        print(f"4-fold CV (held-out, threshold train-fold-only): "
              f"total {r[0]:.2f}  F1 {r[1]:.4f}  AUC {r[2]:.4f}\n")
        orc = oracle(d, y, feats)
        print()
        retune(d, feats)
        print()
        freeze(d, feats)
        return
    if a.convergence_check:
        convergence_check(d, y)
        return
    if a.oracle:
        oracle(d, y)
        return
    if a.retune:
        retune(d)
        return
    if a.freeze:
        freeze(d)
        return

    print(f"{len(d)} pairs, {a.folds}-fold CV. Threshold chosen on the training fold only.\n")
    print(f"{'statistic':<44}{'held-out total':>16}{'F1':>9}{'AUC':>9}")
    print("-" * 78)

    def shipped(dd, tr):
        return np.minimum(sc, zn)

    rows = []
    base = cv(d, shipped, a.folds)
    print(f"{'(a) shipped min(score,zncc)':<44}{base[0]:>16.2f}{base[1]:>9.4f}{base[2]:>9.4f}")
    rows.append(("(a) shipped min(score,zncc)", base))

    # SHIP-ELIGIBLE features: available on the SHIPPED default path
    # (verification="zncc", no return_hypotheses): score, zncc, peak_ratio,
    # pose_peak, psr, apce (locate() lines 1164-1166) and margin/winner_margin
    # (line ~1032, every path) -- plus anything derivable from those at the
    # feature-extraction step (min/product/gap/ratio/peak*pose). rank and band
    # are NOT on this path (only with verification != "zncc" or
    # return_hypotheses=True, lines 874-886); n_hyp is not in the integrator's
    # available set either. Trials containing rank/band/n_hyp are REFERENCE
    # only (comparability with rejector_cv.py).
    six = ["score", "zncc", "peak_ratio", "pose_peak", "psr", "apce"]
    SHIP_ELIGIBLE = set(six) | {"margin", "score_zncc_min", "score_zncc_prod",
                                "score_zncc_gap", "zncc_over_score",
                                "peak_pose_prod"}
    ship_trials = [
        ("(b)  logistic: 6 shipped features", six),
        ("(b*) logistic: 7 = 6 + margin", six + ["margin"]),
        ("(b*) 7 + score_zncc_min", six + ["margin", "score_zncc_min"]),
        ("(b*) 7 + score_zncc_prod", six + ["margin", "score_zncc_prod"]),
        ("(b*) 7 + score_zncc_gap", six + ["margin", "score_zncc_gap"]),
        ("(b*) 7 + zncc_over_score", six + ["margin", "zncc_over_score"]),
        ("(b*) 7 + peak_pose_prod", six + ["margin", "peak_pose_prod"]),
        ("(b*) 6 + score_zncc_min", six + ["score_zncc_min"]),
        ("(b*) 6 + score_zncc_prod", six + ["score_zncc_prod"]),
        ("(b*) 6 + score_zncc_gap", six + ["score_zncc_gap"]),
        ("(b*) 6 + zncc_over_score", six + ["zncc_over_score"]),
        ("(b*) 6 + peak_pose_prod", six + ["peak_pose_prod"]),
        ("(b*) 7 + min,prod,gap,ratio,peak_pose",
         six + ["margin", "score_zncc_min", "score_zncc_prod", "score_zncc_gap",
                "zncc_over_score", "peak_pose_prod"]),
        ("(b*) 7 + min + prod", six + ["margin", "score_zncc_min",
                                       "score_zncc_prod"]),
    ]
    # Comparability trials: rejector_cv.py's families, incl. rank/band/n_hyp
    # which the shipped default path does NOT compute. Reference only.
    trials = [
        ("(c)  logistic: 9 = 6 + rank,band,margin (ref)", EXTENDED),
        ("(d)  9 + score_zncc_prod (ref)", EXTENDED + ["score_zncc_prod"]),
        ("(d)  9 + score_zncc_gap (ref)", EXTENDED + ["score_zncc_gap"]),
        ("(d)  9 + zncc_over_score (ref)", EXTENDED + ["zncc_over_score"]),
        ("(d)  9 + peak_pose_prod (ref)", EXTENDED + ["peak_pose_prod"]),
        ("(d)  9 + n_hyp (ref)", EXTENDED + ["n_hyp"]),
        ("(b*) 7 + n_hyp (ref: n_hyp not on default path)",
         six + ["margin", "n_hyp"]),
        ("(d)  6 + score_zncc_prod,gap,ratio,peak_pose,n_hyp (ref)",
         six + ["score_zncc_prod", "score_zncc_gap", "zncc_over_score",
                "peak_pose_prod", "n_hyp"]),
        ("(d)  all 15 (ref)", feats_all),
    ]
    best = ("(a) shipped min(score,zncc)", base[0], base[2])
    for name, wanted in ship_trials + trials:
        u, miss = available(d, wanted)
        if miss:
            print(f"{name:<44}{'-- skipped (CSV lacks ' + ', '.join(miss) + ')':>40}")
            continue
        r = cv(d, make_logistic(X_all, u, y), a.folds)
        print(f"{name:<44}{r[0]:>16.2f}{r[1]:>9.4f}{r[2]:>9.4f}")
        rows.append((name, r))
        if name.startswith("(b") and set(u) <= SHIP_ELIGIBLE \
                and r[2] > best[2] + 1e-9:
            best = (name, r[0], r[2])   # SHIP choice: ship-eligible rows only
    print(f"\nSHIP-ELIGIBLE best held-out AUC: {best[0]}  AUC {best[2]:.4f}  "
          f"total {best[1]:.2f}")
    print(f"vs shipped scalar: AUC {base[2]:.4f} -> {best[2]:.4f} "
          f"({10 * (best[2] - base[2]):+.2f} calibration pts)")


if __name__ == "__main__":
    main()

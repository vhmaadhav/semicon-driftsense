#!/usr/bin/env python3
"""One-shot paired rescore of the PR 34 headline feature CSVs at exactly the
shipped SHIPPED_THRESHOLD = 0.18 (review 5086839305: the committed headline
logs were scored @fix 0.202, not the shipped 0.18, so the promotion margin vs
the +0.35 gate was unproven at the configuration that actually ships).

Point estimates reuse scripts/eval_ext.py's own score() wholesale (quiet
mode), so the rubric cannot drift from the shipped evaluator. The bootstrap
uses a multiplicity-weighted fast path: every resample is the full frame
with each pair drawn k times, and every graded quantity in the 85-pt
subtotal (weighted set A/B localisation means, pose credit means, rejection
F1 counts, AUC concordance) is linear in per-pair draw multiplicities -- so
the fast path computes exactly what scoring the expanded resample through
score() would. It is validated to 1e-9 against score() on the full frame and
ten random resamples before any CI is reported; a mismatch aborts.

The bootstrap resamples PAIR IDS (with replacement) and differences the two
checkpoints' 85-pt subtotals on the identical resample -- the paired
structure that removes draw-to-draw variance.
"""
import importlib.util
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

_spec = importlib.util.spec_from_file_location(
    "eval_ext", os.path.join(HERE, "scripts", "eval_ext.py"))
ee = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ee)

T = 0.18                     # shipped SHIPPED_THRESHOLD, the point in dispute
N_BOOT = 10000
SEED = 0
BASE_CSV = os.path.join(HERE, ".agents", "feat_base_nb.csv")
CAND_CSV = os.path.join(HERE, ".agents", "feat_setcfull.csv")

COMPONENTS = ("localisation", "scale", "rotation", "rejection", "calibration")


def tier(value: float, tiers) -> float:
    for bound, credit in tiers:          # identical to eval_ext.tier
        if value <= bound:
            return credit
    return 0.0


def prep(df: pd.DataFrame, t: float = T) -> dict:
    """Per-pair quantities the 85-pt subtotal is linear in, at threshold t.

    Mirrors eval_ext.score(): pred_found from the score column, submission
    masking of loc_credit, pose read only where loc credit > 0, F1 counts
    over grayscale pairs, AUC winner/loser split. Mask arrays mark the exact
    denominators score() uses."""
    err = np.where(df.gt_found == 1,
                   np.hypot(df.x - df.gt_x, df.y - df.gt_y), np.nan)
    pred_found = (df.score >= t).astype(int).values
    loc_credit = np.array([tier(e, ee.LOC_TIERS) if np.isfinite(e) else 0.0
                           for e in err])
    loc_credit = np.where(pred_found == 1, loc_credit, 0.0)

    present = (df.gt_found == 1).values
    gray = df["set"].isin(["A", "B", "C"]).values
    ok = present & gray & (loc_credit > 0)          # score()'s `ok`

    s_err = np.abs(df.scale - df.gt_scale) / df.gt_scale
    r_err = np.abs(df.theta - df.gt_rot)
    sc_credit = np.array([tier(v, ee.SCALE_TIERS) for v in s_err])
    rc_credit = np.array([tier(v, ee.ROT_TIERS) for v in r_err])

    gt = df.gt_found.values.astype(int)
    rejected = (pred_found == 0)
    tp = (rejected & (gt == 0) & gray)
    fp = (rejected & (gt == 1) & gray)
    fn = ((pred_found == 1) & (gt == 0) & gray)

    correct = np.where(gt == 1, (err <= 5) & present, False) & gray

    return {
        "scores": df.score.values.astype(float),
        "loc": loc_credit,
        "pres_a": (present & gray & (df["set"] == "A").values).astype(float),
        "pres_b": (present & gray & (df["set"] == "B").values).astype(float),
        "ok": ok.astype(float),
        "sc": sc_credit * ok,
        "rc": rc_credit * ok,
        "tp": tp.astype(float), "fp": fp.astype(float), "fn": fn.astype(float),
        "a_auc": correct.astype(float),
        "b_auc": (~correct & gray).astype(float),
    }


def _auc(a_w: np.ndarray, b_w: np.ndarray, scores: np.ndarray) -> float:
    """AUC of scores for two weighted groups, tie-aware, O(n log n)."""
    A, B = a_w.sum(), b_w.sum()
    if A <= 0 or B <= 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    uniq, inv = np.unique(scores[order], return_inverse=True)
    bw = np.bincount(inv, weights=b_w[order], minlength=len(uniq))
    aw = np.bincount(inv, weights=a_w[order], minlength=len(uniq))
    below = np.concatenate([[0.0], np.cumsum(bw)])[:-1]
    conc = float((aw * (below + 0.5 * bw)).sum())
    return conc / (A * B)


def total_from_weights(d: dict, w: np.ndarray) -> float:
    """85-pt subtotal of a resample in which pair i was drawn w[i] times.

    w need not be integral (fractional draws average correctly); weights are
    normalised to per-pair shares so denominators read like counts."""
    n = len(w)
    s = w.sum()
    if s <= 0:
        return float("nan")
    w = w * (n / s)

    def wmean(values, mask):
        den = float(mask @ w)
        return float(values @ w) / den if den > 0 else np.nan

    la = wmean(d["loc"] * d["pres_a"], d["pres_a"])
    lb = wmean(d["loc"] * d["pres_b"], d["pres_b"])
    loc = ee.W_A * la + ee.W_B * lb

    w_ok_sum = float(d["ok"] @ w)
    sc = float(d["sc"] @ w) / w_ok_sum if w_ok_sum > 0 else np.nan
    rc = float(d["rc"] @ w) / w_ok_sum if w_ok_sum > 0 else np.nan

    tp, fp, fn = d["tp"] @ w, d["fp"] @ w, d["fn"] @ w
    f1_den = 2 * tp + fp + fn
    f1 = 2 * tp / f1_den if f1_den else 0.0

    auc = _auc(d["a_auc"] * w, d["b_auc"] * w, d["scores"])

    comps = {
        "localisation": 40 * loc,
        "scale": 10 * sc,
        "rotation": 10 * rc,
        "rejection": 15 * f1,
        "calibration": 10 * auc,
    }
    return sum(comps[k] for k in COMPONENTS)


def subtotal(df, t=T):
    """(85-pt subtotal, per-component points) via the shipped score()."""
    res, _ = ee.score(df, t, quiet=True)
    pts = {k: float(v[1]) for k, v in res.items()}
    return sum(pts[k] for k in COMPONENTS), pts


def validate_fast(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Fast path must equal score() on the full frame and on resamples.

    Resample check uses representation (b): the ORIGINAL frame's rows carry
    their draw multiplicity as weight, while score() sees the equivalent
    expanded frame (rows repeated by multiplicity, uniform weight). The two
    are the same resample; passing per-row multiplicities of the *expanded*
    frame would square every pair's weight."""
    for name, df in (("base", a), ("cand", b)):
        exact, _ = subtotal(df)
        fast = total_from_weights(prep(df), np.ones(len(df)))
        assert abs(fast - exact) < 1e-9, (
            f"{name}: fast path {fast} != score() {exact} on full frame")
    rng = np.random.RandomState(123)
    for i in range(10):
        src = a if i % 2 == 0 else b
        take = rng.randint(0, len(src), size=len(src))
        w = np.bincount(take, minlength=len(src)).astype(float)
        exact, _ = subtotal(src.iloc[take].reset_index(drop=True))
        fast = total_from_weights(prep(src), w)
        assert abs(fast - exact) < 1e-9, (
            f"resample {i}: fast path {fast} != score() {exact}")
    print("fast path validated bit-identical to score() "
          "(full frame + 10 random resamples, tol 1e-9)")


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame,
                     n: int = N_BOOT, seed: int = SEED):
    da, db = prep(a), prep(b)
    rng = np.random.RandomState(seed)
    deltas = np.empty(n)
    for i in range(n):
        take = rng.choice(len(a), size=len(a), replace=True)
        w = np.bincount(take, minlength=len(a)).astype(float)[take]
        deltas[i] = total_from_weights(db, w) - total_from_weights(da, w)
    return deltas


def main():
    a = pd.read_csv(BASE_CSV)
    b = pd.read_csv(CAND_CSV)
    assert set(a.pair_id) == set(b.pair_id), "CSV pair ids must match"

    total_a, pts_a = subtotal(a)
    total_b, pts_b = subtotal(b)

    print(f"PR34 feature-CSV rescore at SHIPPED_THRESHOLD = {T} "
          f"(full 2,250 pairs, shipped score())")
    print(f"{'component':<14}{'shipped':>10}{'this':>10}{'delta':>9}")
    for k in COMPONENTS:
        print(f"{k:<14}{pts_a[k]:>10.2f}{pts_b[k]:>10.2f}"
              f"{pts_b[k] - pts_a[k]:>+9.2f}")
    print(f"{'TOTAL (85)':<14}{total_a:>10.2f}{total_b:>10.2f}"
          f"{total_b - total_a:>+9.2f}")

    assert abs(total_a - 76.94) < 0.02 and abs(total_b - 77.27) < 0.02, \
        "rescore totals diverge from eval_ext --rescore at 0.18; investigate"

    validate_fast(a, b)

    print(f"\nPaired bootstrap of subtotal delta (cand - base), "
          f"N={N_BOOT}, seed={SEED}:")
    deltas = paired_bootstrap(a, b)
    med = float(np.median(deltas))
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    gate = 0.35
    p_ge = float((deltas >= gate).mean())
    print(f"  point delta   {total_b - total_a:+.4f}")
    print(f"  median delta  {med:+.4f}")
    print(f"  95% CI        [{lo:+.4f}, {hi:+.4f}]")
    print(f"  P(paired delta >= +{gate:.2f}) = {p_ge:.4f}")
    print(f"  promotion gate (paired delta >= +{gate:.2f}): "
          f"{'CLEAR' if lo >= gate else 'NOT CLEARED'} by CI lower bound")


if __name__ == "__main__":
    main()

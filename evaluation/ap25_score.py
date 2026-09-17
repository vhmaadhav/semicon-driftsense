"""Rubric arithmetic for the AP-25 evaluation, with no rendering import.

Kept separate from `make_eval_report.py` on purpose: the PDF needs reportlab
and matplotlib, but the scoring only needs pandas and numpy. Splitting them
lets `tests/test_eval_report_parity.py` pin this arithmetic against the
official scorer in any environment, including CI, where the PDF toolchain is
not installed.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "ap25")
FIGDIR = os.path.join(HERE, "figures")

# Hex is the single source of truth: matplotlib wants the string, reportlab
# wants a Color. Defining both from one place stops the figure and the table
# palettes drifting apart.
INK_H, MUTED_H = "#1c1c1e", "#5f6368"  # noqa: F841 (re-exported by the renderer)
RULE_H, BAD_H, GOOD_H, WARN_H = "#d4d7dd", "#b3261e", "#1e6f43", "#8a5a00"
BAD_H_ = BAD_H  # noqa: F841


LOC_TIERS = ((1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40))
SCALE_TIERS = ((0.01, 1.00), (0.02, 0.60), (0.05, 0.30))
ROT_TIERS = ((0.25, 1.00), (0.50, 0.60), (1.00, 0.30))
W_A, W_B = 0.45, 0.55


def tier(value, tiers):
    for bound, credit in tiers:
        if value <= bound:
            return credit
    return 0.0


# ---------------------------------------------------------------- styles


# ------------------------------------------------------- data + scoring
def load():
    per = pd.read_csv(os.path.join(DATA, "ap25_per_pair.csv"))
    comp = pd.read_csv(os.path.join(DATA, "ap25_components.csv"))
    base = pd.read_csv(os.path.join(DATA, "ap25_baseline_zncc.csv"))
    ab = pd.read_csv(os.path.join(DATA, "ap25_rowrefine_ab.csv"))
    # The frozen per-pair CSV is the rubric scorer's own dump, so the merged
    # columns carry _x/_y suffixes from pandas' merge: the prediction is
    # `theta_x`, the ground-truth angle `gt_rot`. Normalise to one spelling so
    # the rest of this script reads plainly.
    per = per.rename(columns={"theta_x": "theta", "theta_y": "theta_gt"})
    per = per.merge(base[["pair_id", "x", "y", "score"]],
                    on="pair_id", how="left", suffixes=("", "_base"))
    per = per.merge(ab[["pair_id", "err_rows_off", "err_rows_on"]],
                    on="pair_id", how="left")
    per["err"] = np.where(per.gt_found == 1,
                          np.hypot(per.x - per.gt_x, per.y - per.gt_y), np.nan)
    per["base_err"] = np.where(per.gt_found == 1,
                               np.hypot(per.x_base - per.gt_x,
                                        per.y_base - per.gt_y), np.nan)
    per["dx"] = per.x - per.gt_x
    per["dy"] = per.y - per.gt_y
    per["s_pct"] = 100 * np.abs(per.scale - per.gt_scale) / per.gt_scale
    per["r_err"] = np.abs(per.theta - per.gt_rot)
    per["loc_credit"] = np.where(
        per.pred_found == 1,
        per.err.map(lambda e: tier(e, LOC_TIERS) if np.isfinite(e) else 0.0), 0.0)
    per["base_credit"] = np.where(
        per.score_base >= 0.55,
        per.base_err.map(lambda e: tier(e, LOC_TIERS) if np.isfinite(e) else 0.0), 0.0)
    return per, comp, base, ab


def score_of(per, ref=None):
    """Recompute the rubric from the frozen per-pair CSV.

    Deliberately independent of the scorer's own dump so the PDF cannot simply
    echo a number nobody re-derived. Two details are easy to get wrong and both
    are load-bearing:

    * pose credit is averaged over the A/B/C rows only -- Set D is bonus-only
      and never contributes to the 10+10 pose blocks;
    * the `ok` mask (pairs earning pose credit) is `loc_credit > 0` on the
      *predicted* decision, not on the ground-truth presence.
    """
    pres = per[per.gt_found == 1]
    gray = per[per.set.isin(["A", "B", "C"])]
    A = pres[pres.set == "A"]
    B = pres[pres.set == "B"]
    D = pres[pres.set == "D"]
    a, b = A.loc_credit.mean(), B.loc_credit.mean()
    loc = W_A * a + W_B * b
    ok = gray[(gray.gt_found == 1) & (gray.loc_credit > 0)]
    # `s_err` is the FRACTIONAL scale error and `s_pct` is the same number x100.
    # SCALE_TIERS is written as fractions (0.01 = 1%), so tier against s_err --
    # tiering the percentage here silently sends every pair to the 0.0 bucket.
    sc = ok.s_err.map(lambda v: tier(v, SCALE_TIERS)).mean()
    rc = ok.r_err.map(lambda v: tier(v, ROT_TIERS)).mean()
    tp = int(((gray.pred_found == 0) & (gray.gt_found == 0)).sum())
    fp = int(((gray.pred_found == 0) & (gray.gt_found == 1)).sum())
    fn = int(((gray.pred_found == 1) & (gray.gt_found == 0)).sum())
    f1 = 2 * tp / (2 * tp + fp + fn)
    correct = np.where(gray.gt_found == 1, (gray.err <= 5).fillna(False), False)
    pos, neg = gray.score.values[correct], gray.score.values[~correct]
    auc = float((pos[:, None] > neg[None, :]).mean()
                + 0.5 * (pos[:, None] == neg[None, :]).mean())
    d_credit = D.loc_credit.mean()
    bonus6 = bool(d_credit >= 0.40 and all(v >= 0.50 for v in (a, b, f1, auc)))
    bonus4 = bool(f1 >= 0.90)
    return dict(a=a, b=b, loc=loc, loc_pts=40 * loc, sc=sc, sc_pts=10 * sc,
                rc=rc, rc_pts=10 * rc, f1=f1, rej_pts=15 * f1, tp=tp, fp=fp,
                fn=fn, auc=auc, cal_pts=10 * auc, d=d_credit, bonus6=bonus6,
                bonus4=bonus4,
                subtotal=40 * loc + 10 * sc + 10 * rc + 15 * f1 + 10 * auc,
                total=40 * loc + 10 * sc + 10 * rc + 15 * f1 + 10 * auc
                + 6 * bonus6 + 4 * bonus4, pres=pres, gray=gray, ok=ok)


# ---------------------------------------------------------------- figures

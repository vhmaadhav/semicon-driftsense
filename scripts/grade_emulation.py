#!/usr/bin/env python3
"""Stratified blind-grade emulation for the Phase 2 bonus probability.

PR #24 review blocker 4: eval_ext.py --sample 200 draws an *unstratified*
sample, but the disclosed blind-grade composition is stratified --
A=70, B=70, C=40 (set D=20 is excluded from the grayscale grade; the
rejection F1 runs over exactly the 180 A/B/C pairs). This module reproduces
that composition exactly and runs a stratified bootstrap over a per-pair
results CSV to answer one planning question: how likely is the +6 bonus
(F1 >= 0.90 on the 180-pair grade) at a given found threshold?

Rubric (corrected semantics, identical credit tiers to scripts/eval_ext.py
-> score()):

* localisation: 1 / 0.8 / 0.6 / 0.4 credit at <=1 / 2 / 3 / 5 px, weighted
  0.45 * set A + 0.55 * set B, ZERO for a present pair the system declined
  (register.py writes no pose/location fields for a declined answer);
* pose (scale <=1/2/5% and rotation <=0.25/0.5/1.0 deg, credits 1/0.6/0.3),
  scored only where localisation earned credit;
* rejection F1 on the found flag (present-as-positive) at
  found = score >= t over the 180 grayscale pairs;
* calibration AUC, present-only (correct = present pair localised <=5 px).

Bootstrap: N stratified draws (draw i uses RandomState(seed + i) per set so
draw 0 equals a standalone stratified_draw call with the same seed), the
rubric scored on each draw, and

    P(bonus) = P(F1 >= 0.90)   over draws
    E[total + 4*P(bonus)] = E[total] + 4 * P(bonus)

where total is the 85-point measurable subtotal (loc 40 + pose 20 +
rejection 15 + calibration 10).

Dependencies: stdlib + numpy + pandas only (torch-free on purpose -- this
runs offline on per-pair CSVs, never on images).
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

# Published Phase 2 credit tiers (kept in lockstep with scripts/eval_ext.py).
LOC_TIERS = ((1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40))
SCALE_TIERS = ((0.01, 1.00), (0.02, 0.60), (0.05, 0.30))
ROT_TIERS = ((0.25, 1.00), (0.50, 0.60), (1.00, 0.30))
W_A, W_B = 0.45, 0.55

# Disclosed blind-grade composition. Set D (20 pairs, optical) is excluded
# from the grayscale grade; the rejection F1 is over exactly these 180 pairs.
BLIND_COMPOSITION = {"A": 70, "B": 70, "C": 40}
BONUS_F1 = 0.90          # the +6 bonus gate used for planning: F1 >= 0.90
BONUS_WEIGHT = 4.0       # E[total + 4*P(bonus)] weighting (6 pts discounted)
DEFAULT_THRESHOLDS = (0.1587, 0.18, 0.25)

REQUIRED_COLUMNS = (
    "pair_id", "set", "gt_found", "score",
    "x", "y", "gt_x", "gt_y",
    "scale", "theta", "gt_scale", "gt_rot",
)


def tier(value, tiers):
    """Credit for value under tiers ((bound, credit), ...): the credit of
    the first bound the value does not exceed; 0.0 past the last bound."""
    for bound, credit in tiers:
        if value <= bound:
            return credit
    return 0.0


# ---------------------------------------------------------------------------
# Stratified draw
# ---------------------------------------------------------------------------

def stratified_draw(df, quotas=None, seed=0):
    """Exact stratified sample: per-set seeded shuffle (np.random.RandomState
    permutation), take exactly the quota, union. Order-stable (rows keep the
    original frame order).

    Raises ValueError (clearly) if a set is smaller than its quota.
    """
    quotas = dict(BLIND_COMPOSITION if quotas is None else quotas)
    parts = []
    for set_name in sorted(quotas):
        quota = quotas[set_name]
        sub = df[df["set"] == set_name]
        if len(sub) < quota:
            raise ValueError(
                f"set {set_name!r} has {len(sub)} rows but the blind-grade "
                f"quota needs {quota}; cannot draw an exact stratified "
                f"sample from this frame"
            )
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(sub))[:quota]
        parts.append(sub.iloc[np.sort(idx)])
    out = pd.concat(parts)
    # Order-stable union: restore the original frame order.
    return out.iloc[np.argsort(out.index.values, kind="stable")]


# ---------------------------------------------------------------------------
# Rubric (corrected semantics) on a grayscale A/B/C frame
# ---------------------------------------------------------------------------

def _prepare(df):
    """Precompute per-pair arrays the rubric needs (NaN-safe)."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    gt_found = df["gt_found"].to_numpy().astype(int)
    score = df["score"].to_numpy().astype(float)
    err = np.where(
        gt_found == 1,
        np.hypot(df["x"].to_numpy() - df["gt_x"].to_numpy(),
                 df["y"].to_numpy() - df["gt_y"].to_numpy()),
        np.nan,
    )
    raw_loc = np.where(
        np.isfinite(err),
        np.array([tier(e, LOC_TIERS) for e in err], dtype=float),
        0.0,
    )
    s_err = np.abs(df["scale"].to_numpy() - df["gt_scale"].to_numpy()) / df["gt_scale"].to_numpy()
    r_err = np.abs(df["theta"].to_numpy() - df["gt_rot"].to_numpy())
    raw_sc = np.array([tier(v, SCALE_TIERS) for v in s_err], dtype=float)
    raw_rc = np.array([tier(v, ROT_TIERS) for v in r_err], dtype=float)
    sets = df["set"].to_numpy()
    return {
        "gt_found": gt_found, "score": score, "err": err,
        "raw_loc": raw_loc, "raw_sc": raw_sc, "raw_rc": raw_rc,
        "isA": sets == "A", "isB": sets == "B",
    }


def _f1(score, gt, t, positive):
    """F1 at found = score >= t; positive='present' or 'reject'."""
    pf = score >= t
    if positive == "reject":
        tp = int(((pf == 0) & (gt == 0)).sum())   # correct reject
        fp = int(((pf == 0) & (gt == 1)).sum())   # rejected a real one
        fn = int(((pf == 1) & (gt == 0)).sum())   # missed an absent
    else:
        tp = int(((pf == 1) & (gt == 1)).sum())
        fp = int(((pf == 1) & (gt == 0)).sum())
        fn = int(((pf == 0) & (gt == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def f1_found(gray, t):
    """F1 on the found flag (present-as-positive) at found = score >= t."""
    return _f1(gray["score"].to_numpy().astype(float),
               gray["gt_found"].to_numpy().astype(int), t, "present")


def f1_reject(gray, t):
    """Conservative planning variant: reject-as-positive F1. eval_ext prints
    both; they are NOT the same metric and must not be quoted as one."""
    return _f1(gray["score"].to_numpy().astype(float),
               gray["gt_found"].to_numpy().astype(int), t, "reject")


def rubric(gray, t):
    """Corrected-semantics rubric on a grayscale (A/B/C) frame at threshold t."""
    p = _prepare(gray)
    gt, score, err = p["gt_found"], p["score"], p["err"]
    said = score >= t
    present = gt == 1
    loc_credit = np.where(present & said, p["raw_loc"], 0.0)

    res = {}
    parts = {}
    for s, mask in (("A", p["isA"]), ("B", p["isB"])):
        sel = mask & present
        parts[s] = float(loc_credit[sel].mean()) if sel.any() else float("nan")
    loc = W_A * parts["A"] + W_B * parts["B"]
    res["loc_A"], res["loc_B"], res["loc"] = parts["A"], parts["B"], loc

    ok = present & (loc_credit > 0)
    res["scale"] = float(p["raw_sc"][ok].mean()) if ok.any() else float("nan")
    res["rot"] = float(p["raw_rc"][ok].mean()) if ok.any() else float("nan")

    res["f1_found"] = _f1(score, gt, t, "present")
    res["f1_reject"] = _f1(score, gt, t, "reject")

    correct = present & (err <= 5)
    a, b = score[correct], score[~correct]
    res["auc"] = (float((a[:, None] > b[None, :]).mean()
                        + 0.5 * (a[:, None] == b[None, :]).mean())
                  if len(a) and len(b) else float("nan"))

    total = (40 * loc + 10 * res["scale"] + 10 * res["rot"]
             + 15 * res["f1_found"] + 10 * res["auc"])
    res["total"] = float(total) if np.isfinite(total) else float("nan")
    res["n"] = len(gray)
    res["n_present"] = int(present.sum())
    return res


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap(df, thresholds=None, draws=10000, seed=0):
    """Stratified bootstrap: draws exact-composition blind grades; per
    threshold, P(F1 >= 0.90) over draws and E[total] + 4*P(bonus).

    Draw i uses np.random.RandomState(seed + i) per set (fixed set order),
    so the draw sequence is deterministic in (seed, draws, frame order) and
    shared across thresholds (which draw is taken does not depend on t).
    """
    if thresholds is None:
        thresholds = list(DEFAULT_THRESHOLDS)
    gray = df[df["set"].isin(BLIND_COMPOSITION)]
    q = BLIND_COMPOSITION
    p = _prepare(gray)
    sets = gray["set"].to_numpy()

    totals = {t: [] for t in thresholds}
    f1s = {t: [] for t in thresholds}

    score_all, gt_all = p["score"], p["gt_found"]
    for i in range(draws):
        idx_parts = []
        for s in sorted(q):
            rng = np.random.RandomState(seed + i)
            block = np.flatnonzero(sets == s)
            idx_parts.append(block[np.sort(rng.permutation(len(block))[:q[s]])])
        idx = np.concatenate(idx_parts)
        sc, gt = score_all[idx], gt_all[idx]
        present = gt == 1
        err_d = p["err"][idx]
        # Positional credit for a present pair; the threshold mask (did the
        # system decline it?) is applied per threshold below.
        loc_credit_all = np.where(present & np.isfinite(err_d),
                                  p["raw_loc"][idx], 0.0)
        isA, isB = p["isA"][idx], p["isB"][idx]
        correct = present & (err_d <= 5)
        for t in thresholds:
            said = sc >= t
            loc_credit = np.where(said, loc_credit_all, 0.0)
            selA, selB = isA & present, isB & present
            locA = float(loc_credit[selA].mean()) if selA.any() else float("nan")
            locB = float(loc_credit[selB].mean()) if selB.any() else float("nan")
            loc = W_A * locA + W_B * locB
            okm = present & (loc_credit > 0)
            sc_credit = float(p["raw_sc"][idx][okm].mean()) if okm.any() else float("nan")
            rc_credit = float(p["raw_rc"][idx][okm].mean()) if okm.any() else float("nan")
            f1 = _f1(sc, gt, t, "present")
            a, b = sc[correct], sc[~correct]
            auc = (float((a[:, None] > b[None, :]).mean()
                         + 0.5 * (a[:, None] == b[None, :]).mean())
                   if len(a) and len(b) else float("nan"))
            total = (40 * loc + 10 * sc_credit + 10 * rc_credit
                     + 15 * f1 + 10 * auc)
            totals[t].append(total)
            f1s[t].append(f1)

    out = []
    for t in thresholds:
        tot = np.asarray(totals[t], dtype=float)
        f1 = np.asarray(f1s[t], dtype=float)
        p_bonus = float((f1 >= BONUS_F1).mean())
        e_total = float(np.nanmean(tot)) if np.isfinite(tot).any() else float("nan")
        out.append({
            "threshold": t,
            "p_f1_ge_bonus": p_bonus,
            "e_total": e_total,
            "e_total_plus_bonus": (e_total + BONUS_WEIGHT * p_bonus
                                   if np.isfinite(e_total) else float("nan")),
            "mean_f1": float(np.mean(f1)),
            "draws": draws,
        })
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Stratified blind-grade emulation (A=70 B=70 C=40; "
                    "F1 over the 180 grayscale pairs) and bonus-probability "
                    "bootstrap over a per-pair results CSV.")
    ap.add_argument("--csv", required=True, help="per-pair results CSV "
                    "(columns: pair_id, set, gt_found, score, x, y, gt_x, "
                    "gt_y, scale, theta, gt_scale, gt_rot)")
    ap.add_argument("--threshold",
                    default=",".join(str(t) for t in DEFAULT_THRESHOLDS),
                    help="comma-separated found thresholds "
                         "(default 0.1587,0.18,0.25)")
    ap.add_argument("--draws", type=int, default=10000,
                    help="bootstrap draws (default 10000)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    thresholds = [float(x) for x in str(a.threshold).split(",") if x.strip()]
    df = pd.read_csv(a.csv)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        ap.error(f"CSV is missing required columns: {missing}")

    gray = df[df["set"].isin(BLIND_COMPOSITION)]
    print(f"CSV: {a.csv}  ({len(df)} pairs; grayscale A/B/C frame = {len(gray)}: "
          + ", ".join(f"{s}={int((gray['set'] == s).sum())}"
                      for s in sorted(BLIND_COMPOSITION)) + ")")
    print("Blind composition: " + ", ".join(f"{s}={q}"
          for s, q in sorted(BLIND_COMPOSITION.items()))
          + f"  (D excluded; F1 over {sum(BLIND_COMPOSITION.values())} pairs)")
    print(f"Bootstrap: {a.draws} stratified draws, seed {a.seed}, "
          f"bonus gate F1 >= {BONUS_F1}")
    print()

    header = f"{'t':>8}{'F1_found':>10}{'F1_rej':>10}{'P(F1>=.90)':>13}" \
             f"{'E[total]':>10}{'E[tot]+4P':>11}"
    print(header)
    print("-" * 62)
    # One bootstrap pass for ALL thresholds: the stratified draws do not
    # depend on t, so the draw sequence is shared (and computed once).
    bs_all = bootstrap(df, thresholds=thresholds, draws=a.draws, seed=a.seed)
    results = []
    for t, bs in zip(thresholds, bs_all):
        f1f, f1r = f1_found(gray, t), f1_reject(gray, t)
        results.append((t, f1f, f1r, bs))
        print(f"{t:>8.4f}{f1f:>10.4f}{f1r:>10.4f}"
              f"{bs['p_f1_ge_bonus']:>13.4f}{bs['e_total']:>10.2f}"
              f"{bs['e_total_plus_bonus']:>11.2f}")
    print("-" * 62)
    best = max(results, key=lambda r: r[3]["e_total_plus_bonus"])
    print(f"argmax E[total + 4*P(bonus)]: t = {best[0]:.4f} "
          f"(E = {best[3]['e_total_plus_bonus']:.2f})")
    print()
    print("Caveat: loc/pose/AUC totals here use the CSV's latent columns "
          "BEFORE the pending Set-D masking fix (Task 1) lands -- they are "
          "provisional. F1 and P(F1 >= 0.90) depend ONLY on the score and "
          "gt_found columns, so the bonus probabilities are final regardless.")
    return results


if __name__ == "__main__":
    main()

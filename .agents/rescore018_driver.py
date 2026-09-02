#!/usr/bin/env python3
"""Paired rescore of the PR 34 headline feature CSVs at exactly the shipped
SHIPPED_THRESHOLD = 0.18 (review 5086839305: the committed headline logs were
scored @fix 0.202, not the shipped 0.18, so the promotion margin vs the +0.35
gate was unproven at the configuration that actually ships).

v2 (review 5087501487): fixes the bootstrap weighting bug -- the old code
computed `np.bincount(take)[take]`, which is the weight of the DRAWN pair at
each draw-order position, not the multiplicity of original row i -- aligns
both frames explicitly by pair_id (set equality proves membership only, not
row order), and replaces the per-resample Python loop with a vectorised
bootstrap (matrix products over blocks of resamples, bounded memory).

Correctness is established by CROSS-VALIDATION against independent
implementations, not by a single self-consistent fast path:

  C1. Full-frame point estimates: fast path == score() to 1e-9.
  C2. Six random resamples: vectorised weights == score() on the expanded
      frame (rows repeated by multiplicity) to 1e-9.
  C3. Vectorised bootstrap == an explicit per-resample reference loop over
      200 identical draws (same seed) to 1e-9.
  C4. Weighted AUC == brute-force pairwise AUC on 30 random weighted cases,
      including duplicated scores, to 1e-9.

The bootstrap resamples PAIR INDICES with replacement and differences the
two checkpoints' 85-pt subtotals on the identical draw -- the paired
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


# --------------------------------------------------------------------------
# Aligned paired frames
# --------------------------------------------------------------------------
def load_aligned() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both CSVs and align them row-for-row by pair_id.

    Review 5087501487: set equality proves membership only, not row order;
    a paired bootstrap differences row i of one frame against row i of the
    other, so the frames MUST be reordered into one shared pair order."""
    a = pd.read_csv(BASE_CSV)
    b = pd.read_csv(CAND_CSV)
    assert len(a) == len(b), f"frame length mismatch: {len(a)} vs {len(b)}"
    merged = a.merge(b, on="pair_id", suffixes=("_a", "_b"), how="inner",
                     validate="one_to_one")
    assert len(merged) == len(a), "pair_id sets differ between the frames"
    for col in ("set", "gt_found", "gt_x", "gt_y", "gt_scale", "gt_rot"):
        same = (merged[f"{col}_a"] == merged[f"{col}_b"]).all()
        assert same, f"ground-truth column {col!r} disagrees between frames"
    a_al = merged[[f"{c}_a" for c in a.columns if c != "pair_id"]] \
        .rename(columns=lambda c: c[:-2])
    b_al = merged[[f"{c}_b" for c in b.columns if c != "pair_id"]] \
        .rename(columns=lambda c: c[:-2])
    a_al.insert(0, "pair_id", merged["pair_id"].values)
    b_al.insert(0, "pair_id", merged["pair_id"].values)
    return a_al, b_al


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
    """85-pt subtotal of a resample in which original pair i was drawn
    exactly w[i] times. w[i] MUST be indexed by original row position --
    the review-5087501487 bug was passing the draw-order vector m[take]
    here instead of the multiplicity vector bincount(take).

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


# --------------------------------------------------------------------------
# Vectorised bootstrap (no loops)
# --------------------------------------------------------------------------
def draw_weights(n: int, draws: int, seed: int) -> np.ndarray:
    """(draws x n) matrix W with W[r, i] = times pair i drawn in resample r.

    Integer bincount semantics per row: identical to
    np.bincount(rng.choice(n, n, replace=True), minlength=n). Implemented
    with one flat bincount over (row, col) linear indices -- O(draws * n)
    memory; the naive (draws, n, n) comparison tensor would need ~8 GB for
    the C3 cross-check and is why v1's validation run ballooned."""
    rng = np.random.RandomState(seed)
    take = rng.choice(n, size=(draws, n), replace=True)
    flat = (take + np.arange(draws)[:, None] * n).ravel()
    return np.bincount(flat, minlength=draws * n).reshape(draws, n) \
        .astype(np.float64)


def bootstrap_vectorised(da: dict, db: dict, draws: int = N_BOOT,
                         seed: int = SEED, block: int = 1000) -> np.ndarray:
    """Paired subtotal deltas for all resamples, fully vectorised.

    Processed in memory-sized blocks of `block` resamples; within a block
    every resample is one matrix product -- no per-resample Python loop."""
    n = len(da["scores"])
    out = np.empty(draws)
    for s in range(0, draws, block):
        e = min(s + block, draws)
        W = draw_weights(n, e - s, seed + s)   # distinct seed per block
        out[s:e] = bootstrap_from_weights(W, da, db)
    return out


def bootstrap_from_weights(W: np.ndarray, da: dict, db: dict) -> np.ndarray:
    """Delta per row of W. Every graded quantity is linear in w, so all
    resamples evaluate as one (draws x n) @ (n,) matrix product."""
    n = W.shape[1]
    scale = n / W.sum(axis=1, keepdims=True)              # per-row share norm
    w = W * scale                                         # rows sum to n

    def component(d):
        loc_a = (d["loc"] * d["pres_a"]) @ w.T / (d["pres_a"] @ w.T)
        loc_b = (d["loc"] * d["pres_b"]) @ w.T / (d["pres_b"] @ w.T)
        loc = ee.W_A * loc_a + ee.W_B * loc_b
        sc = (d["sc"] @ w.T) / (d["ok"] @ w.T)
        rc = (d["rc"] @ w.T) / (d["ok"] @ w.T)
        tp, fp, fn = d["tp"] @ w.T, d["fp"] @ w.T, d["fn"] @ w.T
        f1_den = 2 * tp + fp + fn
        f1 = np.where(f1_den > 0, 2 * tp / np.maximum(f1_den, 1e-300), 0.0)
        return 40 * loc, 10 * sc, 10 * rc, 15 * f1

    la, sa, ra, fa = component(da)
    lb, sb, rb, fb = component(db)

    # AUC per resample: sort the FIXED score column once, then accumulate
    # per-unique-score weighted masses for all resamples at once.
    def auc_row(d):
        wt = w.T                                          # (n, draws)
        a_w = d["a_auc"][:, None] * wt
        b_w = d["b_auc"][:, None] * wt
        A = a_w.sum(axis=0)
        B = b_w.sum(axis=0)
        order = np.argsort(d["scores"], kind="mergesort")
        uniq, inv = np.unique(d["scores"][order], return_inverse=True)
        bw = np.zeros((len(uniq), w.shape[0]))
        np.add.at(bw, inv, b_w[order])
        aw = np.zeros((len(uniq), w.shape[0]))
        np.add.at(aw, inv, a_w[order])
        below = np.concatenate([np.zeros((1, w.shape[0])),
                                np.cumsum(bw, axis=0)])[:-1]
        conc = (aw * (below + 0.5 * bw)).sum(axis=0)
        return np.where((A > 0) & (B > 0), conc / np.maximum(A * B, 1e-300),
                        np.nan)

    auc_a = auc_row(da)
    auc_b = auc_row(db)
    return ((lb - la) + (sb - sa) + (rb - ra) + (fb - fa)
            + 10 * (np.nan_to_num(auc_b) - np.nan_to_num(auc_a)))


def bootstrap_reference(da: dict, db: dict, draws: int, seed: int) -> np.ndarray:
    """Explicit per-resample loop over the SAME draws -- the independent
    reference C3 cross-checks the vectorised path against."""
    n = len(da["scores"])
    W = draw_weights(n, draws, seed)
    out = np.empty(draws)
    for r in range(draws):
        out[r] = total_from_weights(db, W[r]) - total_from_weights(da, W[r])
    return out


# --------------------------------------------------------------------------
# Cross-validation suite (C1-C4); any mismatch aborts the run
# --------------------------------------------------------------------------
def brute_auc(a_w: np.ndarray, b_w: np.ndarray, scores: np.ndarray) -> float:
    """O(n^2) pairwise AUC, tie-aware: the independent AUC reference."""
    A, B = a_w.sum(), b_w.sum()
    if A <= 0 or B <= 0:
        return float("nan")
    conc = float((a_w[:, None] * (b_w[None, :] * (scores[:, None] > scores[None, :])
                                  + 0.5 * b_w[None, :] * (scores[:, None] == scores[None, :]))).sum())
    return conc / (A * B)


def validate(a: pd.DataFrame, b: pd.DataFrame, da: dict, db: dict) -> None:
    total_a, _ = subtotal(a)
    total_b, _ = subtotal(b)
    fast_a = total_from_weights(da, np.ones(len(a)))
    fast_b = total_from_weights(db, np.ones(len(b)))
    assert abs(fast_a - total_a) < 1e-9 and abs(fast_b - total_b) < 1e-9, (
        f"C1 full-frame mismatch: base {fast_a} vs {total_a}, "
        f"cand {fast_b} vs {total_b}")

    rng = np.random.RandomState(123)
    for i in range(6):
        src = a if i % 2 == 0 else b
        take = rng.randint(0, len(src), size=len(src))
        w = np.bincount(take, minlength=len(src)).astype(float)  # multiplicities
        expanded = src.iloc[take].reset_index(drop=True)
        exact, _ = subtotal(expanded)
        fast = total_from_weights(prep(src), w)
        assert abs(fast - exact) < 1e-9, (
            f"C2 resample {i}: vectorised weights {fast} != score() {exact}")

    ref = bootstrap_reference(da, db, draws=200, seed=777)
    vec = bootstrap_from_weights(draw_weights(len(a), 200, 777), da, db)
    assert np.max(np.abs(ref - vec)) < 1e-9, (
        f"C3 vectorised vs reference loop diverge: max {np.max(np.abs(ref - vec))}")

    rs = np.random.RandomState(99)
    for i in range(30):
        m = 40 + int(rs.randint(0, 60))
        scores = np.round(rs.rand(m) * 4, 1)     # coarse -> forced ties
        a_w = (rs.rand(m) < 0.5) * rs.rand(m)
        b_w = (rs.rand(m) < 0.5) * rs.rand(m)
        fast = _auc(a_w, b_w, scores)
        brute = brute_auc(a_w, b_w, scores)
        assert abs(fast - brute) < 1e-9, f"C4 AUC case {i} diverges"
    print("cross-validation PASSED: "
          "C1 full-frame == score(); "
          "C2 6 resamples == score() on expanded frames; "
          "C3 vectorised == reference loop over 200 identical draws; "
          "C4 weighted AUC == brute-force on 30 tie-forcing cases (1e-9)")


def main():
    a, b = load_aligned()
    da, db = prep(a), prep(b)
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

    validate(a, b, da, db)

    print(f"\nPaired bootstrap of subtotal delta (cand - base), "
          f"N={N_BOOT}, seed={SEED}, fully vectorised, no loops:")
    deltas = bootstrap_vectorised(da, db)
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

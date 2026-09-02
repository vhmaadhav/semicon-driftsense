#!/usr/bin/env python3
"""Staged-200 gate statistics for the PR 34 headline rescore at 0.18.

The full-frame analysis (rescore018_driver.py, RESCORE_SHIPPED_T018.md)
runs on all 2,250 pairs. The rubric actually grades on a stratified
200-pair stage: A=70, B=70, C=40 pairs, rejection F1 over the staged
grayscale pairs. This script reruns the SAME paired comparison
(feat_base_nb.csv shipped weights vs feat_setcfull.csv candidate,
threshold 0.18) at grading sample size and measures how often each
checkpoint clears the gates:

  * paired 85-pt subtotal delta (candidate - shipped) vs the +0.35 gate
  * rejection F1 @0.18 over the 180 staged grayscale pairs vs the 0.90 gate

Sampling spec (exact): numpy RandomState(7), one rng for all resamples;
per resample draw WITH replacement 70 row-indices from set A, 70 from B,
40 from C (strata taken from the aligned base frame; the 'set' column is
identical across frames -- load_aligned asserts it), concatenate the 180
indices, and index BOTH frames with the SAME indices (the pairing).
Subtotals come from the committed driver's cross-validated fast path
(prep + total_from_weights); resample #0's staged frames are additionally
pushed through the real evaluator (drv.subtotal -> scripts/eval_ext.py
score()) and must match to 1e-9 or the run aborts.

Resources: N=5000, ~1 min runtime, three (5000,) float arrays plus
180-row frames. Imports the committed driver; modifies no tracked file.
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER = os.path.join(HERE, ".agents", "rescore018_driver.py")
N_STAGED = (("A", 70), ("B", 70), ("C", 40))   # grading stage composition
N_BOOT = 5000
SEED = 7
THRESH = 0.18        # driftsense.config.SHIPPED_THRESHOLD
F1_GATE = 0.90       # staged rejection-F1 gate
DELTA_GATE = 0.35    # paired subtotal-delta promotion gate

t0 = time.time()

spec = importlib.util.spec_from_file_location("drv", DRIVER)
drv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drv)

a, b = drv.load_aligned()
sets = a["set"].values
assert set(np.unique(sets)) == {"A", "B", "C"}, "unexpected set labels"
idx_by_set = {s: np.where(sets == s)[0] for s, _ in N_STAGED}
print("full-frame strata:",
      ", ".join(f"{s}={len(v)}" for s, v in idx_by_set.items()))

rng = np.random.RandomState(SEED)
deltas = np.empty(N_BOOT)
f1_base = np.empty(N_BOOT)
f1_cand = np.empty(N_BOOT)
spot = None

for r in range(N_BOOT):
    idx = np.concatenate(
        [rng.choice(idx_by_set[s], size=n, replace=True) for s, n in N_STAGED])
    a_s, b_s = a.iloc[idx], b.iloc[idx]
    pa, pb = drv.prep(a_s, THRESH), drv.prep(b_s, THRESH)
    w = np.ones(len(idx))
    ta = drv.total_from_weights(pa, w)
    tb = drv.total_from_weights(pb, w)
    deltas[r] = tb - ta
    for d, store in ((pa, f1_base), (pb, f1_cand)):
        tp, fp, fn = d["tp"].sum(), d["fp"].sum(), d["fn"].sum()
        den = 2.0 * tp + fp + fn
        store[r] = 2.0 * tp / den if den > 0 else 0.0
    if r == 0:
        assert (a_s["set"].value_counts().to_dict()
                == {"A": 70, "B": 70, "C": 40}), "stage composition wrong"
        sub_a, _ = drv.subtotal(a_s, THRESH)      # real evaluator score()
        sub_b, _ = drv.subtotal(b_s, THRESH)
        gap_a, gap_b = abs(sub_a - ta), abs(sub_b - tb)
        spot = (ta, sub_a, gap_a, tb, sub_b, gap_b)
        if max(gap_a, gap_b) >= 1e-9:
            sys.exit(f"SPOT-CHECK FAILED at resample #0: "
                     f"base |fast-score()|={gap_a:.3e}, "
                     f"cand |fast-score()|={gap_b:.3e} (tol 1e-9)")

# anomaly guards (task spec): stop rather than publish impossible numbers
assert f1_base.std() <= 0.1 and f1_cand.std() <= 0.1, "F1 std > 0.1: anomaly"
assert ((0 <= f1_base) & (f1_base <= 1)).all() and \
       ((0 <= f1_cand) & (f1_cand <= 1)).all(), "F1 outside [0,1]: anomaly"
for name, rate in (("base", float((f1_base >= F1_GATE).mean())),
                   ("cand", float((f1_cand >= F1_GATE).mean()))):
    assert 0.0 <= rate <= 1.0, f"{name} gate rate outside [0,1]: anomaly"

lo, hi = np.percentile(deltas, [2.5, 97.5])
print(f"\n=== staged-200 paired subtotal delta (cand - base), "
      f"N={N_BOOT}, seed={SEED} ===")
print(f"mean    {deltas.mean():+.4f}")
print(f"median  {np.median(deltas):+.4f}")
print(f"std     {deltas.std(ddof=1):.4f}")
print(f"95% CI  [{lo:+.4f}, {hi:+.4f}]")
print(f"P(delta >= +{DELTA_GATE:.2f}) = {(deltas >= DELTA_GATE).mean():.4f}")
print(f"P(delta > 0)             = {(deltas > 0).mean():.4f}")

print(f"\n=== staged rejection F1 @ {THRESH} (over the 180 grayscale pairs) ===")
print(f"base: mean {f1_base.mean():.4f}  median {np.median(f1_base):.4f}  "
      f"std {f1_base.std(ddof=1):.4f}  P(F1 >= {F1_GATE:.2f}) "
      f"{(f1_base >= F1_GATE).mean():.4f}")
print(f"cand: mean {f1_cand.mean():.4f}  median {np.median(f1_cand):.4f}  "
      f"std {f1_cand.std(ddof=1):.4f}  P(F1 >= {F1_GATE:.2f}) "
      f"{(f1_cand >= F1_GATE).mean():.4f}")
c_only = ((f1_cand >= F1_GATE) & (f1_base < F1_GATE)).mean()
b_only = ((f1_base >= F1_GATE) & (f1_cand < F1_GATE)).mean()
print(f"decision flips: cand clears & base fails {c_only:.4f} | "
      f"base clears & cand fails {b_only:.4f}")

print("\n=== spot-check (resample #0, staged frames vs real score()) ===")
print(f"base fast {spot[0]:.6f} vs score() {spot[1]:.6f}  |gap| {spot[2]:.2e}")
print(f"cand fast {spot[3]:.6f} vs score() {spot[4]:.6f}  |gap| {spot[5]:.2e}")
print(f"tolerance 1e-9 -> {'PASS' if max(spot[2], spot[5]) < 1e-9 else 'FAIL'}")

print(f"\nwall time {time.time() - t0:.1f}s")

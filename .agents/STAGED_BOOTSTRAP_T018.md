# Staged-200 gate statistics at SHIPPED_THRESHOLD = 0.18 — PR #34 headline CSVs

Companion to `RESCORE_SHIPPED_T018.md`, which rescored the PR #34 headline
CSVs — `.agents/feat_base_nb.csv` (shipped weights) vs
`.agents/feat_setcfull.csv` (candidate) — on all 2,250 pairs at 0.18. The
rubric actually grades on a stratified 200-pair stage (A=70, B=70, C=40;
F1 over the staged grayscale pairs), so this document reruns the SAME
paired comparison at grading sample size: how often does each checkpoint
clear the gates when only one stage is drawn? The CSVs contain only
grayscale sets (A=875, B=875, C=500), so the staged frame here is the
stage's 180 grayscale pairs; a real stage's 20 non-grayscale pairs add
nothing (every `score()` component is gray-masked).

## Headline — stratified paired bootstrap, N=5000, seed 7

| statistic | value |
|---|---:|
| staged paired subtotal delta (cand − shipped), mean / median | +0.3223 / +0.2974 |
| 95% CI (2.5 / 97.5 pct) | [−0.5618, +1.3174] |
| P(delta ≥ +0.35 promotion gate) | 0.456 |
| P(delta > 0) | 0.741 |
| base F1 @0.18, mean / median / sd | 0.9080 / 0.9114 / 0.0332 |
| candidate F1 @0.18, mean / median / sd | 0.9199 / 0.9231 / 0.0308 |
| P(F1 ≥ 0.90 bonus gate), base → candidate | 0.635 → 0.773 |

Decision-relevant number: the candidate's staged gate rate vs the base's —
**77.3% vs 63.5%**, with asymmetric flips (cand clears & base fails 15.6%
of stages; the reverse 1.7%). Measured F1 sd 0.031–0.033 matches the ~0.03
of the prior staged analysis (`REJECTOR_NONLINEAR.md`: 0.0324–0.0341).

## Full-frame vs staged

| statistic | full 2,250 (`RESCORE_SHIPPED_T018.md`) | staged 200 (this doc) |
|---|---:|---:|
| point / mean delta | +0.3289 | +0.3223 (median +0.2974) |
| 95% CI | [+0.0692, +0.5999] | [−0.5618, +1.3174] |
| P(delta ≥ +0.35) | 0.429 | 0.456 |
| F1 @0.18, base → cand | 0.9078 → 0.9198 (point) | 0.9080 → 0.9199 (mean) |
| P(F1 ≥ 0.90) | n/a (both point F1s > 0.90) | 0.635 → 0.773 |

The staged mean matches the full-frame point within Monte-Carlo error (MC
se ±0.007 at N=5000; the 70/70/40 stage is exactly proportional to the
875/875/500 frame), so the whole difference is spread: 180 pairs instead
of 2,250 inflate per-resample subtotal noise ~3.5× (√(2250/180)), and
single-pair F1 flips (~0.01–0.02 each) quantize the staged F1.

## Method

- Import the committed driver `.agents/rescore018_driver.py` as a module
  (no side effects at import); use `load_aligned()` (pair_id alignment;
  `set` asserted identical across frames), `prep()` (masks per staged
  frame — correct as-is) and the cross-validated `total_from_weights()`.
- Sampling spec, exact: one `numpy.random.RandomState(7)` for all
  resamples; N=5000; per resample draw WITH replacement 70 row-indices
  from set A, 70 from B, 40 from C (strata of the aligned frame);
  concatenate the 180 indices; index BOTH frames with the SAME indices
  (the pairing). Per resample: delta from the fast path on the staged
  frames; rejection F1 @0.18 over the 180 grayscale pairs from the same
  prep dicts: f1 = 2tp/(2tp+fp+fn), tp/fp/fn = d['tp'/'fp'/'fn'].sum(),
  recorded for base and candidate separately.
- Spot-check: resample #0's staged frames through the real evaluator
  (`drv.subtotal` → `scripts/eval_ext.py score()`) vs the fast path —
  base 77.284598 vs 77.284598, candidate 78.084189 vs 78.084189,
  |gap| 1.4e-14 ≤ 1e-9 → **PASS** (the driver aborts on mismatch).
  Anomaly guards (F1 sd ≤ 0.1, F1 ∈ [0,1], rates ∈ [0,1]) passed.
  Runtime 8 s; memory: three (5000,) float arrays plus 180-row frames.

## Reproduce

```python
import importlib.util
import numpy as np
spec = importlib.util.spec_from_file_location(
    "drv", "/tmp/pr34-wt/.agents/rescore018_driver.py")
drv = importlib.util.module_from_spec(spec); spec.loader.exec_module(drv)
a, b = drv.load_aligned()
by = {s: np.where((a["set"] == s).values)[0] for s in "ABC"}
rng = np.random.RandomState(7)
D, F1b, F1c = [], [], []
for r in range(5000):          # stratified 70/70/40, replacement, seed 7
    idx = np.concatenate([rng.choice(by[s], size=n, replace=True)
                          for s, n in (("A", 70), ("B", 70), ("C", 40))])
    pa, pb = drv.prep(a.iloc[idx]), drv.prep(b.iloc[idx])  # same idx = paired
    w = np.ones(len(idx))
    D.append(drv.total_from_weights(pb, w) - drv.total_from_weights(pa, w))
    for d, out in ((pa, F1b), (pb, F1c)):   # F1@0.18 over the 180 pairs
        tp, fp, fn = d["tp"].sum(), d["fp"].sum(), d["fn"].sum()
        out.append(2 * tp / (2 * tp + fp + fn))
D, F1b, F1c = map(np.array, (D, F1b, F1c))
print(D.mean(), np.median(D), np.percentile(D, [2.5, 97.5]),
      (D >= .35).mean(), (D > 0).mean(),
      (F1b >= .9).mean(), (F1c >= .9).mean(), F1b.std(), F1c.std())
```

`.agents/staged_bootstrap_driver.py` reproduces every number above and
adds the spot-check and anomaly guards (~8 s).

## Interpretation — what this licenses and what it does not

- DOES quantify gate-decision risk at grading sample size for BOTH
  checkpoints: on a drawn stage the candidate clears the F1 ≥ 0.90 bonus
  gate 77.3% vs 63.5% for shipped, and the paired delta reaches +0.35 in
  45.6% of stages (P(delta > 0) = 0.741).
- Does NOT change the full-frame +0.3289 point estimate, its CI
  [+0.0692, +0.5999], or the NOT-CLEARED verdict on the +0.35 gate; the
  staged CI is wider by construction (1/√n at n=180), not a different
  effect. The full-frame estimate stays the decision-grade one.
- The staged CI includes negative deltas (~26% of stages score the
  candidate below shipped) — stage noise, not regression evidence: every
  full-frame component delta is non-negative beyond noise (worst: scale
  −0.0052). P(delta ≥ +0.35) is similar staged (0.456) vs full-frame
  (0.429) despite the lower staged median — the 3.5× wider spread
  compensates: a coin-flip gate either way, not "looks better staged".
- Comparability: the staged gate rates in `.agents/REJECTOR_NONLINEAR.md`
  (53.1/59.4/64.7%) come from a different (CV) protocol — out-of-fold
  CV-swept decisions at the then-in-use 0.202, N=20000, unpaired — and
  are NOT directly comparable to the rates here (committed score columns,
  shipped 0.18, no fitting, paired). That doc's threshold note already
  points the 0.18 staged-200 rates here; its 64.7% vs our 63.5% reflects
  F1 @0.202 ≈ @0.18 on the same score column, not one protocol.

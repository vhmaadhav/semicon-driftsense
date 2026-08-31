# Rejector findings: post-hoc present/absent rejection is measured out

**Date:** 2026-08-30 · **Issue:** #6 (closed, documented negative) · **Branch:** `issue-6-rejector-features` (PR #17) · **Machine:** MacBook Air, CPU only

## Question

Can peak-quality features (rank, band, winner-margin) lift the present/absent
rejection F1 from 0.878 to 0.90 — earning the +4 rubric bonus — without
retraining?

## Setup

- Data: `data/ext_p2` restored byte-compatible from the `a06d9df…` run
  (A 875 / B 875 / C 500 = 2,250 pairs).
- Decode: shipped `verification="zncc"` winner, bit-identical with and without
  feature instrumentation (verified on a 60-pair control: max coordinate
  delta 0.0).
- New feature: `matching.winner_margin` — the SELECTED winner's
  `min(score, zncc)` margin over the best runner-up. First implementation
  measured the top-two-strength gap instead; Codex review caught that the
  zncc winner need not be the strength leader, the semantics were corrected,
  and **all numbers below are from the regenerated CSV** with the corrected
  feature.
- Feature CSV: `.agents/ext_features_full.csv` (2,250 pairs, 9 features,
  decode unchanged).

## Result 1 — 4-fold CV on the total rubric

Threshold fit on the training fold only; scored out-of-sample.

| statistic | held-out total | F1 | AUC |
|---|---:|---:|---:|
| shipped `min(score,zncc)` | 75.45 | 0.8716 | 0.9878 |
| logistic: 6 shipped features | 75.59 | 0.8719 | 0.9917 |
| logistic: shipped + rank, band, margin | 75.59 | 0.8738 | 0.9911 |
| logistic: score,zncc,rank,band,margin | 75.41 | 0.8606 | 0.9871 |
| logistic: score,zncc,margin | 75.37 | 0.8626 | 0.9868 |

Best extended fit: **+0.13 points vs the +0.35 promotion gate.** Same noise
band as the earlier 6-feature refit (+0.11). The AUC↑/F1↓ cancellation the
issue predicted reproduces: AUC rises (0.9878 → 0.9911) while F1 barely moves.

## Result 2 — the oracle upper bound (the decisive number)

Fitting the logistic **in-sample** on all 9 features (cheating freely) and
sweeping every threshold:

| statistic | max achievable F1(reject) |
|---|---:|
| shipped 6 features, in-sample oracle | 0.8827 |
| all 9 features, in-sample oracle | **0.8850** |

The bonus bar is **0.90**. Even a perfect threshold on a perfect post-hoc fit
falls 1.5 F1 points short. No feature or threshold rule can cross it on this
decode — the remaining errors are **confident wrong lock-ons** (Set B pairs
that sit above the threshold at every operating point), not confidence errors.

## Separation sanity (why the features looked promising)

Per-class means on the 60-pair control, before the full run:

| feature | absent (C) | present (A/B) |
|---|---:|---:|
| rank | 0.035 | 0.281 |
| band | 0.200 | 0.834 |
| margin | 0.035 | 0.518 |

The features separate the classes well — which is exactly why AUC is high —
but separation is not the binding constraint. The F1 ceiling is set by the
confident-wrong-lock-on mass that no monotone threshold can peel off without
losing an equivalent mass of real pairs.

## Conclusion

As issue #6 itself anticipated: the post-hoc rejection question is **closed
for good**. The 0.90 bar is reachable only through the training path —
issue #11 (Set C shard expansion + found-head retrain on the GPU/TPU-VM
host), whose "training, not post-hoc" argument this experiment now measures
directly.

## Reproduce

```
python scripts/rejector_cv.py .agents/ext_features_full.csv          # CV table
python scripts/eval_ext.py data/ext_p2/*_0000 --features --out .agents/ext_features_full.csv
```

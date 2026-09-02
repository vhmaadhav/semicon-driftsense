# Nonlinear rejector: beats the linear family, but the headroom is nearly gone

**Date:** 2026-09-01 · **Refs:** issue #6 / PR #17 (`REJECTOR_FINDINGS.md`), issue #11, issue #30 Lever 2

## Why re-open a closed question

`REJECTOR_FINDINGS.md` bounds post-hoc rejection at an in-sample oracle F1 of
**0.8850** under a 0.90 bar and calls it measured out. Two things make that
bound non-binding today:

1. It is **explicitly scoped** to the linear logistic family: *"It does not
   bound a nonlinear rejector (trees, kernels) or genuinely new features."*
2. Its numbers come from the **0.456M model** (shipped F1 0.8716, total 75.45).
   The wide 1.02M model now ships at **F1 0.9082** — already above that
   oracle ceiling. The bound describes a feature distribution we no longer have.

## Implementation

Gradient-boosted trees (Newton/xgboost-style, depth 3, 200 rounds, lr 0.06) in
**pure numpy** — no sklearn in any venv, and a fitted ensemble exports to plain
arrays, so nothing new would ship. Protocol identical to `scripts/rejector_cv.py`:
fit on the training folds, sweep the threshold on those same folds, score the
held-out fold. Scored on the **total rubric**, not F1 alone, because
`register.py` zeroes 40 localisation + 20 pose points on any declined pair.

Script: `scratchpad/nl_rejector.py`, `scratchpad/gate_oof.py`.

## Validation against the recorded numbers

Run on the same `.agents/ext_features_full.csv` the original study used:

| statistic | held-out total | F1 |
|---|---:|---:|
| shipped `min(score,zncc)` | **75.45** | **0.8718** |
| linear logistic | 75.47 | 0.8692 |
| GBT (nonlinear) | 75.58 | 0.8839 |

The shipped row reproduces `REJECTOR_FINDINGS.md` (75.45 / 0.8716) exactly, so
the harness is sound. On the old model the nonlinear family beats both the
shipped statistic (+0.012 F1) and the linear fit (+0.015) out-of-sample —
the linear bound was real, and the nonlinear family does clear it.

## The result that matters — on the CURRENT wide model

`.agents/feat_base_nb.csv` (shipped weights, `--no-band`, `--features`; the run
reproduces 77.15 / F1 0.9082 exactly, confirming instrumentation is decode-neutral):

| statistic | held-out total | F1 | AUC |
|---|---:|---:|---:|
| shipped `min(score,zncc)` | **76.89** | 0.8998 | 0.9878 |
| linear logistic | 76.75 | 0.8965 | 0.9928 |
| GBT (nonlinear) | 76.80 | **0.9042** | 0.9900 |

The nonlinear edge survives but **shrinks by two-thirds**: +0.0044 F1 here
against +0.012 on the old model, and it costs **−0.09 total points** because
the extra rejections forfeit localisation credit. The wide model has already
absorbed most of what the rejector was recovering.

## Gate probability, fully out-of-fold

Stratified 200-pair emulation (A=70/B=70/C=40, F1 over the 180 grayscale pairs,
N=20000), using out-of-fold binary decisions so nothing is fitted in-sample:

| statistic | mean F1 | sd | P(F1 >= 0.90) | E[bonus] of 4 |
|---|---:|---:|---:|---:|
| shipped `min(score,zncc)` | 0.9008 | 0.0341 | 53.1% | 2.13 |
| GBT | 0.9046 | 0.0324 | 59.4% | 2.37 |
| paired delta | +0.0038 | | P(GBT better) **54.5%** | |

For reference, the shipped decode's rejection column (`score`, fixed
threshold) sits at mean F1 0.9083 / **P(gate) 64.7%** in this sweep — better
than either CV-swept variant above, because sweeping the threshold for *total
points* sacrifices F1. Threshold note (2026-09-02): this sweep ran at the
then-in-use 0.202; the shipped operating point is `SHIPPED_THRESHOLD = 0.18`
(`driftsense/config.py`, rescore evidence in
`.agents/RESCORE_SHIPPED_T018.md`), and the 0.18 staged-200 gate rates are in
`.agents/STAGED_BOOTSTRAP_T018.md`.

## Conclusion

Post-hoc rejection is now measured out for the **nonlinear** family too, on this
model. The GBT is a genuine but marginal improvement — P(better) 54.5% is a
coin flip, and the direct-points cost cancels most of the bonus gain. **Do not
ship it**; the complexity is not repaid.

Two findings that outlive this experiment:

1. The linear-family bound in `REJECTOR_FINDINGS.md` **was** beatable — the
   nonlinear family clears it on both models. The doc's caution about its own
   scope was correct and should not be read as "rejection is closed forever".
2. **The +4 bonus is not banked.** P(clearing 0.90) is 53–65% depending on
   statistic — a coin flip, not the "earned, margin 0.008" the 2250-pair point
   estimate suggests. That is ~1.4 points of expected loss carried unscored.
   Raising F1 is worth far more than its 15-point block implies, because of the
   4-point cliff. The route remains training (issue #11), not post-hoc.

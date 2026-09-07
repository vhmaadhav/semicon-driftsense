# Early-exit gate fire rate (Stage 4)

**Date:** 2026-09-07 · **Data:** fresh 500-pair `data/holdout_p2` (405 present,
95 absent), generated with the `.agents/SUBPIXEL_DRIFT.md` recipe at its
previously-unused seed — `scripts/gen_data.py --split holdout_p2 --num-samples
500 --seed 20260902 --noise randomized --phase2 --absent-frac 0.2
--crops-per-canvas 1`. Never trained on. · **Decode:** shipped
(`band=SHIPPED_BAND`, `subpixel_rows=SHIPPED_SUBPIXEL_ROWS`, `hypotheses=3`),
4 threads, idle machine. · **Tool:** `scripts/early_exit_rate.py` · **Rows:**
`.agents/early_exit_rate_holdout_p2.csv` (500),
`.agents/early_exit_ab_holdout_p2.csv` (150, paired A/B).

`driftsense.config.EARLY_EXIT_GATES` was validated for *correctness* in
`.agents/PR51_CAMPAIGN.md` — bit-identical answers against a full no-early-exit
decode. Its **fire rate** had never been recorded, so the 1.18× headline in
`FAILURE_ANALYSIS.md` could not be attributed and there was no evidence either
way on whether loosening the gates would buy anything. This is that number.

## Result 1 — the gate fires on one pair in five

| | pairs | share |
|---|---:|---:|
| reachable (≥2 hypotheses returned) | 489 | 97.8% |
| **fired** | **103** | **20.6% of all, 21.1% of reachable** |
| — gate 0 `(0.85, 0.75, 0.25, None)` | 74 | 14.8% |
| — gate 1 `(0.72, 0.72, 0.35, 0.04)` | 29 | 5.8% |

Network forward passes — the quantity the gate exists to reduce, at ~86% of a
pair and one per hypothesis — **1,184 paid of the 1,345 a no-early-exit decode
would pay: 12.0% skipped, 1.136×.**

Replicated on the independent 150-pair A/B arm: 20.7% fired, 13.7% of passes
skipped. The rate is stable, not a small-sample artefact.

`pose_candidates` returns fewer than the requested three hypotheses often
enough to notice but not often enough to matter: 2.2% of pairs got one
candidate and could never fire. This is a rounding effect on the fire rate, not
a ceiling on it.

## Result 2 — paired A/B: 1.152×, and still bit-identical

150 pairs, each decoded twice, arm order alternating by pair index so neither
arm systematically runs second with the warm cache.

| | median | mean | p90 |
|---|---:|---:|---:|
| gates on | 3.565 s | 3.162 s | 4.492 s |
| gates off | 3.947 s | 3.644 s | 4.545 s |

**Overall 1.152×.** Paired saving per pair: median **+0.087 s**, mean +0.482 s.
The median is near zero because 79% of pairs skip nothing; on the 31 pairs that
*did* fire the median saving is **+2.264 s**, which is the "close to 3× on the
pairs that qualify" claim in `config.py` holding up.

**Answers identical to the gates-off arm on 150/150 pairs** (x, y, theta, scale,
confidence, exact equality). That independently reconfirms PR #51's correctness
result on a shard it was never checked against.

Both framings in the tree are therefore true and should not be conflated:

* per qualifying pair, the gate is worth ~3× — `config.py` is right;
* fleet-wide it is worth **1.15×** — and that is the number the efficiency
  budget gets. `FAILURE_ANALYSIS.md`'s 1.18× is consistent with this, at the
  optimistic end.

## Result 3 — the "near misses" are not near, and the one lever is the wrong one

386 pairs reached a gate and did not fire. Which term blocked them:

| gate | score | zncc | ratio | gap | blocked by exactly one term |
|---|---:|---:|---:|---:|---:|
| 0 | 261 | **354** | 199 | 0 | 130 |
| 1 | 208 | **346** | 175 | 142 | 156 |

The single-term-blocked pools look like the retune opportunity. They are not.
Of the 130 pairs gate 0 blocks on exactly one term, **112 are blocked by
`zncc`**; of gate 1's 156, **134 are**. And they are not marginal — median
`zncc` among them is **0.60 against a 0.75 / 0.72 floor**, a shortfall of
0.13–0.15, not a hair.

So converting them means dropping the native-ZNCC floor to roughly 0.59. That
is the worst available knob, for a reason the `locate_phase2` docstring already
states: a wrong scale basin correlates near zero at full resolution while the
right one sits around 0.9, and ZNCC is what `choose()` uses to pick the winner.
A floor at 0.6 sits squarely in the ambiguous middle — the region where the
early exit would start committing to a first hypothesis that the discarded ones
could legitimately have beaten. The gate's whole correctness argument is that
this cannot happen.

`score` (15 and 12 pairs) and `ratio` (3 and 5) are the only genuinely tight
terms, and there are too few of them to be worth the revalidation.

**Conclusion: there is no cheap retune of `EARLY_EXIT_GATES` available.** The
volume is real, it is concentrated on the single term that guards correctness,
and the shortfall on that term is large. Stage 4 should not spend effort here.

## Result 4 — absent pairs never fire, and they are the expensive ones

Not something the fire rate was looking for, and the more useful finding:

| | pairs | network passes | share of all network work | median |
|---|---:|---:|---:|---:|
| present | 405 | 915 | 77.3% | 3.034 s |
| **absent** | **95** | **269** | **22.7%** | **4.010 s** |

**0 of 95 absent pairs fired** — all 103 fires are on present pairs (25.4% of
them). This is not a defect: an absent pair has no strong match, so `score` and
`zncc` stay low by construction and the gate correctly declines to commit.

The consequence is that absent pairs pay the **full three-hypothesis price to
conclude there is nothing there**, and are ~1 s slower per pair than present
ones. At the blind set's expected absent fraction that is roughly a fifth of
all network work spent confirming an absence.

That reframes Stage 1. The trained presence head is currently justified by the
+4 bonus alone (~1.4 points of unscored expected value, `.agents/REJECTOR_NONLINEAR.md`).
It would *also* be the efficiency lever the gate retune isn't: a cheap early
absence call short-circuits the pairs that today cost the most and save the
least. Worth pricing before Stage 4 spends anything on quantisation.

## What this does and does not license

* **Do** treat 1.15× as the gate's real contribution to the runtime budget. With
  a measured median of ~3.5 s against 5 s, headroom stays ~1.4×, consistent with
  `.agents/` runtime reality and still gating Stage 5 (width 96→128) behind
  Stage 4.
* **Do not** retune `EARLY_EXIT_GATES` on the strength of the near-miss counts.
  Result 3 is a negative result and should be cited as one.
* **Do not** read Result 4 as a claim that a presence head would pay for itself
  on runtime — it says the absent pairs are where the unskipped work is, not
  how cheaply it can be skipped. Pricing that is a separate measurement.

Ceiling, for reference: if every reachable pair fired, the decode would pay 500
passes instead of 1,345 — 2.69×. Today's 1.136× is 8% of the way there, and
Result 3 says the remaining 92% is not reachable by moving thresholds.

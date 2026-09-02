# Set B: the loss is precision, not failure — and the lever is training-side

**Date:** 2026-09-01 · Measured on `.agents/cand_base_nb.csv` (shipped weights, `--no-band`, 2250 pairs)

## Where Set B's 3.32 points actually go

Set B credit 0.8247 vs Set A 0.9758. Decomposed by credit tier:

| tier | credit | set A | set B | contribution to the gap |
|---|---:|---:|---:|---:|
| ≤1px | 1.00 | 92.7% | **57.6%** | **−7.72 pts** |
| ≤2px | 0.80 | 5.9% | 23.2% | +3.04 |
| ≤3px | 0.60 | 0.2% | 8.0% | +1.03 |
| ≤5px | 0.40 | 0.0% | 3.8% | +0.33 |
| >5px | 0.00 | 1.1% | 7.4% | +0.00 |
| | | | | **−3.32 pts** |

**The 65 gross failures (>5px) cap out at +1.63 points even if every one were
fixed.** The larger prize is the 306 pairs that land correctly but imprecisely —
203 in the 1–2px band alone. Set B's median error is 0.826px against Set A's
0.319px. This is a **precision** problem wearing a failure problem's clothes,
and issue #30 Lever 1's framing (severity coverage → gross failures) aims at the
smaller half.

## The single variable that explains it: raster drift jitter

Set B pairs bucketed by `drift_jitter_px` from their generation manifest:

| quartile | jitter px | median err | ≤1px | >5px |
|---|---|---:|---:|---:|
| Q1 | 0.34–0.77 | 0.433 | **87.7%** | 0.0% |
| Q2 | 0.77–1.21 | 0.733 | 65.6% | 1.8% |
| Q3 | 1.21–1.66 | 1.210 | 41.6% | 10.5% |
| Q4 | 1.66–2.10 | 1.556 | **35.6%** | 17.4% |

Set A shows the same monotone trend but far milder (99.5% → 84.5%) because its
jitter range only spans 0.15–0.80. Drift jitter drives **both** the precision
loss and the gross failures. Severity level is a proxy; jitter is the mechanism.

**Upside if Q2–Q4 reached Q1's tier mix: +3.36 points. Half the way: +1.68.**

## What is already exhausted (do not re-attempt)

- **Inference-side sub-pixel refinement is bounded out.** `PHASE2_STATE.md` §3e:
  with true pose supplied, the best rigid ZNCC sits at 1.077px median while the
  shipped pipeline is at 0.685px. We already beat correlation on 59.4% of pairs.
  Every interpolation/polish idea is bounded by a line worse than where we are.
- **`--jitter-power -1`** (up-weight high-drift pairs, the §3e prescription) is
  already in `scripts/wide_run.sh:40` — the wide model has it.
- **Rotation-aware coarse sweep**: independently re-confirmed today. Set B
  failure rate by |rotation| is 5.1 / 10.8 / 4.9 / 5.7 / 10.3% — no trend.
- **Nearest-to-centre tie-breaking**: 42/58 split on which is closer to centre.
  Would break as many as it fixes. New measured negative.
- **Pose bias correction**: none left. Removing the median signed bias moves
  scale error 0.427% → 0.426%.

## The untried lever

Training samples the pool **uniformly** (`train.py:265`,
`RandomSampler(replacement=False)`), and the B shards are balanced 25/25/25/25
across severity levels. So high-jitter pairs get extra *loss weight*
(`--jitter-power -1`) but never extra *exposure*. A jitter- or severity-weighted
sampler is a distinct lever that has not been measured, and it points at exactly
the Q3/Q4 mass above. The jury's note that the blind 200-pair set skews harder
than the sample argues for it independently.

Caveat worth pricing in: it may be partly redundant with `--jitter-power -1`,
which already tilts the gradient the same way. Expected value is real but
unproven.

## Blocker: the validation signal is blind

`scripts/wide_run.sh:38` passes `--val-limit 12` against a 200-scene set, so
every run scores 12 scenes per epoch:

| history | epochs | n | distinct acc@1px values |
|---|---:|---:|---:|
| `p7`, `p8`, `p9`, `setc` | 30, 30, 30, 9 | 12 | **1** (pinned at 0.7500) |
| `wide` | 34 | 12 | 4, frozen at 0.7500 from epoch 22 |

`acc@1px` is quantised to 1/12 = 8.3% steps and does not move. **Within-run
checkpoint selection and early stopping are uninformative** — which is why
`_last.pt` is the only checkpoint anyone uses, and why `driftsense_setc_last.pt`
sat unevaluated. Any Set B retrain would fly blind for hours and then need a
full 9-minute 2250-pair eval to learn anything.

**Fix this before spending GPU hours on Set B**: raise `--val-limit` toward the
200 scenes that already exist. Validation cost rises, but a run you cannot steer
is worse than a slower one you can.

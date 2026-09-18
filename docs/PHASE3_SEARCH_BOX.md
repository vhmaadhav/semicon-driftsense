# Phase 3: the pose search box, the threshold, and the label convention

Measured 2026-09-18, after the Phase 3 briefing (`Applied_Materials_Phase3_Briefing.pdf`)
fixed the contract. Four defects, one root cause: **`phase3.py` inherited Phase 2's calibrations, and
Phase 3's are different.** Net effect on the held-out 60-pair set: **49.42 -> 69.75 of 85.**

Everything below is on 60 generated pairs (53 present / 7 absent) at stage rotation
`U(-10, +10)` — the first Phase 3 set in this repo that has any rotation at all.

## Why no rotated set existed before

Neither generator can make one:

* the organizer's `generate_cad_dataset.py` never passes `search_rotation_deg`, so it
  keeps the `0.0` default — `src/cad_pipeline.py` supports rotation fully, the CLI just
  does not reach it;
* this repo's `generate_phase3_dataset.py` hard-codes `theta 0.0, scale 10.0` into
  `ground_truth.csv`.

So `PHASE3_COMPLETENESS.md`'s "rotation is not exercised … the single largest gap"
was exactly right, and it hid the three defects below. The set used here was generated
by driving `build_cad_geometry` / `render_cad_sample` directly and reproducing the
angle the pipeline draws (`default_rng(strip_rng_seed + 2).uniform(-r, +r)`).

**That reproduction is self-checking**: if the angle were wrong the rotation error would
be uniform over ±10°, and it measures a median of 0.104°. The localisation ground truth
is independent of it either way — `gt_x`/`gt_y` come from the pipeline's own rotated crop.

## The three defects

| # | what | evidence |
|---|---|---|
| 1 | `locate_phase2` searched `PHASE2_ROTATION_BOUNDS = (-5, 5)` and then **clipped `theta` into it**, so no pair past 5° could be answered — and Phase 3's cap is `MAX_SEARCH_ROTATION_DEG = 10.0` | at 8–10° the shipped decode declines or misplaces 8 of 8 |
| 2 | it also swept `PHASE2_SCALE_BOUNDS = (8, 12)` for a magnification Phase 3 pins near 10 (`SCALE_FACTOR = 10`), spending samples and inviting wrong-scale basins | narrowing to 9–11 is worth +1.0/85 on its own |
| 3 | `--threshold` defaulted to `SHIPPED_THRESHOLD = 0.55`, swept on Phase 2's 20%-absent distribution; Phase 3 discloses 8.3% absent, and a declined present pair forfeits localisation (40) **and** pose (20) | 0.55 declined 48 of 226 present pairs on the 250-pair fit set |
| 4 | `--label-convention` defaulted to Phase 2's `"center"`, subtracting 0.5 px from a coordinate the Phase 3 generator already defines in the edge convention (`cad_pipeline.py:227-229`: `centre = top-left + w/2`) | signed dy median -0.5145 -> -0.0145; pairs inside 1 px 9/53 -> 25/53 |

## Measured, 85 measurable points (loc 40 + scale 10 + rot 10 + rejection 15 + AUC 10)

| arm | loc/40 | scale/10 | rot/10 | rej/15 | cal/10 | **/85** | median s/pair |
|---|---|---|---|---|---|---|---|
| shipped — scale 8–12, rot ±5, T=0.55 | 17.21 | 6.11 | 5.25 | 11.74 | 9.11 | **49.42** | 1.75 |
| rotation only — rot ±10 | 20.83 | 7.51 | 7.25 | 13.03 | 8.37 | **56.98** | 1.72 |
| scale only — 9–11 | 18.42 | 6.57 | 5.19 | 11.74 | 9.61 | **51.52** | 1.66 |
| both bounds | 22.19 | 7.06 | 6.75 | 12.32 | 9.69 | **58.00** | 1.69 |
| scale pinned at 10.0, rot ±10 | 20.83 | 7.74 | 7.21 | 12.95 | 9.16 | **57.88** | **1.28** |
| both bounds + T=0.20 | 26.42 | 8.79 | 8.57 | 13.76 | 8.89 | 66.43 | 1.78 |
| **+ edge convention (shipped now)** | **29.74** | **8.79** | **8.57** | **13.76** | **8.89** | **69.75** | 1.64 |

**+20.33 / 85 at unchanged runtime.** Paired on localisation alone the bounds change is
**+4.98/40, 95% CI [+1.36, +8.75]**, 10 pairs rescued against 2 broken (20,000-sample
paired bootstrap).

### Null control

Passing the Phase 2 box explicitly (`--scale-bounds 8 12 --rotation-bounds -5 5`)
reproduces the pre-change output **byte-identically**, and `register.py` on the
organizers' 25-pair Phase 2 set is byte-identical before and after. The bounds are a new
parameter with the Phase 2 values as its default; Phase 2 cannot move.

### Where the error goes, by true stage angle

| \|θ\| | n | shipped median px | after median px |
|---|---|---|---|
| 0–2° | 9 | 1.78 | 1.53 |
| 2–5° | 23 | 1.63 | 1.48 |
| 5–8° | 13 | 2.58 | 1.38 |
| 8–10° | 8 | — (all declined/misplaced) | 3 of 8 recovered |

## The threshold, and the ambiguity that decides it

**The positive class of the rejection F1 is not resolved by the source material**, and this
constant lives or dies on it. `driftsense/rubric.py` has documented it since Phase 2:
under *reject*-as-positive an always-found system scores exactly **0.000**; under
*present*-as-positive the same system scores **0.875**.

An earlier revision of this file swept the threshold under present-positive alone and
landed on 0.20. That is a mistake of method: `rubric.py`'s own policy is that the operating
point must be "near-optimal under either reading", which is what makes the ambiguity
survivable. Swept under both:

| | 250-pair fit | | | 60-pair held out | | |
|---|---|---|---|---|---|---|
| T | reject | present | **worst** | reject | present | **worst** |
| 0.20 | 58.20 | 70.43 | 58.20 | 56.29 | 69.93 | 56.29 |
| 0.70 | 61.74 | 69.74 | 61.74 | 66.29 | 70.58 | 66.29 |
| **0.75** | 63.54 | 69.48 | **63.54** | 65.28 | 69.76 | **65.28** |
| 0.80 | 63.21 | 68.67 | 63.21 | 65.28 | 69.76 | 65.28 |

`PHASE3_THRESHOLD = 0.75` maximises the worst case over both readings on both sets:
**+5.34 / +8.99** under the reading that would hurt if we guessed wrong, against
−0.95 / −0.17 under the other. It also turns the rejector back into a working component —
6 of 7 absent pairs rejected on the held-out set against 1 of 7 at T=0.20.

**Corollary worth stating because the earlier revision asserted the opposite:** "T=0 is the
argmax at every assumed absent rate from 8% to 40%" is true *only* under present-positive.
Under reject-positive, T=0 scores 0.0000 on the 15-point block by construction.

## The earlier single-reading sweep, for reference

Fitted on **250 pairs** (226 present / 24 absent, seed 777001), validated on the
**60-pair set** (seed 20260918) that was never used to choose it. One decode per set at
`--threshold 0`, re-thresholded offline so every arm reads the same answers.

| T | 0.00 | 0.15 | 0.20 | 0.25 | 0.40 | 0.55 (Phase 2) | 0.70 |
|---|---|---|---|---|---|---|---|
| fit 250 /85 | 67.58 | 66.84 | **66.84** | 66.74 | 65.20 | 59.37 | 44.24 |
| held-out 60 /85 | 68.14 | — | — | 65.91 | — | 56.69 | — |

0.55 declines **48 of 226** present pairs on the fit set. `PHASE3_THRESHOLD = 0.20`.

**T = 0 is the literal argmax — and stays the argmax at every assumed absent rate from
8% to 40%** (present pairs held fixed, absent pairs reweighted), so the result is not an
artefact of our generated absent fraction. It is still not adopted: the briefing says
high-confidence false grabs "carry heavy penalties", which a plain F1 term does not
express, and the organizers' own README discloses a size signature on their absent
decoys, so their absents may be separable in a way ours are not — a constant `found`
column cannot exploit that at all. 0.20 is the best non-degenerate point at every
assumed rate and costs 0.74 against T=0.

### The threshold is not the real problem

| | present above T | absent above T |
|---|---|---|
| T = 0.20 | 0.97 | **0.88** |
| T = 0.40 | 0.92 | 0.67 |
| T = 0.55 | 0.79 | 0.33 |

The shipped `legacy_min` confidence **barely separates the two classes on CAD-reference
pairs**: AUC 0.836 here against 0.9877 on Phase 2 data. No threshold repairs a statistic
that does not separate, and 25 of the 100 points (rejection 15 + calibration 10) ride on
it. `PHASE3_MEASUREMENT.md` already measured the replacement — the confidence **margin**
(best peak minus best competing peak) separates good pose basins from bad at **AUC
0.948**. Wiring the margin into the `score` column is the largest remaining lever in the
Phase 3 rubric; this constant only stops the current statistic from throwing away
localisation and pose on top of the points it was already losing.

## Measured nulls — recorded so they are not re-derived

| tried | result |
|---|---|
| beam-PSF blur on the rendered reference (sigma 2/3/5) | **flat**: 70.39 / 69.73 / 70.28 / 70.50 of 85. ZNCC is normalised and the 10x INTER_AREA reduction already dominates. |
| widening the hypothesis set 3 -> 8 and ranking perfectly | **+3 pairs only**. 6 of 9 failures never generate a correct candidate; oracle is 47/53, not 53/53. Same structure as the Phase 2 Set B selector ceiling. |
| scale pinned at 10.0 | **-3.12 / 85** |
| reference-free global theta estimator | 4.42 deg median vs the decode's own 0.104 deg |

### What that leaves

Every constant is now fixed and both search-space ceilings are hit. Localisation sits at
29.74/40 against an oracle ceiling near 33-35 for anything that is not a better model. The
network is worth **+34.5** over the naive ZNCC baseline on Phase 2 and only **+8.4** on
Phase 3 (70.39 vs 61.95) — same weights, and the only change is that the reference became a
rendered design. Its confidence is now the weaker of the two signals (score AUC 0.834
against ZNCC's 0.865), the oracle-pose test caps localisation at 27.92/40, and 5 of 8
wrong-basin failures have the pose right to 0.5 deg and still pick the wrong repeat. Those
are three independent readings of one fact: **the remaining points are a training problem,
not a decode problem.**

## Still open

1. **8–10° is still where the points are.** 5 of 8 fail after the fix. `make_template`
   warps into a fixed 100×100 canvas with `BORDER_REPLICATE`, so at 10° roughly a tenth
   of the template is fabricated edge-replication rather than reference content. The
   fixed canvas is deliberate (comparing `TM_CCOEFF_NORMED` across different template
   sizes biases any scale search), so this is a real trade-off to measure, not an
   obvious bug to patch.
2. **Rotation grid density.** `coarse_rotations = 11` over ±10° is a 2° step against
   Phase 2's 1°. The local refine widens with the range (`span_r = (hi-lo)/10`), so the
   coverage is there, but the density has not been A/B'd at the wider range.
3. **Scale: window, not pin — measured.** `--scale-bounds 10 10` LOSES **3.12 / 85**:
   localisation 29.74 → 26.42 against scale credit 8.79 → 9.25. The scale freedom is
   load-bearing — a slight magnification mismatch absorbs the raster shear — so removing
   it costs more than the scale tier gains. It is 32% faster (1.10 vs 1.62 s/pair) and
   that is not worth 3.12 points. An earlier revision of this file called it a dead heat;
   that was measured under the wrong label convention.

## Reproducing

```bash
python phase3.py --input pairs.csv --output predictions.csv          # new defaults
python phase3.py --input pairs.csv --output predictions.csv \
    --scale-bounds 8 12 --rotation-bounds -5 5 --threshold 0.55      # old behaviour
```

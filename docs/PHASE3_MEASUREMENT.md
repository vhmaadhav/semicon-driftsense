# Phase 3: measurement record for the GDS narrative

This document records **what was measured** for Phase 3, on what data, with
which protocol — including two methodology errors that were caught and
corrected. It ships with the Phase 3 read path so the claims can be checked
rather than taken on trust.

## What was built

| piece | what it does |
|---|---|
| `phase3.py` | Phase 3 entry point — `--input pairs.csv --output predictions.csv`, same contract as `register.py` |
| `driftsense/gds.py` | `.gds` → 1000×1000 grayscale raster, brightness derived from layer index |
| `driftsense/pairs3.py` | Phase 3 `pairs.csv` reader with asserted column resolution (from PR #82) |

`register.py` is untouched. Phase 1 and Phase 2 still apply, and `register.py`
remains the graded command until Phase 3 opens.

### Correctness anchor

`driftsense/gds.render_reference()` produces a raster **bit-identical** to the
organizer-side CAD generator's own renderer on real generated Phase 3 files:

```
00000: identical=True  differing_px=0 (0.000%)
00001: identical=True  differing_px=0 (0.000%)
00010: identical=True  differing_px=0 (0.000%)
```

That is the strongest available check that the reference reaches the matcher in
the same convention the data was built with.

## The dataset

200 pairs generated with the mentor-side Phase 3 generator
(`generate_cad_dataset.py --num-samples 200 --seed 20260917`):

| | |
|---|---|
| pairs | 200 (187 present / 13 absent = 6.5%) |
| layers | 8 real GDSII layers per sample |
| architectures | 97 DRAM / 103 FinFET |
| wall time | 1 m 48 s, peak RSS ~370 MB |

Verified end-to-end through the new entry point (8 pairs, `--allow-fallback`):

```
median localisation error 0.93 px, 62% within 1 px, 88% within 2 px
```

## What was tested, and the answer

**Question:** for a GDS reference, is edge-based matching or CAD→SEM conversion
better than plain intensity ZNCC?

Four arms on the 200-pair set, all renders cached:

| arm | what |
|---|---|
| `A_intensity` | rasterize with the yield model, ZNCC — what the organizer's own CAD baseline does |
| `B_edges` | Sobel magnitude on both sides |
| `C_cad2sem` | rasterize → simulate an SEM capture → ZNCC |
| `D_cad2sem_edges` | `C` followed by the edge transform |

### Answer: no. Both lose, and edge matching loses clearly.

Held-out protocol (threshold calibrated on one half, scored on the other, 5
stratified splits, 85-point measurable subtotal of the published rubric):

| arm | held-out subtotal | std | vs A |
|---|---|---|---|
| `A_intensity` | **72.72** | 1.37 | — |
| `B_edges` | 66.83 | 2.67 | **−5.89** |
| `C_cad2sem` | 72.40 | 1.28 | −0.31 |
| `D_cad2sem_edges` | 61.33 | 3.14 | **−11.39** |

Localisation medians are near-identical (0.95–1.02 px); the difference is
entirely in the tails. Of 200 samples, 9 fail in **all four** arms — dataset
difficulty, not representation.

### And the deck's premise was tested with its own physics

The deck says: *"The design has no shading, and the SEM brightens edges it knows
nothing about."* The generator's `src/sem_imaging.py` has **no edge-brightening
term** — so a comparison on its output alone would be unfair to the edge arm.

So the search images were regenerated with edge brightening inserted at the
physically correct place (on the specimen, before blur and downsample), at
strengths this repo documents as realistic (0.20, 0.35):

| edge strength | arm | ≤1px | ≤2px | median err |
|---|---|---|---|---|
| 0.00 | `A_intensity` | 52% | 95% | 0.94 |
| 0.00 | `B_edges` | 48% | 90% | 1.00 |
| 0.20 | `A_intensity` | 53% | 95% | 0.94 |
| 0.20 | `B_edges` | 40% | 88% | 1.20 |
| 0.35 | `A_intensity` | 53% | 96% | 0.94 |
| 0.35 | `B_edges` | 48% | 93% | 1.00 |

**Intensity wins at every edge strength, including the ones the deck describes.**
Adding the physics the premise depends on does not change the ranking.

### Consequence

Edge matching should not be adopted. CAD→SEM conversion is a statistical tie
(−0.31) and is therefore not justified either. The measurable headroom found
lies elsewhere: the **confidence margin** (best peak minus best competing peak)
separates good pose basins (median +0.1666) from bad ones (+0.0148) at
**AUC 0.948** — a rejection signal, not a representation swap.

## Two methodology errors, recorded

Both would have produced a confident wrong answer. They are documented because
the same traps apply to anyone re-running this.

**1. A reimplemented rubric.** The first scoring pass used a hand-written
"65-point rubric-equivalent" that differed from the shipped
`driftsense/rubric.py` in three ways that each changed the ranking:

| | my version | shipped rubric |
|---|---|---|
| declined present pairs | credited their localisation | **masked to zero** (`register.py` zero-fills the row, and that is what is submitted) |
| rejection F1 | swept to the maximum (an oracle) | **at the fixed threshold** actually submitted |
| calibration AUC | present vs absent | **correct (present & ≤5px) vs everything else** |

Fixed by vendoring the real scorer (`vendor_rubric.py`, byte-identical to
`driftsense/rubric.py`) and calibrating the threshold on a held-out half
instead of sweeping it on the scored data.

**2. A padded-template centre offset.** A rotation test initially reported edge
matching winning dramatically (intensity 451 px mean error). The template was
rotated on a padded canvas and the *padded* template matched, so every reported
centre was offset by the padding amount — 25 px at 100 px template size. A
control run at **zero rotation** caught it, because it failed too (91 px). After
cropping back to the original bounding box the control reproduces the
no-sweep matcher exactly (1.12 / 1.55 / 1.40 px, identical scores).

**Run the null case as a control when adding a transform.** That is what
separated a "108× win" from a 25 px bug.

## Reproducing

```
cd mentor-phase3
.venv/bin/python generate_cad_dataset.py --num-samples 200 \
    --output-dir ./output --split phase3_200 --seed 20260917
.venv/bin/python phase3_heldout.py            # the held-out arm comparison
.venv/bin/python phase3_edge_forward.py       # the edge-brightened forward model
```

All GDS renders are cached to `cad_raster_cache.npz`, so re-runs take seconds.
Peak RSS stayed under 400 MB (the edge-hardening script blocks its Sobel over
rows to stay under 2.7 GB on the 10000×10000 canvas).

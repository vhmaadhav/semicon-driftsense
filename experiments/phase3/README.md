# Phase 3 experiment scripts

These produced the numbers in [`docs/PHASE3_MEASUREMENT.md`](../../docs/PHASE3_MEASUREMENT.md).
They are **not** part of the submission — they are the measurement record, kept
so the Phase 3 decision (do not adopt edge matching; CAD→SEM is a tie) can be
re-checked rather than taken on trust.

`vendor_rubric.py` is a byte-identical copy of `driftsense/rubric.py`, vendored
so these scripts score with the shipped rubric. If `driftsense/rubric.py`
changes, re-copy it — a divergence here would silently invalidate the numbers.

They expect a dataset at `output/phase3_200/` produced by the organizer-side
Phase 3 generator (`generate_cad_dataset.py --num-samples 200 --seed 20260917`),
and `gdstk` + `opencv` + `pandas` importable.

| script | what it measures |
|---|---|
| `phase3_ab.py` | four-arm localisation A/B (intensity / edges / CAD→SEM / CAD→SEM+edges) |
| `phase3_rubric_real.py` | the same arms scored with the vendored shipped rubric |
| `phase3_heldout.py` | **the headline result** — threshold calibrated on one half, scored on the other |
| `phase3_edge_forward.py` | regenerates search images with edge brightening in the forward model, to test the deck's premise with its own physics |

Renders are cached to `cad_raster_cache.npz`; first run is ~2 minutes, later
runs are seconds.

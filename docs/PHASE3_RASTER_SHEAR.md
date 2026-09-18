# Phase 3: the raster shear, and what can actually be measured about it

Measurement record for issue #101. Everything here was executed. Where a
hypothesis was wrong it is recorded as wrong, because two of the three
estimators built for this failed and knowing *why* is most of the value.

The decision record for the Phase 3 representation is
[`PHASE3_MEASUREMENT.md`](PHASE3_MEASUREMENT.md); the read-path verification
record is [`PHASE3_REVIEW_DIAGNOSTICS.md`](PHASE3_REVIEW_DIAGNOSTICS.md).

## 1. The defect, confirmed at the source

The organizer's Search capture shears every scan row
(`src/sem_imaging.py`, `apply_raster_drift`):

```python
shear     = shear_amplitude_px * (rows / max(h - 1, 1))
row_shift = shear + jitter
map_x     = np.arange(w)[None, :] + row_shift[:, None]
cv2.remap(img, map_x, map_y, ...)      # SAMPLES at x + s  =>  content moves -s
```

`gt_x`/`gt_y` are computed on the **pre-drift** geometry
(`src/cad_pipeline.py`, `render_cad_sample`), so imaged content sits
`-A*row/(h-1)` from where the answer says it is. The matcher is not wrong; the
label is defined somewhere the content no longer is.

Verified directly rather than inferred. Two sets generated from identical seeds
at `A = 0` and `A = 3` differ only in the shear, and a per-row phase
correlation between them recovers it:

```
fit of the row-to-row shift against y/999:  slope -3.0553  intercept +0.0532
                                            (model  -3.0000          0.0000)
  y[  0, 250) median shift -0.2847   (model -0.3754)
  y[250, 500) median shift -1.1253   (model -1.1261)
  y[500, 750) median shift -1.9290   (model -1.8769)
  y[750,1000) median shift -2.6892   (model -2.6276)
ground truth identical across amplitudes: True
```

### The decode does not absorb it

60 pairs, rotation U(-10, +10), Phase 3 calibrations from PR #102, wrong-basin
pairs (|error| > 20 px) excluded so the fit measures the bias and not the tail:

| A | dx ~ y/999 slope | intercept | median dy | median error | ≤1 px |
|---|---|---|---|---|---|
| 0 | −0.1796 | +0.1194 | +0.035 | 0.261 px | 95% |
| 1 | −1.3212 | +0.2732 | −0.004 | 0.625 px | 88% |
| 2 | −2.4089 | +0.2939 | +0.029 | 1.195 px | 36% |
| 3 | −3.6016 | +0.4662 | +0.012 | 1.775 px | 31% |
| 4 | −4.2365 | +0.1584 | +0.028 | 2.422 px | 19% |

Residual slope against A: **−1.0394·A − 0.271**. So **104% of the amplitude
survives the decode as an x-only bias**, `dy` stays flat, and the correction is
simply `x += A*y/(h-1)` with no gain term. This also corrects an expectation in
the issue: the cost is larger than the ≈18% absorption the original
measurement suggested.

## 2. The measurement rig

`scripts/gen_phase3_rotated.py` grew `--shear`, `--jitter` and `--paired`.
`--paired` draws one seed per sample index up front, so sample *i* sees the
same seed at every amplitude: identical mat layout, crop site, architecture and
rotation, and — because the Search acquisition RNG is derived from that same
draw inside `render_cad_sample` — identical shot noise, detector noise and
per-row jitter. Two sets at `--shear 0` and `--shear 3` then differ in exactly
one term.

That pairing is not a convenience. It is what makes the `A = 0` arm a **null
control** rather than a differently-seeded lookalike, and it caught a real
misreading: a per-pair "bias" that looked deterministic across amplitudes
turned out to be the jitter's own random slope, which is identical across a
paired sweep by construction.

Two disjoint sweeps, 5 amplitudes × 60 pairs each: **dev** (seed 20260918) and
**held-out** (seed 20261124). Every set is 56–57 present / 3–4 absent.

## 3. Two estimators that failed, and the reason they had to

Both were built, measured against the paired sweep, and rejected. Both show
**unit gain in A** — the physics is right — sitting on a **per-pair additive
offset of sd ≈ 3 px** that survives with zero noise, zero jitter and zero
drift, i.e. is a property of the design canvas rather than of the acquisition.

| estimator | idea | result |
|---|---|---|
| reciprocal-basis non-orthogonality | a shear makes an orthogonal Manhattan lattice non-orthogonal; rotation cannot, and cancels between the two grating families | gain 1.00, per-pair offset sd **3.2 px**; unchanged on a noise-free, drift-free canvas |
| alias-column de-shear | de-shear the full-frame correlation-peak columns until they are sharpest (the route issue #101 proposes) | gain 1.00, offset sd **2.5 px**; additionally collapses under rotation — at 10° the columns tilt 176 px across the frame, and the batch estimate becomes *anti*-correlated with A (fit −1.78·A) |

The reason is structural, not a tuning failure: **a global shear maps a lattice
to a lattice.** Nothing in one frame separates "this layout was sheared" from
"this layout's own lattice is slightly oblique". On the generator's canvases
that obliquity is real at the 0.02 px-per-lattice-step level, which is exactly
the size of the signal across 999 rows.

This is why the scan-distortion literature estimates affine drift from **two**
acquisitions rather than one:

* Ophus, Ciston & Nelson, *Ultramicroscopy* **171** (2016) 104
  ([arXiv:1507.00320](https://arxiv.org/abs/1507.00320)) — image pairs with
  **orthogonal scan directions**.
* Sang & LeBeau, *Ultramicroscopy* **138** (2014) 28 — a revolving series;
  works from a single frame only because a crystal's `[001]`/`[010]`
  orthogonality is exact. Ours is not.
* Snella, *Drift correction for scanning-electron microscopy*, MIT (2010) —
  affine fit between successive images.
* Jones et al., *Adv. Struct. Chem. Imaging* **1**:8 (2015) — non-rigid
  registration against a reference, which is the family the shipped estimator
  belongs to.

A CD-SEM design-based-metrology search turned up nothing that estimates
single-frame scan drift against a GDS reference, so this is not a known
solved problem being re-derived.

## 4. What is identifiable, and what it costs

The one quantity a single frame *does* identify is the displacement field
between the frame and a **known undistorted model of its content** — and Phase
3 hands that over: the reference is the exact design.
`matching.row_offsets` aligns the posed template's rows against the search's,
so any obliquity the design carries is on both sides and cancels, and the slope
of those offsets against row index is `-A/(h-1)` with **no geometric bias**.

Two terms live in that slope:

```
row fit:     dx/dy = -A/rows - tan(dtheta)
column fit:  dy/dx =          +tan(dtheta)
```

`dtheta` is the decode's own rotation error, and it is not a nuisance that
averages away — over a 100-row window rotation and shear are first-order
degenerate, so `polish_pose` absorbs part of the shear into the angle. Measured
on the dev sweep, the row fit **alone** reads only 0.32 of the amplitude.
Raster drift is x-only (`map_y` is the identity), so the transposed fit sees
`dtheta` and not the shear:

| column weight | pooled gain | max sweep residual | per-pair sd |
|---|---|---|---|
| 0.0 | 0.32–0.54 | 0.47 | 4.0 |
| **0.5** | **0.80** | **0.54** | **4.7** |
| 1.0 | 0.97 | 0.42 | 9.9 |
| 1.5 | 1.16 | 0.39 | 13.4 |

0.5 is the optimum on pooled accuracy. Restricting sites to ±300 px of the
reported centre matters for the same reason: sites in a neighbouring mat carry
their own fabrication distortion and scatter at sd 12 against 3.5 for sites in
the reference's own mat.

### Why the correction is pooled over the batch

One site spans ~100 rows, where the ramp is only `0.1*A` px, and the per-row
1-D offset read carries ~1.3 px of noise. Measured over ten sets:

```
per-pair amplitude sd : 4.47 px
pooled (batch) SE     : 0.79 px     (~45 measured pairs per set)
```

Against a signal that only ranges over 0–4 px, **per-pair estimation does not
work**, and the module does not pretend otherwise: `measure` is per-pair,
`pool` is per-batch, and one amplitude is applied to the whole run. Note where
the noise comes from — the 0.5 px per-row jitter alone would allow ~1.7 px, so
this is reader-limited, not jitter-limited. A better per-row offset estimator
is the lever if anyone revisits it.

**This couples the pairs of one run.** With `--shear-correct` on, a pair's
reported `x` depends on the other pairs in the same batch. That is why the flag
is off by default, and it is the one design decision in this change that is not
purely technical.

## 5. Calibration and the gate

The pooled reading is linear in A and is inverted through a fitted line. Each
sweep's own fit, and what each costs when transferred to the other:

| fitted on | line | applied to | mean localisation delta |
|---|---|---|---|
| dev | `0.8252·A − 0.0011` (max resid 0.10) | held-out | **+3.93 / 40** |
| held-out | `0.6892·A − 0.1840` (max resid 0.21) | dev | **+3.91 / 40** |
| both (shipped) | `0.7572·A − 0.0926` (max resid 0.51) | — | — |

The ~16% gain spread between two 60-pair sweeps is the honest transfer error
and is reported rather than tuned away. The shipped constants are the fit over
all ten sets, which is what the leave-one-sweep-out agreement above licenses.
Erring low is the safe direction: an under-correction still removes most of the
bias, while an over-correction adds error of its own.

`SHEAR_GATE = 1.5` px. Below it the reading is not worth acting on, and this is
what makes the null control exact rather than merely small.

## 6. Acceptance

Shipped calibration, gate 1.5, scored with `scripts/score_phase3.py`'s
localisation block.

**Held-out sweep** (seed 20261124, 57 present pairs per set):

| A | Â | Â − A | loc off | loc on | Δ | median off | median on | ≤1 px off | ≤1 px on |
|---|---|---|---|---|---|---|---|---|---|
| 0.0 | −0.396 | −0.396 | 29.61 | **29.61** | **+0.00** | 0.404 | 0.404 | 75% | 75% |
| 1.0 | 1.029 | +0.029 | 28.07 | 28.07 | +0.00 | 0.859 | 0.859 | 61% | 61% |
| 2.0 | 1.851 | −0.149 | 24.00 | 27.51 | +3.51 | 1.707 | 0.569 | 28% | 67% |
| 3.0 | 2.689 | −0.311 | 21.75 | 28.91 | +7.16 | 2.138 | 0.490 | 11% | 70% |
| 4.0 | 3.325 | −0.675 | 18.25 | 28.21 | +9.96 | 2.826 | 0.595 | 4% | 67% |

Mean **+4.13 / 40** over the sweep (**+3.93** with the dev-only calibration,
i.e. strictly held out).

**Dev sweep**, same calibration: +0.00, +0.00, +4.00, +6.71, +10.71 → mean
**+4.29 / 40**.

### Null control

`A = 0` and `A = 1` are gated off and the output is **byte-identical**. Through
the real entry point, on the held-out `A = 0` set:

```
shear-correct off : loc 29.61/40  med 0.404px  <=1px 41/57   SUBTOTAL 69.72/85
shear-correct on  : loc 29.61/40  med 0.404px  <=1px 41/57   SUBTOTAL 69.72/85
# shear: n=43 batch_median=-0.3922 estimate=-0.3957 applied=0.0000
```

And on the held-out `A = 4` set:

```
shear-correct off : loc 18.25/40  med 2.826px  <=1px  2/57   SUBTOTAL 58.84/85
shear-correct on  : loc 28.21/40  med 0.595px  <=1px 37/57   SUBTOTAL 68.80/85
# shear: n=38 batch_median=2.4248 estimate=3.3246 applied=3.3246
```

Scale, rotation, rejection F1 and calibration AUC are **identical** in both
runs. The correction moves `x` and nothing else, by construction.

### Runtime

`drift_shear.measure` costs **48–56 ms per pair** (median over ten sets).
End to end, four alternating runs of the same 60-pair set:

```
off  median 1.928  1.949     on  median 2.005  1.965   s/pair
```

**+0.05 s/pair, about +2.6%.**

## 7. Reproducing

```bash
# a paired amplitude sweep (I4C_ROOT is the organizer checkout)
for A in 0 1 2 3 4; do
  I4C_ROOT=../drift-sense-i4c python scripts/gen_phase3_rotated.py \
      --n 60 --rot 10 --shear $A --paired --seed 20260918 --out /tmp/p3/dev_A$A
done

# decode once, measure once, then explore the calibration for free
for A in 0 1 2 3 4; do
  python experiments/phase3/shear_sweep.py decode  --root /tmp/p3/dev_A$A
  python experiments/phase3/shear_sweep.py measure --root /tmp/p3/dev_A$A
done
python experiments/phase3/shear_sweep.py table \
    $(for A in 0 1 2 3 4; do echo --root /tmp/p3/dev_A$A; done) --calibrate

# end to end
python phase3.py --input /tmp/p3/dev_A3/pairs.csv --output /tmp/pred.csv --shear-correct
python scripts/score_phase3.py /tmp/pred.csv /tmp/p3/dev_A3/ground_truth.csv run
```

## 8. `search_gds_path` IS the full search canvas -- correction

An earlier revision of this document claimed the opposite. It was wrong, and the
error is worth recording because it came from checking one of the two export
paths.

`generate_cad_dataset.py`, the organizer's batch CLI, writes only
`reference_gds_path` -- which is what the first check looked at. But `app.py`,
the bookmark/export path that produces the published curated sets, writes both:

```python
# drift-sense-i4c/app.py:526-528
zf.writestr(f"{prefix}/reference.gds", _gds_bytes(sample["reference_cell"]))
full_cell = _full_canvas_cell(geom["mats"], geom["num_layers"])
zf.writestr(f"{prefix}/search.gds", _gds_bytes(full_cell))
```

`_full_canvas_cell` is the **whole 10000x10000 nm design canvas**, not the
reference site. `driftsense.pairs3.WITHHELD_FIELDS` already said as much --
only `reference_sem_path` and `params_json_path` are withheld, never
`search_gds_path`.

**That makes the estimator in this document the fallback, not the main line.**
With the full search canvas the registration is a direct geometric problem:
PR #105 reports 85.00/85 on an 800-pair dev split through it, against the
~70/85 the image matcher reaches. Nothing here is needed on that path.

What this module is still for:

* the image-matcher path PR #105 falls back to when `search.gds` is missing or
  is not the whole canvas -- PR #105's own caveat records that fallback as
  "much weaker on rotated pairs", and the shear bias measured in section 1 is
  a large part of why;
* the measurement in sections 1-3, which stands on its own: the 104% pass-through
  of the label bias, and the demonstration that single-frame shear estimation is
  degenerate without a frame-wide model of the design. Both remain true, and the
  second is precisely *why* the search canvas is worth so much.

Two further points from the organizers' finals briefing that this measurement
depends on:

* **The Search image rotates; the CAD reference does not** (*"in this phase CAD
  is not rotated, but the other guy can rotate"*). Every number here is measured
  at rotation U(-10, +10) for that reason, which is also why the rotation-error
  cancellation in section 4 is load-bearing rather than a refinement.
* **Barrel distortion will not appear in the graded data** (*"we will not have
  a data set which has a barrel distortion"*). A radial distortion would add a
  row-dependent x term this estimator would read as shear. `barrel_distortion_k`
  is 0 throughout, and should stay 0 in any set used to re-measure this.

# Sub-pixel: the residual localisation error is the centre scan row's drift sample

**Date:** 2026-09-02 · Branch `subpixel/centre-row-drift` off `phase2` (c6e7e8d)
· All numbers below measured on `data/ext_p2`, shipped weights.

This doc **supersedes the "sub-pixel is bounded out" verdict** in
`PHASE2_STATE.md` §3c/§3e and `SETB_WHERE_THE_POINTS_ARE.md`. Those measured a
*rigid* estimator and generalised its ceiling to every inference-side method.
The residual is not label noise; it is a specific, measurable, single-row
displacement, and part of it is recoverable without retraining.

---

## 1. The error is one-dimensional

`.agents/cand_base_nb.csv`, 1,750 present pairs:

| set | median &#124;dx&#124; | median &#124;dy&#124; | ratio |
|---|---:|---:|---:|
| A | 0.3103 | 0.0290 | 10.7x |
| B | 0.7930 | 0.0812 | 9.8x |

y is already essentially exact. The whole 40-point localisation budget is x.

## 2. Why: drift is horizontal, per-row, and the label reads one row

`generator/src/sem_imaging.py:126-128` perturbs **`map_x` only**, one value per row:

```python
row_shift = (shear + jitter).astype(np.float32)
map_x = np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None]
```

`driftsense/generate.py:386` draws that jitter **i.i.d. per row**
(`rng.normal(0, p.drift_jitter_px, size=h)`), and `generate.py:463-464` takes
the shift of the **single row** the target centre lands on:

```python
row = int(np.clip(round(py), 0, len(row_shift) - 1))
ax, ay = px - float(row_shift[row]), py
```

`refine_zncc` correlates the whole ~100-row template, so it recovers the row
*average*. Residual = `row_shift[centre] - mean(row_shift)`, and because the
jitter is white, no rigid fit and no smoothing can remove it.

## 3. The model predicts the measured error

Joined to per-pair `drift_jitter_px` from the shard manifests, 8 bins over a
14x range of drift:

| set | `std(dx) / drift_jitter_px` | `corr(dx, shear)` |
|---|---:|---:|
| A (n=865) | **1.112** | −0.071 |
| B (n=810) | **1.083** | +0.024 |

Slope 1.0 is the prediction. The smooth shear term is already fully absorbed by
the existing matcher; only the white per-row part survives.

A **row-lag scan** settles it — correlating the correction built from row
`centre+k` against the observed `dx`:

| row lag k | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| corr | −0.045 | −0.050 | −0.035 | **+0.340** | +0.055 | +0.048 | +0.052 |

A sharp isolated peak on exactly the labelled row, with neighbours at noise.
That is the signature of a per-row white process and it is what identifies the
mechanism.

**Also ruled out:** no quantisation defect (`frac(pred x)` and `frac(dx)` are
uniform, chi2 4.5 and 7.0 on 9 df) and no removable bias (subtracting global or
per-severity median `dx` moves set B only 0.729 -> 0.693).

## 4. Why §3e's bound was the wrong family

§3e measured the best rigid ZNCC at **1.077 px** with true pose supplied, against
set B median drift of ~1.1 px. That is not a coincidence: a rigid fit recovers
`mean(row_shift)` while the label wants `row_shift[centre]`, so its error is
**exactly σ by construction**. §3e correctly bounded rigid estimators and then
generalised the bound to all inference-side methods. Its own explanation —
"no single (x, y) aligns a wobbled pattern to a flat one" — is the argument for
a non-rigid estimator, not against one.

This also explains why **centre-weighting the ZNCC changed nothing (1.077 ->
1.077)**: a soft weight over many rows still averages, and averaging k rows
converges to the population mean, not to `row_shift[centre]`. Only k = 1 works.

## 5. The estimator

`driftsense.matching.drift_row_refine`, enabled by `subpixel_rows`
(`SHIPPED_SUBPIXEL_ROWS = True` after the A/B below; set it to False to revert):

1. `row_offsets` — 1-D normalised correlation of each search row against the
   corresponding row of the posed template (`make_template` already returns the
   reference in the search frame, so rows correspond one-to-one), sub-pixel by
   the existing `parabolic`.
2. Dewarp the patch by the measured per-row offsets and **re-run `refine_zncc`**
   on the flattened patch.
3. Re-apply the centre row's own offset.

Step 2 matters: correcting x post-hoc from the row offset alone is worth only
~0.5 correlation, because the same wobble also degrades the rigid match that
produced x. Dewarping fixes both.

### Measured design choices

Per-row measurement accuracy, from injecting a known extra row shift into real
frames and re-reading it: **median 0.040 px**, corr 0.99+ against truth. The
per-row read is not the limit.

Recovery of the drift field's amplitude: `sd(white residual of d) /
drift_jitter_px` = **1.000**, corr **+0.922** over a 10x range. The field is
read at full amplitude.

Knob sweeps on all 1,750 present pairs (localisation /40):

| config | applied | loc /40 |
|---|---:|---:|
| shipped (off) | — | 35.71 |
| **min_corr 0.30, iters 1, lag 12, clamp 3.0** | 81% | **36.29** |
| min_corr 0.30, iters 2 | 81% | 36.29 |
| min_corr 0.40 | 73% | 36.28 |
| min_corr 0.20 | 89% | 36.20 |
| min_corr 0.50 | 66% | 36.24 |
| min_corr 0.15, iters 3, lag 16 | 92% | 36.13 |
| clamp 1.5 px | 81% | (sample) 37.73 vs 38.01 at 3.0 |
| clamp 1.0 px | 81% | (sample) 37.53 vs 38.01 at 3.0 |

Two results worth keeping:

* **Loosening `min_corr` to correct more pairs is worse.** A badly measured row
  is worse than no correction, so declining ~19% of pairs is deliberate.
* **The clamp is load-bearing.** Without it the re-match occasionally jumps to a
  wrong periodic repeat: `<=1px` still improved while total credit *fell*,
  because a handful of pairs were thrown past 5 px into zero credit.
* One iteration equals two, so the estimator runs a single pass.

A gate on estimated drift amplitude was swept (0.20–0.70 px) and is **not**
worth adding: best gate 0.40 gave 36.30 against 36.29 ungated.

## 6. Result — full 2,500-pair paired A/B (2026-09-02)

Both arms run end to end through `scripts/eval_ext.py` on the same six shards,
shipped weights, `SHIPPED_THRESHOLD = 0.18`.
Logs: `.agents/eval_sp_off.log`, `.agents/eval_sp_on.log`;
frames: `.agents/cand_sp_off.csv`, `.agents/cand_sp_on.csv`.

| component | off | on | delta |
|---|---:|---:|---:|
| localisation (40) | 35.46 | **36.05** | **+0.59** |
| &nbsp;&nbsp;set A credit | 0.9737 | **0.9785** | +0.0048 |
| &nbsp;&nbsp;set A `<=1px` | 92.7% | **95.1%** | +2.4 pp |
| &nbsp;&nbsp;set B credit | 0.8151 | **0.8379** | +0.0228 |
| &nbsp;&nbsp;set B `<=1px` | 57.6% | **67.1%** | **+9.5 pp** |
| &nbsp;&nbsp;set B median err | 0.83 px | **0.62 px** | −0.21 px |
| pose scale (10) | 8.96 | 8.96 | 0.0000 |
| pose rotation (10) | 9.02 | 9.02 | 0.0000 |
| rejection (15) | 13.62 | 13.62 | 0.0000 |
| calibration (10) | 9.88 | 9.88 | 0.0000 |
| **TOTAL / 85** | **76.94** | **77.53** | **+0.59** |
| set D bonus credit | 0.9808 | 0.9832 | +0.0024 |

**v2 (pitch unwrapping + drift-scaled clamp), same protocol:**

| | shipped | v2 | delta |
|---|---:|---:|---:|
| localisation (40) | 35.46 | **36.09** | **+0.63** |
| set A `<=1px` | 92.7% | **95.5%** | +2.8 pp |
| set B `<=1px` | 57.6% | **68.1%** | **+10.5 pp** |
| set B median err | 0.83 px | **0.59 px** | −0.24 px |
| **TOTAL / 85** | **76.94** | **77.57** | **+0.63** |

Paired bootstrap: v2 vs shipped **+0.6258**, 95% CI [+0.4165, +0.8380],
P(delta >= +0.35) = **0.995**. 201 pairs improved, 81 worsened, net +120.

**v2 vs v1 is a wash and is NOT established: +0.0366, 95% CI [−0.0782,
+0.1500], P(>0) = 0.736.** The +0.17 that v2 showed on a 900-pair sample did
not replicate on the full 2,500. v2 is kept because it is non-inferior on every
component and its point estimate is positive -- the same basis on which the p9
weights were promoted -- not because it beat v1. `v1` (no unwrap, fixed 3.0 px
clamp) is the simpler fallback if the extra code is not wanted.

**Localisation-only, verified at full precision:** across all 2,500 pairs
`scale`, `theta`, `score` and `y` are byte-identical between arms; only `x`
differs (1,640 pairs). The small pose/calibration movements in the report come
from which pairs *qualify* for pose scoring (it is gated on localisation), not
from changed pose values.

**Every non-localisation component is bit-identical**, including the rejection
counts (correct-rej 453, lost-real 45, missed-abs 47 in both arms). That is by
construction -- the correction runs after every pose decision is final and
writes only `best["x"]`, so it cannot move scale, rotation, the confidence or
the threshold behaviour. It is also the cheapest available check that the hook
is wired where it claims to be.

### Paired bootstrap (10,000 resamples, seed 0, aligned by `pair_id`)

| | |
|---|---:|
| point delta | +0.5893 |
| median | +0.5881 |
| 95% CI | **[+0.4101, +0.7730]** |
| P(delta > 0) | 1.0000 |
| **P(delta >= +0.35)** — the issue #19 promotion gate | **0.9943** |

250 pairs changed tier: **180 improved, 70 worsened, net +110.**

The gate is cleared at 99.4%. For contrast, the Set C fine-tune in PR #34
reached P = 0.429 on the same gate and was promoted as "best measured" rather
than as a cleared gate.

### Runtime

Isolated cost of `make_template` + `drift_row_refine`, single process, 4 threads,
120 real pairs: **median 2.2 ms, p90 2.6 ms, max 6.2 ms** (v1 was 1.9 / 2.1 / 2.3) — against a 50 ms
budget and a ~1.4-3.5 s pair. `register.py` end-to-end on the smoke set: median
0.66 s. The change does not touch issue #7's coarse-sweep bottleneck.

### What it does not reach

The ceiling if the x-error collapsed entirely is **+2.38** (tier redistribution
holding the 65 gross `>5px` pose failures fixed; the curve saturates at a
residual of ~0.4 px because the tier is `<=1 px`). We capture about a quarter of
it. The remaining gap is severity 4, where the per-row peaks are hardest to
read; the declines are not recoverable headroom (section 8).

## 7. Open

- Severity 4 improves least; at σ≈2 px the row peaks are hardest to read.
- The decline rate is **not** a lever -- see section 8, where every attempt to
  raise coverage measured neutral or worse.
- Barrel distortion is applied to the search frame *after* drift and inverted in
  the label, but is not modelled per-row here. Within a 100 px window it acts as
  a mild stretch and will bias the row correlation slightly.
- This is *our* generator's drift model. The method degrades to a no-op when
  there is no per-row jitter, so the downside on a differently-drifted blind set
  is bounded at zero rather than negative.

## 8. The decline rate is the optimum, not a limitation (measured 2026-09-02)

The estimator declines ~19–30% of pairs depending on the draw. Four separate
attempts to apply it to more pairs, 700–900 present pairs each:

| attempt | applied | loc /40 |
|---|---:|---:|
| current (centre gate 0.30, total clamp 3.0 px) | 69% | **36.33** |
| centre gate 0.20 | 76% | 36.14 |
| centre gate 0.12 | 78% | 36.09 |
| dewarp-only fallback when the centre row is unusable | 87% | 36.25 |
| total clamp 4.0 px | 70% | 36.37 |
| total clamp 6.0 px | 72% | 36.09 |
| **no total clamp** | 80% | **33.32** |
| guard the re-match (`<=2 px`) instead of the total | 80% | 33.32 |
| re-match guard + centre offset capped at 3 sd | 79% | 33.78 |
| shipped (no correction) | — | 35.80 |

Decline reasons on 700 pairs: centre row unusable 119, clamp 71, window 18,
too few rows 4, applied 488.

Three results worth not re-deriving:

1. **The clamp is load-bearing and cannot be replaced by a smarter guard.**
   Removing it costs 3.0 points — well below doing nothing. Guarding the
   *re-match* displacement instead changes nothing at all, because that guard
   never fires: the runaway is entirely in the centre-row offset term, where the
   1-D correlation has locked onto the wrong lattice repeat. Only a bound on the
   total displacement catches it.
2. **The dewarp on its own is worth exactly 0.0000.** The dewarp-only fallback
   applies to 17% more pairs and reproduces the baseline credit to four
   decimals. Flattening the wobble just re-finds the same mean-frame position
   the rigid match already had; *all* of the gain is the centre-row offset.
   The dewarp earns its place only by giving that offset a cleaner base.
3. **Loosening the centre-row gate is monotonically worse** (0.30 -> 0.20 ->
   0.12 gives 36.25 -> 36.14 -> 36.09), confirming decoupled what the global
   `min_corr` sweep in §5 found. A badly measured row is worse than no
   correction, so declining is the correct behaviour and not a coverage bug.

Clamp 4.0 scored +0.04 over 3.0 on 900 pairs — inside noise, not adopted.


## 9. Research campaign: what else was tried (2026-09-02)

Seven ideas from the sub-pixel / DIC / PIV literature, each measured on 900
present pairs against the same baseline. **Two worked, five did not.**

| idea | applied | set B `<=1px` | loc /40 | verdict |
|---|---:|---:|---:|---|
| v1 baseline | 69% | 67.8% | 36.33 | reference |
| **lattice-pitch unwrapping** | 77% | **70.0%** | **36.50** | **kept** |
| **drift-scaled clamp** | — | — | — | **kept (stacks)** |
| bandpass row prefilter (1-D DoG along x) | 73% | 67.2% | 36.33 | tied, dropped |
| PSR reliability gate (>4) | 12% | 60.0% | 35.88 | much worse |
| PSR gate (>3) + bandpass | 32% | 62.6% | 36.02 | much worse |
| absolute + wide differential fusion | 75% | 55.4% | 34.91 | **much worse** |
| Gaussian peak fit | 77% | 68.9% | 36.44 | worse |
| centroid peak fit | 77% | 67.6% | 36.30 | worse |

Findings worth not re-deriving:

* **The layout pitch is ~9.9 px** (median, resolvable on 94% of pairs from the
  template's own row autocorrelation) against drift sd <= 2.1 px. So a row
  sitting a whole pitch off the field is a repeat error, not a 5-sigma sample.
  Unwrapping those is the one idea here that moved every component at once.
* **PSR is the wrong reliability metric for a periodic layout.** The lattice
  puts large sidelobes in every correlation curve, so PSR is low even for good
  rows and admits only 12-18% of them.
* **Fusing a wide row-to-row differential is actively harmful** (set A 94.1% ->
  80.5%). Adjacent rows of a real layout are not copies of each other, so the
  differential conflates vertical layout structure with drift. The earlier
  observation that its *amplitude* tracks `drift_jitter_px` does not make it
  unbiased per row.
* **Peak-locking is present but is not the binding constraint.** The parabolic
  fit's offsets are strongly non-uniform (chi2 = 248 on 9 df against a 21.7
  critical value, the classic U-shape piling up at integers), yet both standard
  remedies made things worse -- a normalised correlation peak on a periodic
  lattice is not bell-shaped, so the Gaussian model is misspecified and the
  centroid is pulled by sidelobes.

## 10. The ceiling, and why 90% on set B needs the pose track

| set B `<=1px` | |
|---|---:|
| shipped | 57.6% |
| **v2 (this work)** | **68.1%** |
| where the correction applies today | 81.1% |
| ceiling with a *perfect* x correction | **91.7%** |

The 91.7% wall is not estimator quality: **7.4% of set B pairs are gross >5 px
pose-basin failures** where the match sits on the wrong lattice site entirely,
and no `x` correction can rescue a pair whose pose is wrong. That is issue #37 /
PR #35. Sub-pixel work alone tops out near **81%** -- the quality the correction
already reaches where it applies; the remainder is coverage on pairs whose rows
are genuinely unreadable. **90%+ requires the rotation-aware pose fix merged
alongside this**; the two address different halves and should compound.

## 11. Fresh-holdout validation (2026-09-02, PR #41 review)

The §6 and §9 numbers are **post-selection**: `min_corr`, the clamp, the lag and
the iteration count were swept on all 1,750 present pairs of `data/ext_p2`, and
the headline A/B was then run on that same pool. The reported CI therefore
treated tuned-on examples as untouched evaluation data. Flagged in review; this
section is the fix.

**Protocol (single go/no-go, no further tuning).** The implementation was frozen
first. A fresh 500-pair Phase 2 set was generated at the previously unused seed
`20260902` (`scripts/gen_data.py --split holdout_p2 --num-samples 500 --seed
20260902 --noise randomized --phase2 --absent-frac 0.2 --crops-per-canvas 1`),
and the run was done on the **intended final stack** — `phase2` + #34 (its Set C
checkpoint in `weights/driftsense.pt`) + #35 with `RERANK_ROTATION = False` —
rather than on `phase2` alone, because #34 changes the checkpoint every decode
depends on.

**Independence, checked two ways and both non-vacuous:**

| check | holdout | ext_p2 | overlap |
|---|---:|---:|---:|
| distinct `sample_entropy` | 500 | 2500 | **0** |
| distinct search-image sha256 | 500 | 2500 | **0** |

(The first attempt at this check used `pair_sha256` and returned "0 overlap"
**vacuously** — `gen_data.py` writes a leaner manifest that has no such column,
so the holdout side of the comparison was an empty set. Both checks above are
confirmed populated on both sides.)

**Result — 500 pairs, 405 present / 95 absent:**

| component | off | on | delta |
|---|---:|---:|---:|
| localisation (40) | 35.48 | **36.48** | **+1.00** |
| &nbsp;&nbsp;set A `<=1px` | 89.3% | 90.0% | +0.7 pp |
| &nbsp;&nbsp;set A credit | 0.9453 | 0.9453 | **0.0000** |
| &nbsp;&nbsp;set B `<=1px` | 56.1% | **72.9%** | **+16.8 pp** |
| &nbsp;&nbsp;set B median err | 0.88 px | **0.48 px** | −0.40 px |
| pose scale (10) | 8.198 | 8.199 | +0.001 |
| pose rotation (10) | 8.869 | 8.889 | +0.020 |
| rejection (15) | 13.44 | 13.44 | **0.0000** |
| calibration (10) | 9.86 | 9.83 | −0.026 |
| **TOTAL / 85** | **75.84** | **76.84** | **+1.00** |

Paired bootstrap: median **+1.0045**, 95% CI **[+0.5184, +1.4780]**,
P(delta > 0) = **1.0000**. 70 pairs improved a tier, 27 worsened, net +43.

**The effect is larger on untouched data than on the tuned pool** (+1.00 vs
+0.63), so the post-selection concern does not reverse the sign — the tuned-pool
estimate was, if anything, conservative. The holdout draws drift from
`U(0.1, 2.0)` (median 1.01 px) rather than the severity ladder, so more of its
pairs carry the drift the correction recovers; the two effect sizes are not
expected to match.

**Set A / set B here is our own drift-based labelling** (`<=0.80 px` = A, the
`ext_p2` set A ceiling), not the organizer's set definition, since a
`--noise randomized` split has no A/B structure. The pooled +1.00 is the
headline; the split is reported for shape only.

**Two components moved, both accounted for:**

* **`<=5px` tier: 2 pairs pushed out, 0 rescued** (set B 96.9% -> 96.1%). Real,
  small, and the honest cost of the change against +43 net tier improvements.
* **Calibration AUC −0.0026.** `score` is byte-identical on all 500 pairs, so
  this is not a confidence regression: AUC ranks confidence against
  *correctness*, and correctness is defined by localisation, so improving
  localisation reshuffles which pairs count as correct.

`scale`, `theta`, `score` and `y` are identical on all 500 pairs; only `x`
differs (310 pairs) — the same localisation-only property verified in §6.

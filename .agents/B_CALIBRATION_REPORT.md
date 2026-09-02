# B: Calibration / AUC workstream — findings (in progress)

**Date:** 2026-08-30 (workstream B) · **Data:** `.agents/ext_features_full.csv` (2,250 pairs A/B/C, own holdout generator, never trained on)
**Protocol:** identical to `scripts/rejector_cv.py` — 4-fold, `RandomState(0).permutation(n) % 4`, threshold grid = train-fold quantiles `linspace(0.001, 0.6, 200)` picked on the train fold's **total rubric** (`points()`), AUC per `optimize_threshold.py` (`correct = pres & err<=5` vs the rest, ties 0.5). Logistic = plain GD (`fit_rejector.fit_logistic` hyperparams: iters=4000, lr=0.5, l2=1e-3, standardised, no bias decay), statistic = `-P(absent)`. Fit on train folds only; all reported numbers held-out unless marked in-sample.

**Reference baselines (REJECTOR_FINDINGS.md Result 1, same protocol):**

| statistic | held-out total | F1(reject-pos) | AUC |
|---|---:|---:|---:|
| shipped `min(score,zncc)` | 75.45 | 0.8716 | 0.9878 |
| logistic: 6 shipped features | 75.59 | 0.8719 | 0.9917 |
| logistic: shipped + rank,band,margin | 75.59 | 0.8738 | 0.9911 |

*(sections below appended as results land)*

## Status / plan

- [ ] Data sanity (class counts, missingness, per-set split)
- [ ] CV table: (a) shipped scalar, (b) 6-feat logistic, (c) 9-feat logistic, (d) new derived features
- [ ] In-sample oracle F1 ceiling (Result-2 analogue)
- [ ] r_delta / peak-width spec (future work, no re-inference run)
- [ ] Threshold re-tune on frozen fused score (both conventions, downward-biased)
- [ ] Frozen constants (fit on FULL 2,250 only after CV recorded) + `driftsense/calibration.py` + tests
- [ ] SHIP / DO-NOT-SHIP recommendation

## Literature grounding

- Monotone maps (Platt/isotonic/temperature) provably cannot change AUC — Guo et al., *On Calibration of Modern Neural Networks*, arXiv:1706.04599. Value must come from the feature vector.
- Strongest new statistic per serial-EM literature: r_delta = primary − secondary correlogram peak, plus peak width — Buniatyan et al., arXiv:1705.08593.

## Result 1 — 4-fold CV, AUC-first (fresh run, this workstream)

Command: `venv313/bin/python scripts/fit_calibration.py` (protocol identical to
`scripts/rejector_cv.py`; rows (a)-(c) reproduce that script's fresh output
exactly, so the numbers are comparable to the recorded 0.9878/0.9917 baselines).

| statistic | held-out total | F1(reject) | AUC |
|---|---:|---:|---:|
| (a) shipped `min(score,zncc)` | 75.54 | 0.8778 | 0.9877 |
| (b) logistic: 6 shipped features | 75.65 | 0.8773 | **0.9915** |
| (c) logistic: 9 = 6 + rank,band,margin | 75.59 | 0.8734 | 0.9911 |
| (d) 9 + score*zncc | 75.56 | 0.8737 | 0.9906 |
| (d) 9 + \|score-zncc\| | 75.61 | 0.8770 | 0.9906 |
| (d) 9 + zncc/score | 75.60 | 0.8756 | 0.9909 |
| (d) 9 + peak_ratio*pose_peak | 75.63 | 0.8740 | 0.9909 |
| (d) 9 + n_hyp | 75.53 | 0.8707 | 0.9911 |
| (d) 6 + all five derived | 75.66 | 0.8789 | 0.9912 |
| (d) all 15 | 75.57 | 0.8760 | 0.9900 |

**Reading:** every variant's held-out AUC sits in the 0.9900–0.9915 band, i.e.
within ±0.0015 of the 6-feature logistic with no winner outside noise. **None
of the new derived features (score*zncc, |score−zncc|, zncc/score,
peak_ratio*pose_peak, n_hyp) improves held-out AUC over (b) 0.9915.** The
AUC gain over the shipped scalar is (b)'s +0.0038 → +0.04 calibration points,
already known from the prior study (recorded there as 0.9917 on a fresh run
of the same protocol; the ±0.0002 difference between the two fresh runs is
run-to-run jitter of the same fold assignment, not a protocol difference).
Consistent with Guo et al. (arXiv:1706.04599): the headroom is in which
signals enter the feature vector, and the shipped six already carry it — the
derived combinations are algebraic compositions of signals already present,
so a linear model extracts nothing new from them.

## Result 2 — optimiser convergence (documented, not assumed)

GD (iters=4000, lr=0.5, l2=1e-3) on the full 2250, 9 features:
grad-norm 8.27e-01 → 1.28e-04 (×6,500); max per-iter parameter step at iter
4000 = 4.9e-05; logloss 0.69315 → 0.11778, still decreasing by <2e-4 over the
last 1,000 iterations. IRLS (quadratic convergence, ridge 1e-6) reaches
logloss 0.113913 vs GD's 0.117783 — GD is ~4e-3 logloss short of the true
optimum (weakly-separated features → very flat likelihood), which is why the
FROZEN constants are IRLS-fit, while the CV table keeps GD to stay identical
to rejector_cv.py's protocol. `scripts/fit_calibration.py --convergence-check`
reproduces this table.

## Result 3 — in-sample oracle F1 ceiling (Result 2 analogue of REJECTOR_FINDINGS)

9-feature logistic fitted in-sample on all 2,250 (cheating freely), every
threshold swept: **max F1(reject-positive) = 0.8919** (t = −0.4206). Compare
REJECTOR_FINDINGS.md Result 2: 0.8827 / 0.8850 (different threshold grid:
2,000 quantiles here vs 300 there, which recovers a slightly higher ceiling).
Scope of the bound is the same: thresholding a linear-logistic statistic. The
0.90 bonus bar remains out of reach for this family; the residual errors are
confident wrong lock-ons, not ranking errors — which is also why held-out AUC
(0.9915) and F1 move in different currencies.

## Future work (spec, NOT run) — true r_delta / peak-width features

Buniatian et al. (arXiv:1705.08593) rank correlogram hypotheses by the gap
between the primary and secondary peak. Our `margin` column is the closest
existing analogue but is computed at the *hypothesis* level, not on the
correlation *surface*. The surface statistic is available at zero extra
template-matching cost inside `driftsense/matching.py::refine_zncc`
(line 740-742): `res = cv2.matchTemplate(window, template, TM_CCOEFF_NORMED)`
is already computed over the ±`REFINE_RADIUS`(=4) px refine window and then
thrown away except for its argmax. To record per pair (integrator-owned file,
so this is a handoff spec, not a change I made):

1. `r_delta` = `res.max() - second_local_max(res)` where the second local max
   is the largest value ≥2 px away from the argmax (suppress a disc, take max).
2. `peak_width_hm` = full width at half maximum of `res` through the peak,
   along x and y separately (count samples ≥ max/2 while walking out from the
   peak until it drops below; parabolic-interpolated sub-sample width).
3. Optionally the same statistics on the *coarse* verification response (the
   `locate` heatmap when `return_heatmap=True`), where the surface is large
   enough for a meaningful sidelobe structure — the 9×9-ish refine response
   can only separate hypotheses that are already 2-4 px apart.

Feasibility note: populating these columns for the 2,250 holdout pairs means
one instrumented re-decode (eval_ext --features style, ~90 min on the
reference laptop). NOT run here per campaign instruction. Worth one attempt
only because `margin`'s AUC contribution was itself ~neutral (Result 1
row (c) vs (b)); the surface-level r_delta is a different statistic and the
literature's strongest candidate, but the prior on a large held-out gain is low.

## Result 4 — threshold re-tune on the frozen fused score (full 2,250, in-sample by construction)

The fused statistic is the 9-feature logistic calibrated P(present) (see
Result 5 for the frozen constants; the threshold is expressed in those
probability units). Tuned against the TOTAL rubric (Subtlety 1: a declined
present pair forfeits loc+pose) with the campaign's downward bias, and
reported under BOTH F1 conventions (Subtlety 2).

Reference — shipped scalar `min(score,zncc)` at SHIPPED_THRESHOLD=0.18,
in-sample on the same 2,250: total 75.35, loc 0.8632, F1(reject) 0.8648,
F1(present) 0.9611 (computed separately, fresh), AUC 0.9876, declined-present 71,
declined-absent 435, accepted-absent 65.

| operating point | threshold | total | loc | F1 rej+ | F1 pres+ | AUC | declined-present | declined-absent | accepted-absent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped scalar @ 0.18 | 0.18 | 75.35 | 0.8632 | 0.8648 | 0.9611 | 0.9876 | 71 | 435 | 65 |
| fused, total-rubric optimum | −0.5317 | 75.85 | 0.8674 | 0.8871 | 0.9684 | 0.9910 | 47 | 436 | 64 |
| fused, downward-biased choice | −0.5317 | 75.85 | 0.8674 | 0.8871 | 0.9684 | 0.9910 | 47 | 436 | 64 |

Downward-bias note: the optimum sits on a flat ridge (t = −0.5442 → 75.78,
−0.5317 → 75.85, −0.5111 → 75.83); every grid point within 0.05 total of the
optimum IS the optimum, so the "lowest threshold within margin" rule selects
the optimum itself — the bias margin is already consumed by the total rubric
(it prices declined-present pairs at 40+20+ F1 points, which pushes the
operating point down harder than an F1-only tune would: F1-only tuning on this
statistic would sit near t ≈ −0.30, declining ~90 present pairs).
The in-sample-vs-held-out caveat of optimize_threshold.py applies: the honest
held-out estimate of the fused statistic is the CV total 75.59 (row (c)), not
this 75.85.

**Both-convention F1 under the fused operating point:** reject-positive 0.8871,
present-positive 0.9684. Under the lenient (present-positive) reading the same
operating point is near-optimal too, as PHASE2_STATE.md argues.

The threshold is NOT shipped by this workstream: config.py and register.py are
integrator-owned. Handoff: the tuning grid lives on the −P(reject) scale
(range [−1, 0]); in calibrate() probability units (P(present), range [0, 1])
the operating point is 0.4683 — reject when calibrate(features) < 0.4683.
(An earlier draft of this section said 0.297; that was a unit-conversion
arithmetic error, caught by the freeze self-check, and corrected here.)

## Result 5 — frozen shipped constants (fit on FULL 2,250, AFTER CV)

Command: `venv313/bin/python scripts/fit_calibration.py --freeze`. Optimiser:
the GD of `fit_rejector.fit_logistic` (iters=4000, lr=0.5, l2=1e-3), i.e. the
exact optimiser every CV number above was computed with — NOT IRLS. Rationale
recorded in Result 2b: the features nearly separate the data, the unregularised
MLE diverges, and IRLS(ridge 1e-6) optimises a different (weaker-ridge)
objective; shipping IRLS constants would decouple the artefact from the
evidence. Raw-scale conversion `a = -w/sd` (P(present) parameterisation),
self-checked against the std-space fit: max |P_std − P_raw| = 5.0e-16.

| feature | coefficient (raw scale) |
|---|---:|
| score | +10.60423859851734 |
| zncc | +2.6805808255644283 |
| peak_ratio | −0.4487484332316817 |
| pose_peak | −6.709839254143747 |
| psr | +0.0014481513887072556 |
| apce | +3.976937953099698e-05 |
| rank | +3.240837308768774 |
| band | +0.8526308159408791 |
| margin | −2.3637465613832664 |
| intercept | −0.7961563537640459 |

Sign caveat (honest, documented): pose_peak and margin carry raw-space
NEGATIVE coefficients although each correlates positively with presence
unconditionally. This is normal for a linear model over correlated features
(pose_peak and zncc are near-duplicates; margin overlaps score/zncc) — the
model is a joint ranker and the shipped property is its held-out CV AUC
(0.9915, row (b)-family protocol), not any single coefficient's sign.

`calibrate(features: dict) -> float` returns P(present) in [0, 1]. It is a
pure weighted sum + sigmoid over outputs `locate_phase2` already computes:
zero inference cost, no image access, no second pass.

## Result 6 — tests (fresh output)

```
$ venv313/bin/python -m pytest tests/test_calibration.py -q
.........                                                                [100%]
```

9 tests, all passing: feature-list/coef consistency, [0,1] over a 500-point
random sweep of the plausible ranges, present-like > absent-like, monotone in
score holding others fixed, exact sigmoid identity, missing-key raises,
fit determinism (bit-for-bit), fit learns separable signal, and the
frozen-coefficients regression pin (accidental refits fail loudly; a
placeholder sentinel also fails the suite if constants are ever emptied).

## Result 7 — SHIP / DO-NOT-SHIP

**RECOMMENDATION: SHIP the 9-feature calibration statistic (component (b)'s
feature vector), DO NOT SHIP any of the new derived features (d), and treat
the threshold change as optional integrator work.**

- AUC-first campaign: held-out AUC 0.9878 → 0.9915 (+0.0038 ≈ +0.04
  calibration points) with zero inference cost, fitting the existing 6
  features. This reproduces the prior study's 0.9917 within run jitter —
  it is the same measured effect, now verified fresh under the AUC-first
  priority.
- No derived feature (product, gap, ratio, peak×pose, n_hyp) or feature
  combination beats (b) on held-out AUC. The frozen 9-feature constants are
  shipped because (c) ≈ (b) within noise and the 9-feature vector is what the
  integrator's decode already records with --features; the AUC case for
  dropping rank/band/margin is not strong enough to justify the churn.
- F1 does NOT improve (0.8778 → 0.8773, within noise): consistent with
  REJECTOR_FINDINGS Result 2 — the residual errors are confident wrong
  lock-ons, not ranking errors. The +4 bonus bar (0.90) remains out of reach
  (oracle 0.8919 in-sample).
- If held-out AUC is the whole scoring question, this ships; if the grader
  also weights F1, the gain is +0.04 total-rubric points (AUC) − 0.01 (F1
  jitter) — positive but tiny, and the integrator may reasonably decline the
  change on churn grounds.
- Threshold: the fused statistic's total-rubric optimum (0.4683 in
  calibrate() units) yields in-sample total 75.85 vs the shipped scalar's
  75.35 on the same 2,250 — but the honest held-out estimate of the fused
  statistic is the CV total 75.59 (row (c)), i.e. +0.14 vs the scalar's 75.54
  CV total, not +0.50. Integrator decides whether a threshold/config change
  is worth +0.14 expected points.

## Reproduce

```
venv313/bin/python scripts/rejector_cv.py .agents/ext_features_full.csv   # baseline cross-check
venv313/bin/python scripts/fit_calibration.py                             # CV table
venv313/bin/python scripts/fit_calibration.py --oracle                    # F1 ceiling
venv313/bin/python scripts/fit_calibration.py --convergence-check         # optimiser evidence
venv313/bin/python scripts/fit_calibration.py --retune                    # threshold, both conventions
venv313/bin/python scripts/fit_calibration.py --freeze                    # frozen constants
venv313/bin/python -m pytest tests/test_calibration.py -q                 # suite
```

---

# ADDENDUM — integrator constraint: shipped default path has no rank/band

The integrator verified in `driftsense/matching.py` that at inference time on
the SHIPPED default path (`verification="zncc"`, no `return_hypotheses`) the
available features are exactly `score, zncc, peak_ratio, pose_peak, psr, apce`
(locate() lines 1164-1166) plus `winner_margin` (line ~1032, every path);
`rank`/`band` exist only when `verification != "zncc"` or
`return_hypotheses=True` (lines 874-886). The SHIP recommendation is therefore
rebuilt from ship-eligible candidates only. Variant (c) with rank/band is kept
below purely for comparability with rejector_cv.py.

## Result 1b — revised CV: ship-eligible (b*) vs reference trials (fresh run)

`venv313/bin/python scripts/fit_calibration.py`. A trial is SHIP-eligible iff
every feature is in `{6 shipped} ∪ {margin} ∪ {min/prod/gap/ratio/peak×pose
derivatives of score,zncc}` (derivatives are legal: computable at the
feature-extraction step from available outputs; `min(score,zncc)` itself is
one of them). `n_hyp` is NOT in the integrator's available set — its trial is
marked (ref).

| statistic | ship-eligible? | held-out total | F1(reject) | AUC |
|---|---|---:|---:|---:|
| (a) shipped `min(score,zncc)` | — (the scalar itself) | 75.54 | 0.8778 | 0.9877 |
| (b) logistic: 6 shipped features | YES | 75.65 | 0.8773 | **0.9915** |
| (b*) 7 = 6 + margin | YES | 75.57 | 0.8762 | 0.9907 |
| (b*) 7 + min | YES | 75.57 | 0.8716 | 0.9900 |
| (b*) 7 + prod | YES | 75.70 | 0.8781 | 0.9902 |
| (b*) 7 + gap | YES | 75.61 | 0.8738 | 0.9902 |
| (b*) 7 + ratio | YES | 75.57 | 0.8770 | 0.9904 |
| (b*) 7 + peak×pose | YES | 75.64 | 0.8768 | 0.9905 |
| (b*) 6 + min | YES | 75.66 | 0.8791 | 0.9914 |
| (b*) 6 + prod | YES | 75.58 | 0.8723 | 0.9913 |
| (b*) 6 + gap | YES | 75.61 | 0.8759 | 0.9915 |
| (b*) 6 + ratio | YES | 75.60 | 0.8745 | 0.9914 |
| (b*) 6 + peak×pose | YES | 75.56 | 0.8712 | 0.9915 |
| (b*) 7 + min,prod,gap,ratio,peak×pose | YES | 75.51 | 0.8692 | 0.9895 |
| (b*) 7 + min + prod | YES | 75.55 | 0.8704 | 0.9898 |
| (b*) 7 + n_hyp | ref: n_hyp not on default path | 75.76 | 0.8846 | 0.9907 |
| (c) 9 = 6 + rank,band,margin | ref: rank/band unavailable | 75.59 | 0.8734 | 0.9911 |
| (d) 9 + each derived (5 rows) | ref | 75.53–75.63 | — | 0.9906–0.9911 |
| (d) 6 + five derived | ref | 75.66 | 0.8789 | 0.9912 |
| (d) all 15 | ref | 75.57 | 0.8760 | 0.9900 |

**Ship-eligible winner by held-out AUC: (b) the plain 6-feature logistic,
AUC 0.9915** (tied 0.9915 by 6+gap and 6+peak×pose within run jitter; the
simpler, already-instrumented family wins the tie). Notable negative: adding
`margin` — the one extra feature the default path actually provides — LOWERS
held-out AUC (0.9915 → 0.9907). Buniatyan-style hypothesis-margin information
adds nothing beyond score/zncc here. The (b*) 7+n_hyp row's high total (75.76)
is F1-driven and not eligible (n_hyp unavailable) and still below (b)'s AUC.

## Result 5b — frozen constants refit on the ship-eligible winner (6 features)

`scripts/fit_calibration.py --freeze` (GD l2=1e-3, FULL 2,250, post-CV,
P(present) raw-scale conversion, self-check max |P_std−P_raw| = 3.9e-16).
These now live in `driftsense/calibration.py` (FEATURES and COEFS/INTERCEPT
updated; the old 9-feature constants were placeholders-by-then — the CV table
that justified them never made them ship-eligible).

| feature | coefficient (raw scale) |
|---|---:|
| score | +8.792353455411558 |
| zncc | +4.771002619826103 |
| peak_ratio | −0.22820720932034558 |
| pose_peak | −6.636234556838288 |
| psr | +0.001277399759015737 |
| apce | +3.981647734693036e-05 |
| intercept | −0.6718057933029007 |

Sign caveat as before: pose_peak's raw-space coefficient is negative although
pose_peak correlates positively with presence unconditionally — joint ranker
over correlated features (pose_peak ≈ duplicate of zncc); the shipped property
is the held-out CV AUC, not single-coefficient signs.

## Result 4b — threshold re-tune on the 6-feature frozen statistic (full 2,250, in-sample)

| operating point | threshold (calibrate units) | total | loc | F1 rej+ | F1 pres+ | AUC | declined-present | declined-absent | accepted-absent |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped scalar @ 0.18 | 0.18 | 75.35 | 0.8632 | 0.8648 | 0.9611 | 0.9876 | 71 | 435 | 65 |
| 6-feat fused, total-rubric optimum | **0.4870** | 75.71 | 0.8669 | 0.8786 | 0.9655 | 0.9916 | 59 | 438 | 62 |

(Both-convention F1 computed fresh; the shipped row's present-positive F1
0.9611 matches the earlier verification.) The optimum sits at the downward
end of the flat ridge, consistent with the downward-bias instruction. Held-out
expectation for the fused statistic remains the CV numbers: total 75.65,
AUC 0.9915 (row (b)) — the 75.71 here is in-sample.

## Result 7b — revised SHIP / DO-NOT-SHIP (integrator-constraint basis)

**RECOMMENDATION: SHIP the 6-feature calibration statistic
{score, zncc, peak_ratio, pose_peak, psr, apce} with the frozen constants in
`driftsense/calibration.py`; DO NOT SHIP margin or any derived feature; DO
NOT gate any code path on rank/band for this purpose.**

- Every candidate feature is computed on the SHIPPED default path
  (verification="zncc", no return_hypotheses). `calibrate()` consumes only the
  six values locate() already returns. Zero inference cost, no decode change,
  no new instrumentation.
- Held-out AUC 0.9878 → 0.9915 (+0.04 calibration points), 4-fold CV,
  threshold train-fold-only — measured fresh twice (original run and this
  addendum run), identical to 4 decimals.
- margin: measured out (0.9907 < 0.9915). Derived features: measured out
  (none beat 0.9915; combinations hurt). rank/band: unavailable on the
  default path, and even in reference trials their family (c) scores below
  (b) — no reason to enable a costlier verification mode for them.
- F1 unchanged (0.8778 → 0.8773): the confident-wrong-lock-on mass is not
  addressable from these features (oracle ceiling 0.8919 in-sample). The +4
  bonus remains out of reach; the calibration gain is the entire prize.
- Threshold: handoff 0.4870 in calibrate() units (in-sample total 75.71 vs
  scalar's 75.35; held-out expectation +0.11 total from row (b) vs row (a)
  CV). Integrator decides; the statistic itself is the AUC play.

---

# ADDENDUM 2 — shippable sets end-to-end (integrator request, `--feature-set`)

Integrator confirms: rank/band are unavailable at inference on the SHIPPED
default decode path, so the SHIPPABLE candidates are subsets of
{score, zncc, peak_ratio, pose_peak, psr, apce, margin} — i.e. "6" and "7m".
`scripts/fit_calibration.py --feature-set {6,7m,9}` now runs the SAME protocol
end-to-end per set: 4-fold CV (held-out total/F1/AUC, threshold
train-fold-only) → in-sample oracle F1 ceiling → threshold re-tune
(downward-biased, both conventions) → frozen raw-scale constants
(GD l2=1e-3, full 2,250, self-checked conversion). `driftsense/calibration.py`
is left as the 9-feature artefact; the integrator writes the shipped module
from the winning constants below.

## Shippable-sets table (all numbers fresh, same protocol, same 2,250 pairs)

| set | features | CV total | CV F1(rej) | **CV AUC** | oracle F1 ceiling | retuned t (calibrate units) | in-sample total @ t | F1 rej+ / pres+ @ t | declined-pres / declined-abs @ t |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped scalar | min(score,zncc) @ 0.18 | 75.54 | 0.8778 | 0.9877 | — | 0.18 (own units) | 75.35 | 0.8648 / 0.9611 | 71 / 435 |
| **6** | score, zncc, peak_ratio, pose_peak, psr, apce | 75.65 | 0.8773 | **0.9915** | 0.8827 | **0.4870** | 75.71 | 0.8786 / 0.9655 | 59 / 438 |
| 7m | 6 + margin | 75.57 | 0.8762 | 0.9907 | 0.8859 | 0.4121 | 75.68 | 0.8764 / 0.9664 | 41 / 422 |
| 9 (ref, not shippable) | 6 + rank, band, margin | 75.59 | 0.8734 | 0.9911 | 0.8919 | — | — | — | — |

Oracle-vs-held-out note (honest reading): 7m's in-sample oracle ceiling
(0.8859) EXCEEDS 6's (0.8827), yet its held-out AUC is LOWER (0.9907 vs
0.9915) and its held-out F1 is lower too (0.8762 vs 0.8773). That is the
in-sample oracle doing exactly what REJECTOR_FINDINGS.md Result 2 warns about
— extra features always look better when the fit may cheat; the held-out
columns are the decision columns. The 9-feature set shows the same pattern
(oracle 0.8919, held-out AUC 0.9911).

## Ranked SHIP recommendation among shippable sets

1. **SHIP: set "6"** — {score, zncc, peak_ratio, pose_peak, psr, apce}.
   Held-out AUC **0.9915** (vs shipped scalar 0.9877, +0.04 calibration pts),
   held-out total 75.65, F1 0.8773. Wins on held-out AUC outright (no
   tie-break needed) AND is the smaller set. Every member is computed on the
   default decode path (locate() lines 1164-1166); zero inference cost.
2. **7m: do not ship** — margin is available on every path but its CV AUC is
   0.9907 (−0.0008 vs 6) and CV F1 0.8762 (−0.0011); adding it costs a
   feature and generalises worse. Kept as measured-out, not opinion.
3. **9: not shippable on the default path** (rank/band need
   verification != "zncc" or return_hypotheses=True) and even as a reference
   row its held-out AUC (0.9911) sits below set 6.

### WINNER: set "6" — exact frozen constants (integrator: paste into shipped module)

Fit: `scripts/fit_calibration.py --feature-set 6 --freeze` — GD
(iters=4000, lr=0.5, l2=1e-3, standardised, unregularised intercept — the CV
protocol's optimiser) on the FULL 2,250-pair holdout, AFTER the CV/oracle/
threshold numbers above were recorded. P(present) = sigmoid(sum c_f·x_f + b);
conversion self-check max |P_std − P_raw| = 3.89e-16.

```
INTERCEPT = -0.6718057933029007
COEFS = {
    'score':       8.792353455411558,
    'zncc':        4.771002619826103,
    'peak_ratio':  -0.22820720932034558,
    'pose_peak':   -6.636234556838288,
    'psr':         0.001277399759015737,
    'apce':        3.981647734693036e-05,
}
```

Operating point (same downward-bias convention as Result 4/4b: lowest grid
threshold within 0.05 total of the optimum — for set 6 the bias rule selects
the optimum itself): **reject when calibrate(features) < 0.4870** (equivalently
−0.5130 on the −P(reject) scale; in-sample total 75.71 vs the shipped scalar's
75.35 on the same pairs; held-out expectation remains the CV row: 75.65 / AUC
0.9915). For completeness, set 7m's constants (not shipped) are in the
`--feature-set 7m` output: INTERCEPT −0.7863266314969231, score 10.6857…,
zncc 4.3977…, peak_ratio −0.2881…, pose_peak −6.4669…, psr 0.0014824…, apce
4.7737e-05, margin −2.6113…, threshold 0.4121.

Sign caveat (unchanged): pose_peak's raw-space coefficient is negative although
pose_peak correlates positively with presence unconditionally — a joint ranker
over correlated features (pose_peak ≈ zncc duplicate); the shipped property is
held-out AUC, not single-coefficient signs.

---

## PR #48 review addendum (2026-09-03): fused6 does not ship

### The measurement that decides it

One decode, both statistics recomputed offline from the `--features` columns, so
coordinates are **identical** and every delta belongs to the statistic. Scored
with `scripts/eval_ext.py` (the parity-pinned scorer), not a reimplementation.

| | fused6 @0.4870 | legacy_min @0.18 | delta |
|---|---:|---:|---:|
| FULL 2,500 (constants fitted here) | 78.09 | 77.91 | **+0.18** |
| HOLDOUT 500 (untouched, seed 20260902) | 76.41 | 76.84 | **−0.43** |

Paired bootstrap, 4,000 resamples, on (localisation + 15·F1) — the two
components that move; pose and calibration are near-identical:

| | delta | 95% CI | P(fused better) |
|---|---:|---|---:|
| full | +0.141 | [−0.105, +0.389] | 0.860 — not significant |
| **holdout** | **−0.443** | **[−0.947, +0.000]** | **0.011** |

Positive but not significant on the pool its constants were fitted on;
significantly negative on data it never saw. That is the overfitting signature.

### Three reasons this is worse than the raw −0.43

1. **It moves away from the +4 bonus.** Rejection F1 on untouched data goes
   0.8958 → 0.8663. The bonus gate is F1 ≥ 0.90, and 4 points dwarfs ±0.4.
2. **The AUC gain is nearly worthless in points.** legacy_min already scores
   **AUC 0.9882** on the full 2,500 against a 10-point component, so 0.9929 buys
   **+0.05 points**. The PR's headline "0.9689 → 0.9927" is a 500-pair
   subsample under the lenient present-only convention, not the ground-truth
   "AUC of the score column against per-pair correctness" that `eval_ext`
   reports separately (0.77–0.81).
3. **The threshold was never the problem.** Swept with the pinned scorer,
   0.4870 is near-optimal for fused6 on both sets (full best 78.12 @0.55 vs
   78.09; holdout best *is* 0.4870). The statistic is what fails to generalise.

**Shipped default reverted to `legacy_min` / 0.18.** The implementation,
constants and tests are retained: set `SHIPPED_CONFIDENCE = "fused6"` and
`SHIPPED_THRESHOLD = 0.4870` together to re-enable after a refit on the Set-C
feature distributions.

## Bonus analysis (2026-09-03)

### +6 (Set D ≥ 0.40 AND Sets A–C ≥ 0.50) — safe, no action

| gate | value | margin |
|---|---:|---:|
| Set D credit | **0.984** | +0.584 |
| min(A, B, C) | **0.848** | +0.348 |

### +4 (rejection F1 ≥ 0.90) — the marginal one, and 0.18 is the right bet

Expected score = localisation + 15·F1 + 4·P(F1 ≥ 0.90), where P is estimated by
the stratified draw the rubric actually grades on (A70/B70/C40, 3,000 draws):

| threshold | FULL expected | HOLDOUT loc / F1 |
|---|---:|---|
| 0.170 | 53.20 | 36.48 / 0.8958 |
| **0.180 (shipped)** | **53.14** | 36.48 / 0.8958 |
| 0.190 | 53.09 | 36.48 / 0.8958 |
| 0.200 | **53.31** | 36.48 / 0.8958 |
| 0.205 | 53.23 | 36.29 / 0.8866 |
| 0.210 | 52.62 | 36.29 / 0.8923 |

0.200 scores +0.17 higher on the fitted pool. **It is not taken**, because it is
a narrow peak 0.01 away from a −0.7 cliff at 0.210, while 0.180 sits on a flat
plateau (0.170/0.180/0.190 span 53.20–53.09). The external CPU benchmark at
`75c4572` is direct evidence the score distribution shifts substantially on
organizer-like data — correct matches scored 0.20–0.43 there against 0.83–0.95
on ours — so a threshold chosen on a narrow peak of *our* distribution is a bad
bet for 4 points. Robustness beats +0.17 of pool-fitted expected value here.

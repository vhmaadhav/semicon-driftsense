"""The shipped Phase 2 submission configuration -- ONE definition.

Every entry point that decodes or scores Phase 2 pairs must read these values
from here, never from a local literal, so the batch submission path
(register.py), the external evaluator (scripts/eval_ext.py) and the parity
tests cannot drift apart. The parity test (tests/test_submission_parity.py)
pins register.py and eval_ext.py against this module.

These are calibrated choices, not spec-derived constants:

* SHIPPED_THRESHOLD was swept against the *total* rubric on the external
  validation set (register.py history; F1-optimal thresholds sit too high
  because a declined present pair forfeits localisation + pose).
* LEGACY_FALLBACK_THRESHOLD gates the no-weights classical ZNCC path only
  (register.py, when the learned model fails to load -- issue #36). Its
  score is raw ZNCC, not the network-calibrated statistic SHIPPED_THRESHOLD
  was swept against, so reusing SHIPPED_THRESHOLD there is a unit mismatch:
  raw NCC on a periodic layout runs high even on wrong/absent matches. Set
  to 0.55, the docx spec's own naive-baseline reference calibration for
  exactly this coarse-NCC statistic (also this repo's
  generator/src/phase2_audit.py default). Deliberately conservative: this
  path only ever runs on a packaging/runtime failure, where a wrong
  confident answer costs far more (forfeits localisation + pose, and hurts
  rejection F1) than a correct decline.
* SHIPPED_BAND = False: the difference-of-Gaussians pre-filter on the coarse
  sweep costs points on both architectures (register.py measurement
  +0.439 / +0.509, PR #18 reached the same conclusion separately).
* SHIPPED_VERIFICATION = "zncc"
SHIPPED_SUBPIXEL_ROWS = True: the native-ZNCC winner; consensus/majority
  are measured research selectors, not shipped ones.
* SHIPPED_SUBPIXEL_ROWS = True: re-place x on the scan row the label is
  defined against, recovering the centre row's raster-drift sample
  (driftsense.matching.drift_row_refine). Full 2,500-pair paired A/B:
  localisation 35.46 -> 36.05, total 76.94 -> 77.53, paired delta +0.589
  with 95% CI [+0.410, +0.773] and P(delta >= +0.35) = 0.994, so it clears
  the issue #19 promotion gate. Costs 1.9 ms median per pair. Pose,
  rejection and calibration are bit-identical -- the correction moves only x.
  Evidence: .agents/SUBPIXEL_DRIFT.md. Set to False to revert entirely.
* SHIPPED_LABEL_CONVENTION = "center": the pixel convention the grader's
  (x, y) labels are written in. It is a property of the dataset, not of the
  model -- see the block above the constant (issue #86).
"""

from __future__ import annotations

# Confidence statistic shipped in the `score` column (ONE definition).
#   "fused6": 6-feature logistic over outputs locate() already computes
#             (score, zncc, peak_ratio, pose_peak, psr, apce) via
#             driftsense.calibration.calibrate(). Frozen constants were fit
#             offline on the 2,250-pair holdout AFTER 4-fold CV
#             (.agents/B_CALIBRATION_REPORT.md); held-out AUC 0.9877 ->
#             0.9915. Zero inference cost, no decode change.
#   "legacy_min": the historical min(network score, native ZNCC).
#   "min_med3": min(network score, ZNCC on a 3x3-median copy of the search
#             frame at the final pose and the rigid answer) -- issue #87. The
#             median touches this one number only; network and localisation
#             keep the raw frame. See the block above SHIPPED_THRESHOLD.
# The parity test pins register.py and eval_ext.py to this module's values.
# --------------------------------------------------------------------------
# Uncontested-hypothesis early exit (PR #51).
#
# pose_candidates returns hypotheses already ranked by coarse peak, and
# choose() takes the highest native ZNCC, so a first hypothesis that verifies
# strongly enough cannot be beaten by the ones behind it. The network is ~86%
# of a pair and is paid once per hypothesis, so skipping the rest is close to
# a 3x saving on the pairs that qualify.
#
# The gates live HERE, not as literals in matching.py, because the PR that
# introduced them documented one rule (0.88 / 0.55 / 0.30) and implemented
# another -- exactly the drift a single definition prevents. Each gate is
# (min network score, min native ZNCC, max peak_ratio, min coarse-peak gap to
# the runner-up); None means that term is not tested. A gate fires only on the
# FIRST hypothesis, and only when a second hypothesis exists.
#
# Both gates are validated in .agents/PR51_CAMPAIGN.md against a full
# no-early-exit decode of the same pairs; tests/test_early_exit_gates.py pins
# these numbers so the documentation and the code cannot drift apart again.
EARLY_EXIT_GATES = (
    (0.85, 0.75, 0.25, None),   # uncontested: no rival peak worth checking
    (0.72, 0.72, 0.35, 0.04),   # clear coarse lead over the runner-up
)

SHIPPED_CONFIDENCE = "min_med3"

# Found threshold, in the units of whichever SHIPPED_CONFIDENCE is active.
# The statistic and its threshold are ONE unit system -- change both or
# neither (tests/test_submission_parity.py pins the coupling, not the value).
#
# Current: SHIPPED_CONFIDENCE="min_med3" gated at 0.55 (issue #87), chosen on
# the v2 dev split only (scripts/gen_phase2_v2_val.py, 500 pairs, seed
# 850001) and confirmed on untouched splits -- see below.
#
# Why the statistic changed: on the dev split the historical legacy_min cannot
# separate present from absent at any threshold (absent max 0.529, present
# min 0.370; best total 82.47 in a narrow band, and 72.00 at 0.55 because 65
# degraded present pairs fall below it). The median-ZNCC term separates it
# (present min 0.592, absent max 0.529) and holds 82.41-82.80 for every
# threshold from 0.35 to 0.60. Ablation: moving legacy_min's ZNCC to the final
# pose changes nothing; the median is the whole effect.
#
# Why 0.55: the rule was fixed before any held-out split was scored -- inside
# the empty band between the dev absent max (0.529) and present min (0.592),
# shifted toward the absent side by the cost ratio (a declined present pair
# also forfeits localisation and pose, ~2.7x an accepted absent pair):
# 0.529 + 0.063 / 3.7 = 0.546 -> 0.55. That it equals
# LEGACY_FALLBACK_THRESHOLD below is a coincidence of two separate
# calibrations in two unit systems, not a shared value.
#
# Confirmed on data not used for the choice (paired against legacy_min@0.18,
# same decode otherwise; localisation, scale and rotation bit-identical):
#   v2 holdout (500, seed 850002)  80.23 -> 82.39, +2.16 [+1.40, +3.09];
#                                  absent accepted 25 -> 0, present declined 0.
#                                  present min 0.571, absent max 0.493; total
#                                  82.20-82.39 for any threshold in [0.45, 0.60].
#   mentor 25-pair v2 set          81.73 -> 83.40 (absent accepted 1 -> 0).
#   fresh 48-pair v2 set (s777)    79.34 -> 81.99 (absent accepted 3 -> 0).
#   generator/output (original generator, pixel-edge labels): 82.75 -> 82.75.
# Runtime unchanged (median 2.11/2.12 s -> 2.06/2.13 s per pair, 4 threads).
#
# Previous pairings, kept consistent if ever restored: "legacy_min" -> 0.18
# (swept on the original Phase 2 distribution); "fused6" -> 0.4870 (a
# calibrated P(present), re-tuned on the 2,250 holdout against the total
# rubric -- .agents/B_CALIBRATION_REPORT.md Result 4b).
SHIPPED_THRESHOLD = 0.55
# The no-weights ZNCC fallback in register.py scores a raw NCC, which is
# neither unit system above, so it carries its own gate. Raised 0.18 -> 0.55 on
# origin/phase2 (#54, issue #36) when the fallback stopped being a silent
# substitution: it now fails closed unless --allow-fallback is passed, and its
# gate is calibrated for raw NCC rather than inherited from the learned path.
LEGACY_FALLBACK_THRESHOLD = 0.55

# 2026-09-03, PR #48 review: fused6 was measured against legacy_min on ONE
# decode (features recorded with --features, both statistics recomputed
# offline, so coordinates are identical and every delta is the statistic's).
#
#                                fused6@0.4870   legacy@0.18    delta
#   FULL 2,500 (fitted here)        78.09           77.91       +0.18
#   HOLDOUT 500 (untouched)         76.41           76.84       -0.43
#
# Paired bootstrap on (localisation + 15*F1), 4,000 resamples:
#   full     delta +0.141  95% CI [-0.105, +0.389]  P(fused better) 0.860
#   holdout  delta -0.443  95% CI [-0.947, +0.000]  P(fused better) 0.011
#
# Positive but not significant on the pool its constants were fitted on;
# significantly NEGATIVE on data it never saw. That is the overfitting
# signature, and it is driven by rejection F1 (holdout 0.8958 -> 0.8663),
# which is also the metric carrying the +4 bonus at F1 >= 0.90 -- fused6 moves
# AWAY from that gate on untouched data.
#
# The calibration AUC gain that motivated fused6 is real but nearly worthless
# in points: legacy already scores AUC 0.9882 on the full set against a
# 10-point component, so 0.9929 buys +0.05 points.
#
# The fused6 implementation, constants and tests are retained; set
# SHIPPED_CONFIDENCE = "fused6" and SHIPPED_THRESHOLD = 0.4870 together (they
# are ONE unit system) to re-enable after a refit on the Set-C feature
# distributions.

SHIPPED_BAND = False
SHIPPED_VERIFICATION = "zncc"
SHIPPED_SUBPIXEL_ROWS = True

# Refine rotation from the vertical offsets of vertical template strips, and
# blend that with polish_pose's answer (driftsense.matching.strip_rotation,
# issue #88). ONE definition; register.py passes it to locate_phase2
# (strip_rot=...), and tests/test_submission_parity.py pins that.
#
# Why a second estimator at all: polish_pose fits rotation with a 2-D ZNCC,
# and on a raster-scanned frame one of those two dimensions is corrupted. A
# rotation error displaces template point (u, v) by (d*v, -d*u); the
# horizontal half varies along the row axis, which is exactly the axis raster
# drift acts on, so the v2 extension's 1-3 px shear alone impersonates
# 0.06-0.17 deg of rotation. The vertical half varies along the column axis,
# where drift, shear, scale error and barrel distortion contribute nothing.
# Started AT the ground-truth pose, polish_pose still walks ~0.2 deg away on
# degraded v2 frames, so the objective is biased, not merely under-searched.
#
# Why a blend and not a replacement: the strip regression is the noisier of
# the two whenever the strips are poorly textured, so the two are combined by
# inverse variance, with the regression's own standard error against a fixed
# prior for polish_pose (STRIP_ROT_SIGMA_PRIOR in matching.py). Measured on
# the v2 dev split (400 present pairs), rotation credit: polish alone 0.895,
# blend 0.958, unblended strip estimate 0.925 -- i.e. adopting the regression
# whole gives back half the gain. The prior is flat from 0.10 to 0.22
# (credit 0.955-0.961); 0.15 is the middle of that plateau.
#
# Dev split, end to end: 82.80 -> 83.41, paired +0.61 95% CI [+0.43, +0.80],
# P(delta >= +0.35) = 0.998; rotation 8.95 -> 9.58 / 10 with every severity
# bucket improving (sev 0: 0.912 -> 1.000, sev 4: 0.817 -> 0.858).
#
# Every knob (strips, lag, iterations, peak floor, prior) was chosen on the
# v2 dev split alone -- 13 configurations x 6 priors, on a surface that is
# flat around the chosen point (driftsense.matching, STRIP_ROT_* block).
#
# Confirmed on data not used for that choice (paired, same decode otherwise;
# scale, rejection and AUC are bit-identical on every set):
#   v2 holdout (500, seed 850002)  82.39 -> 83.18, +0.79 [+0.61, +1.00];
#                                  rotation 8.79 -> 9.54, localisation
#                                  38.81 -> 38.86.
#   v2 stress split (250, severity 3-4 heavy)
#                                  82.53 -> 83.15, +0.62 [+0.31, +0.94];
#                                  rotation 8.76 -> 9.53.
#   mentor 25-pair v2 set          83.40 -> 83.58; rotation 8.89 -> 9.56, and
#                                  one Set B pair already sitting on the 1 px
#                                  tier boundary (0.984 px) crossed it at
#                                  1.064 px -- the stage moves x only through
#                                  the template the drift-row stage builds.
#   generator/output (the ORIGINAL generator, pixel-edge labels, no raster
#                                  shear in its severity ladder): 82.75 ->
#                                  82.96, rotation 9.21 -> 9.43. Unlike #86,
#                                  this is not a v2-only correction.
# Cost: +3 ms median per pair (20 pairs, interleaved on/off in one process at
# 4 threads: 2.270 s -> 2.271 s median).
SHIPPED_STRIP_ROTATION = True

# Pixel convention of the grader's (x, y) labels (issue #86). ONE definition;
# register.py passes it to locate_phase2 (label_convention=...), and
# tests/test_submission_parity.py pins that.
#   "edge":   pixel i spans [i, i+1), so a template placed at top-left p has
#             its centre at p + tw/2. Our own generator (driftsense.generate:
#             area_convention_offset), our training labels and the original
#             Phase 2 generator (generator/src/pipeline.py: gt_x0 + box_w/2)
#             are written this way, so locate_phase2 keeps "edge" as its
#             signature default and every internal evaluator on that data
#             (engine.evaluate, scripts/eval_ext.py) stays correct as it is.
#   "center": pixel i spans [i-0.5, i+0.5] -- OpenCV warpAffine coordinates.
#             The mentor's Phase 2 v2 (extension) generator labels
#             M @ (x0 + 499.5, y0 + 499.5) with the canvas centre (N-1)/2
#             mapped to (1000-1)/2, and its bundled baseline reports
#             loc + (tw-1)/2. Same point, reported 0.5 px up-left.
# The convention also decides WHICH scan row a label's raster-drift sample is
# read from: v2 reads row_shift[round(y_center)], our generator
# row_shift[round(y_edge)] -- a different row on about half of all pairs.
#
# Measured (A/B/C, threshold 0.18; scale, rotation, rejection and AUC are
# bit-identical between the two settings -- only x, y move):
#   mentor 25-pair v2 set:   "edge" 76.71 -> "center" 81.73/85 (loc A 0.911 ->
#                            1.000, B 0.822 -> 0.978); paired +5.02, 95% CI
#                            [+2.67, +7.69].
#   fresh 48-pair v2 set (the mentor's generate_phase2_dataset_v2.py,
#                            --seed 777, seed-disjoint): 77.03 -> 79.34;
#                            paired +2.31, 95% CI [+1.16, +3.57]; mean y
#                            error +0.54 -> +0.04 px.
#   generator/output (pixel-edge labels): "edge" 82.75, "center" 79.93 -- the
#                            same half pixel costs points the other way, which
#                            is why this is a flag and not a new default.
SHIPPED_LABEL_CONVENTION = "center"

# Sub-pixel placement rule for the final ZNCC snap (ONE definition; applied
# at the refine_zncc site in matching.py).
#   "parabola" (SHIPPED): the historical 1-D parabolic fit through the peak.
#   "bicubic": bicubic upsampling of the correlation surface around the peak
#             (driftsense.subpixel.refine_bicubic; Debella-Gilo & Kaab 2011,
#             DOI 10.1016/j.rse.2010.08.012).
# MEASURED 2026-09-03, bicubic NOT shipped -- gate (c) failed on the 60-pair
# holdout draw (RandomState(200), A/B shards): net loc credit +0.20 with
# 1 rescue / 0 breaks, but p95 coordinate shift 0.271 px against the 0.15 px
# stability gate, including one 2.62 px jump (tier-neutral by luck). On the
# official-20 it rescued both Set D boundary pairs (p019 1.004 -> 0.898,
# p020 1.077 -> 0.993; +0.40 credit, 0 breaks) but does NOT rescue Set B's
# p014 -- its ~1.03-1.10 px error is upstream of sub-pixel refinement -- so
# it cannot break the 39.27 localisation tie. Synthetic accuracy tests are
# mixed. Revisit only with the full 2,250-pair paired bootstrap
# (.agents/C_LOCALIZATION_REPORT.md, .agents/integrator_ext60_tmp.py output).
SHIPPED_SUBPIXEL = "parabola"


# ---------------------------------------------------------------------------
# Phase 3 decode (phase3.py). The CAD reference is rendered by driftsense.gds;
# everything after that is locate_phase2 with the settings below. Measured on
# seed-disjoint splits of generator_i4c/generate_cad_varied.py -- see
# docs/PHASE3_MEASUREMENT.md for each number.
# ---------------------------------------------------------------------------

# The CAD generator caps search rotation at 10 deg (MAX_SEARCH_ROTATION_DEG);
# its GUI randomises the magnitude bound in [0, 8]. The Phase 2 box (+/-5)
# would clip every pair beyond it.
PHASE3_ROTATION_BOUNDS = (-10.0, 10.0)
# Keeps the Phase 2 grid spacing (1 deg) over the wider box.
PHASE3_COARSE_ROTATIONS = 21
# The CAD generator labels the centre of the design window divided by the
# 10x scale factor: pixel-edge coordinates, like our own generator.
PHASE3_LABEL_CONVENTION = "edge"
# found = confidence >= this. Starts at the Phase 2 value; recalibrated for
# Phase 3 below once measured.
PHASE3_THRESHOLD = SHIPPED_THRESHOLD
# Image-only fallback (no usable search CAD). The CAD generator labels the
# undrifted position, so for Phase 3:
#   * the drift-row stage is off -- it re-places x on the centre row's jitter,
#     which the Phase 2 label carried and the Phase 3 label does not;
#   * the expected raster shear (search px) is added back to x: shear moves
#     row y left by shear * y / (h - 1). 1.5 is the upstream CLI default
#     (the GUI's Randomize band averages 1.25; any prior from 1.0 to 1.75
#     scored within 0.05 of each other).
# Measured (rubric /85; within-1-px of found present pairs):
#   CLI 400, seed 5:  58.25 -> 60.69  (52% -> 85%)
#   CLI 400, seed 9:  59.09 -> 61.74  (52% -> 87%)
#   organizer's 20 curated samples, image-only: localisation 32.40 -> 35.20 /40
PHASE3_FALLBACK_SHEAR_PX = 1.5
PHASE3_SUBPIXEL_ROWS = False

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
* SHIPPED_VERIFICATION = "zncc": the native-ZNCC winner; consensus/majority
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

# Variance-stabilising transform applied to intensities before correlation
# (issue #13; driftsense/vst.py). ZNCC is the ML statistic under additive
# homoscedastic Gaussian noise, while SEM shot noise is Poisson; a VST makes
# the assumption true instead of assuming it.
#   "none" (SHIPPED): the historical path, no transform.
#   "anscombe": 2*sqrt(x + 3/8), parameter-free (Anscombe 1948).
#   "gat": generalised Anscombe with (a, b) fitted per image from the local
#          mean/variance envelope (Foi et al. 2008).
# Per-stage overrides for experiments: DRIFTSENSE_VST_{COARSE,VERIFY,REFINE},
# then DRIFTSENSE_VST, then this default.
#
# MEASURED 2026-09-08, NOT SHIPPED. 480 present Set B pairs across the full
# severity ladder, paired, coarse stage, band=False (the shipped decode):
# correct-candidate generation 87.7% (none) vs 88.5% (anscombe and gat),
# i.e. rescued 14 / broke 10, net +4 of 480, McNemar exact p=0.54. No subset
# rescues it. ZNCC is invariant to the affine part of the transform, which is
# where the fitted gain lives, so the noise model cannot reach a normalised
# correlation score even in principle -- see the MEASURED block in
# driftsense/vst.py for the derivation and the full tables.
SHIPPED_VST = "none"

# Epsilon for the "dog-override" selector in matching.locate_phase2. A research
# knob, NOT a shipped one: the default selector stays SHIPPED_VERIFICATION =
# "zncc" and no default decode reads this value.
#
# scripts/verify_scores.py measured `zncc_dog` as the best alternative selector
# on Set B -- net +10 recovered pairs against the incumbent ZNCC's +7 -- and it
# was never wired into inference. Swapping the selector wholesale is the wrong
# trade: DoG wins only on the *contested* decisions, and on the ~87% of pairs
# ZNCC already gets right it can only break things (the same asymmetry that
# sank the rescue pass and the band pre-filter above). So the override leaves
# ZNCC owning the decision and lets DoG take it only when DoG disagrees AND the
# ZNCC margin between the two candidates is below this value -- i.e. only where
# ZNCC has effectively abstained. 0.05 is a deliberately narrow starting point
# chosen to reach the ties and nothing else; it has not been swept, so treat it
# as an experiment parameter rather than a measured constant.
DOG_OVERRIDE_MARGIN = 0.05

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


# --------------------------------------------------------------------------
# Phase 3 found-threshold.
#
# SHIPPED_THRESHOLD is a Phase 2 constant and must stay one: it was swept
# against the Phase 2 rubric on the Phase 2 distribution, where ~20% of pairs
# are absent. Phase 3 discloses ~1 site in 12 (8.3%), and that alone moves the
# optimum down, mechanically rather than as a matter of taste:
#
#   * a false ACCEPT costs only rejection F1, and with 8.3% absent there are
#     few pairs that can be falsely accepted at all;
#   * a false DECLINE costs the pair's localisation (40) AND pose (20) -- the
#     row is zero-filled, which is what gets submitted -- and costs F1 too.
#
# Fitted on 250 generated Phase 3 pairs at rotation U(-10,+10) (226 present /
# 24 absent), decoded once at --threshold 0 and re-thresholded offline, scored
# over the briefing's 85 measurable points:
#
#     T      0.00   0.15   0.20   0.25   0.40   0.55(P2)  0.70
#     /85   67.58  66.84  66.84  66.74  65.20   59.37    44.24
#
# Validated on a disjoint 60-pair set never used to choose it: 0.55 scores
# 56.69, 0.20 scores 65.91, T=0 scores 68.14. Phase 2's 0.55 declines 48 of
# 226 present pairs on the fit set and gives up ~8 points.
#
# WHY NOT 0.0, which is the literal argmax -- and is the argmax at EVERY
# assumed absent rate from 8% to 40%, so this is not an artefact of the
# generated absent fraction. Two reasons, both about what is not measured
# here rather than about taste:
#
#   * the briefing says high-confidence false grabs "carry heavy penalties",
#     which a plain F1 term does not express; and
#   * these absent pairs come from our own generator. The organizers' own
#     README discloses that their absent decoys carry a size signature, so
#     their absents may be more separable than ours, and a degenerate `found`
#     column cannot exploit that at all.
#
# 0.20 is the best NON-degenerate point at every assumed absent rate and costs
# 0.74 against T=0 at the disclosed rate. It still declines 12% of absent
# pairs, so the rejector remains a working component rather than a constant.
#
# THE REAL FINDING IS NOT THE THRESHOLD. At 0.20, 88% of absent pairs still
# score above it against 97% of present ones -- the shipped `legacy_min`
# confidence barely separates the two classes on CAD-reference pairs (AUC
# 0.836 here against 0.9877 on Phase 2 data). No threshold can fix a
# statistic that does not separate. docs/PHASE3_MEASUREMENT.md already
# measured the replacement: the confidence MARGIN (best peak minus best
# competing peak) separates good pose basins from bad at AUC 0.948. Wiring
# that into the score column is where the 15 rejection and 10 calibration
# points actually are -- this constant only stops the current statistic from
# throwing away localisation and pose on top of them.
PHASE3_THRESHOLD = 0.20


# --------------------------------------------------------------------------
# Phase 3 label convention.
#
# Phase 2 ships "center" -- established empirically against the organizers'
# Phase 2 ground truth (issue #86), and confirmed on their 25-pair set where it
# drives the signed dy bias to -0.014 px.
#
# Phase 3 is "edge", and this is derived from the generator's source rather
# than fitted. src/cad_pipeline.py:227-229 defines the answer as
#
#     gt_x0, gt_y0 = x0 / SCALE_FACTOR, y0 / SCALE_FACTOR   # crop top-left
#     gt_cx, gt_cy = gt_x0 + box_w / 2.0, gt_y0 + box_h / 2.0
#
# i.e. "centre = top-left corner + width/2", which is exactly the pixel-edge
# convention and exactly the rule the matcher uses internally. "center" then
# subtracts a further 0.5 px from a coordinate that was already correct.
#
# Measured on 60 generated pairs at rotation U(-10,+10): signed dy median
# -0.5145 under "center" against -0.0145 under "edge", localisation 26.42 ->
# 30.94 of 40, pairs inside 1 px 9/53 -> 25/53, subtotal 66.43 -> 71.31 of 85.
#
# The two phases genuinely disagreeing is worth confirming with the organizers
# -- it is one flag either way -- but the Phase 3 evaluation data is produced
# by this same cad_pipeline, so its convention is the one that applies.
PHASE3_LABEL_CONVENTION = "edge"

# Phase 2's drift-row x refinement. Kept ON: the hypothesis that it was the
# source of the residual x bias was tested and is WRONG -- disabling it moves
# the signed dx median from -0.8271 to -0.9156, i.e. slightly worse. It is
# recovering a little of the raster drift, just not much of it.
#
# The residual dx of about -0.83 px is NOT a defect in the matcher. It is the
# generator's raster drift (src/sem_imaging.py:124-130): rows are sheared by
# shear_amplitude_px * row/(h-1) and cv2.remap SAMPLES at x+s, so imaged
# content sits -s from where the ground truth -- computed on the pre-drift
# geometry -- says it is. Fitting dx against y/999 recovers slope -1.2258
# against the model's -1.50, with the per-quartile medians tracking it.
#
# A correction now exists: driftsense.drift_shear, behind phase3.py's
# --shear-correct, OFF by default. Issue #101, docs/PHASE3_RASTER_SHEAR.md.
# Three things in the paragraph above needed correcting once it was measured on
# a PAIRED amplitude sweep (scripts/gen_phase3_rotated.py --paired):
#
#   * the residual is larger than -1.2258 at A = 1.5 suggested. Fitted across
#     A in {0,1,2,3,4}, the slope is -1.0394*A - 0.271: the decode passes
#     essentially 104% of the amplitude through, and the cost runs from 0.26 px
#     median at A = 0 to 2.90 px at A = 4 (<=1px 95% -> 19%).
#   * the GLOBAL edge-family measurement suggested here was built and is
#     MEASURED OUT. Both frame-wide variants have unit gain in A on top of a
#     per-pair offset of sd ~3 px that survives with zero noise, zero jitter
#     and zero drift -- a global shear maps a lattice to a lattice, so one
#     frame cannot separate it from the design's own obliquity. That is why
#     the scan-distortion literature uses two acquisitions.
#   * per-pair estimation really is out of reach, but for a different reason
#     than "ZNCC is flat in it": the reference-driven row-offset slope IS
#     identifiable and unbiased, it is just worth 4.47 px per pair against a
#     0-4 px signal. Pooled over a batch it is worth 0.79 px, which is why the
#     shipped correction is per-batch and why enabling it couples the pairs of
#     one run.
PHASE3_SUBPIXEL_ROWS = True


# --------------------------------------------------------------------------
# Phase 3 confidence statistic.
#
# SHIPPED_CONFIDENCE is "legacy_min" -- min(network score, native ZNCC) -- and
# stays that for Phase 2, where it is what the 0.9877 holdout AUC was measured
# on. It is the wrong choice for Phase 3, and the reason is a domain shift, not
# a tuning accident: the network was trained on SEM-against-SEM pairs, and a
# Phase 3 reference is a RENDERED DESIGN. The network score is therefore the
# weaker of the two signals here, and taking the min drags the stronger one
# down to it.
#
# Measured on 60 generated pairs at rotation U(-10,+10), scoring the rubric's
# calibration block (AUC of the score column, correct-and-within-5px against
# everything else):
#
#     zncc          0.9200      pose_peak    0.8844      score       0.8504
#     legacy_min    0.8370      peak_ratio   0.7630      psr         0.5748
#     score*zncc    0.9052      apce         0.5615      hyp_margin  0.5222
#
# VALIDATED on a disjoint 250-pair set, and the gain shrinks -- this is what
# the tuning set's optimism looks like when it is checked:
#
#                     60-pair (tuned)   250-pair (validation)
#     legacy_min          0.8370              0.8358
#     zncc                0.9200              0.8647
#     gain                +0.83               +0.29
#     pose_peak           0.8844              0.6719   <- genuinely overfit
#
# Adopted at +0.29 of 85, not the +0.83 the small set suggested. It is kept
# because it is positive on both sets, costs nothing, and localisation, pose
# and rejection come out bit-identical -- only the score column moves.
# pose_peak is the cautionary tale: best-looking alternative on 60 pairs,
# collapses to near-useless on 250.
#
# Recorded because it corrects a claim in docs/PHASE3_MEASUREMENT.md as it is
# easy to misread: the confidence MARGIN measured AUC 0.948 there for
# predicting CATASTROPHIC LOCALISATION FAILURE (>20 px against <=2 px) under
# the classical matcher. That is a different question from the rubric's
# calibration block, and on this question the margin is worthless -- AUC
# 0.5222, indistinguishable from chance. Do not carry the 0.948 across.
PHASE3_CONFIDENCE = "zncc"

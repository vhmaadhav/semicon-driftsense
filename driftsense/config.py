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
# The parity test pins register.py and eval_ext.py to this module's values.
SHIPPED_CONFIDENCE = "legacy_min"

# Found threshold, in the units of whichever SHIPPED_CONFIDENCE is active.
# The statistic and its threshold are ONE unit system -- change both or
# neither (tests/test_submission_parity.py pins the coupling, not the value).
#
# Current: SHIPPED_CONFIDENCE="legacy_min", so 0.18 gates min(net, zncc). It
# is the shipped threshold on the shipped learned path, not a fallback value.
# Kept at 0.18 rather than the pool-optimal 0.200 deliberately: 0.200 is a
# narrow peak sitting 0.01 from a -0.7 cliff at 0.210 on our distribution,
# while 0.180 sits on a flat plateau, and the external CPU benchmark at
# 75c4572 shows the score distribution shifts substantially on organizer-like
# data. See docs/CAMPAIGN_2026-09-03_REPORT.md.
#
# If SHIPPED_CONFIDENCE is ever set back to "fused6", this must move to 0.4870
# at the same time -- there the score column is a calibrated P(present) and
# 0.18 in those units decides nothing (re-tuned on the 2,250 holdout against
# the total rubric with the downward bias convention -- declined present pairs
# forfeit localisation+pose -- see .agents/B_CALIBRATION_REPORT.md Result 4b).
SHIPPED_THRESHOLD = 0.18

# The no-weights ZNCC fallback in register.py scores a raw NCC from a single
# template sweep, which is neither statistic above, so it carries its own gate.
# It equals SHIPPED_THRESHOLD today only because the shipped statistic is
# min(net, zncc) and inherits the same historical tuning; they are separate
# constants so that changing one cannot silently move the other.
LEGACY_FALLBACK_THRESHOLD = 0.18

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

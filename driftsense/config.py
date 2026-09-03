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

SHIPPED_CONFIDENCE = "legacy_min"

# Found threshold, in the units of whichever SHIPPED_CONFIDENCE is active.
# The statistic and its threshold are ONE unit system -- change both or
# neither (tests/test_submission_parity.py pins the coupling, not the value).
#
# Current: SHIPPED_CONFIDENCE="legacy_min", so 0.18 gates min(net, zncc) on the
# shipped learned path. It is the shipped threshold, NOT a fallback value --
# the fallback has its own gate below.
#
# If SHIPPED_CONFIDENCE is ever set back to "fused6", this must move to 0.4870
# at the same time: there the score column is a calibrated P(present) and 0.18
# in those units decides nothing (re-tuned on the 2,250 holdout against the
# total rubric with the downward bias convention -- declined present pairs
# forfeit localisation+pose -- see .agents/B_CALIBRATION_REPORT.md Result 4b).
SHIPPED_THRESHOLD = 0.18
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

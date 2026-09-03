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

SHIPPED_THRESHOLD = 0.18
LEGACY_FALLBACK_THRESHOLD = 0.55
SHIPPED_BAND = False
SHIPPED_VERIFICATION = "zncc"
SHIPPED_SUBPIXEL_ROWS = True

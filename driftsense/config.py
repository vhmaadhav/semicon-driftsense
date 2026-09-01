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
* SHIPPED_VERIFICATION = "zncc": the native-ZNCC winner; consensus/majority
  are measured research selectors, not shipped ones.
"""

from __future__ import annotations

SHIPPED_THRESHOLD = 0.18
SHIPPED_BAND = False
SHIPPED_VERIFICATION = "zncc"

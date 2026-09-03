"""
Pattern-collapse / bridging.

This models a real *structural* defect (high-aspect-ratio lines toppling and
sticking to their neighbor, e.g. resist/fin collapse from capillary forces)
-- it is applied once to the fine canvas and therefore shows up consistently
in both the reference crop and the derived search image. It is distinct from
SEM *imaging* artifacts (blur/noise/drift), which live in sem_imaging.py and
are applied per-image since reference and search are captured differently.

Default collapse_threshold_nm=10 is tied to the search image's 10 nm/px
resolution: a 10 nm gap is exactly 1 px there, i.e. right at the edge of what
that image can physically resolve as two separate structures.
"""

import numpy as np


def maybe_collapse_gap(
    gap_nm: float,
    threshold_nm: float,
    rng: np.random.Generator,
    collapse_prob: float = 0.7,
) -> bool:
    """Decide whether a gap between two adjacent lines should bridge/merge.

    Gaps at or above threshold never collapse. Gaps below threshold collapse
    with `collapse_prob`, so the effect is visible but not deterministic
    (mirrors real process variation).
    """
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)

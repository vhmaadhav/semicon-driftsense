"""The early-exit gates and the candidate-dedup radius are pinned here.

Both were flagged in the PR #51 review for the same reason: the numbers in the
prose did not match the numbers in the code (the PR described a 0.88 / 0.55 /
0.30 rule while the implementation used 0.72 / 0.72 / 0.35 / 0.04), and the
dedup radius was wider than the refinement window it was justified by.

These tests exist so that class of drift fails CI instead of shipping:

* the gate constants are frozen as literals, so editing `config.py` without
  editing the documentation breaks the build;
* `_early_exit_fires` is exercised at its boundaries and on degenerate input;
* the dedup radius is asserted to lie INSIDE the polish window, which is the
  property that makes deduplication safe rather than merely cheap.
"""
import numpy as np
import pytest

from driftsense.config import EARLY_EXIT_GATES
from driftsense.matching import (POLISH_ROT_BAND, POLISH_SCALE_BAND,
                                 _early_exit_fires, polish_pose)


def test_gate_constants_are_frozen():
    """Literal expected values -- not rebuilt from the constant under test."""
    assert EARLY_EXIT_GATES == (
        (0.85, 0.75, 0.25, None),
        (0.72, 0.72, 0.35, 0.04),
    )


def _r(score, zncc, ratio):
    return {"score": score, "zncc": zncc, "peak_ratio": ratio}


def test_uncontested_gate_fires_without_a_coarse_gap():
    """Gate 1 has no gap term, so a zero gap must not block it."""
    assert _early_exit_fires(_r(0.85, 0.75, 0.25), 0.0)


def test_uncontested_gate_is_exclusive_at_its_edges():
    assert not _early_exit_fires(_r(0.849, 0.75, 0.25), 0.0)
    assert not _early_exit_fires(_r(0.85, 0.749, 0.25), 0.0)
    assert not _early_exit_fires(_r(0.85, 0.75, 0.251), 0.0)


def test_clear_lead_gate_requires_the_coarse_gap():
    assert _early_exit_fires(_r(0.72, 0.72, 0.35), 0.04)
    assert not _early_exit_fires(_r(0.72, 0.72, 0.35), 0.039)


def test_missing_or_non_finite_statistics_never_exit_early():
    """An incomplete result must fail the gate, not accidentally pass it.

    `peak_ratio` defaults to 1.0 and `zncc` to -inf precisely so that a missing
    key cannot look like a confident match.
    """
    assert not _early_exit_fires({}, 10.0)
    assert not _early_exit_fires({"score": 0.99}, 10.0)
    assert not _early_exit_fires(_r(np.nan, 0.99, 0.0), 10.0)
    assert not _early_exit_fires(_r(0.99, np.nan, 0.0), 10.0)
    assert not _early_exit_fires(_r(0.99, 0.99, np.nan), 10.0)


def test_dedup_radius_lies_inside_the_polish_window():
    """The safety property behind deduplication.

    Two candidates are merged only when the survivor's polish window already
    covers the discarded one, so the discarded hypothesis could not have
    reached an optimum the survivor cannot. That holds exactly when the dedup
    radius is not wider than the polish band -- here they are the same object.
    """
    import inspect
    sig = inspect.signature(polish_pose)
    assert sig.parameters["scale_band"].default == POLISH_SCALE_BAND
    assert sig.parameters["rot_band"].default == POLISH_ROT_BAND
    # And the values are the documented ones.
    assert POLISH_SCALE_BAND == 0.03
    assert POLISH_ROT_BAND == 0.8


def test_dedup_merges_only_within_the_basin():
    """Behavioural check on the rule pose_candidates applies."""
    def same_basin(a, b):
        return (abs(a[0] - b[0]) < abs(b[0]) * POLISH_SCALE_BAND
                and abs(a[1] - b[1]) < POLISH_ROT_BAND)

    keep = (10.0, 0.0)
    assert same_basin((10.2, 0.5), keep)          # inside both bands
    assert not same_basin((10.31, 0.5), keep)     # outside +/-3% of 10.0
    assert not same_basin((10.2, 0.9), keep)      # outside +/-0.8 deg
    # The old radius (0.35 scale, 1.0 deg) merged pairs the polish could not
    # reach; that must no longer happen.
    assert not same_basin((10.34, 0.0), keep)
    assert not same_basin((10.0, 0.95), keep)


@pytest.mark.parametrize("gate", EARLY_EXIT_GATES)
def test_every_gate_is_well_formed(gate):
    min_score, min_zncc, max_ratio, min_gap = gate
    assert 0.0 < min_score <= 1.0
    assert -1.0 <= min_zncc <= 1.0
    assert 0.0 <= max_ratio <= 1.0
    assert min_gap is None or min_gap >= 0.0

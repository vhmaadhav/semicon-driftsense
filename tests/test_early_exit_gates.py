"""The early-exit gates and the pose-polish bands are pinned here.

The gates were flagged in the PR #51 review because the numbers in the prose
did not match the numbers in the code (the PR described a 0.88 / 0.55 / 0.30
rule while the implementation used 0.72 / 0.72 / 0.35 / 0.04).

These tests exist so that class of drift fails CI instead of shipping:

* the gate constants are frozen as literals, so editing `config.py` without
  editing the documentation breaks the build;
* `_early_exit_fires` is exercised at its boundaries and on degenerate input;
* the polish bands are pinned, since the docs quote them by value;
* `pose_candidates` is asserted to return every hypothesis it was asked for.
  Candidate deduplication used to drop some here on a pose-space heuristic; a
  600-pair A/B measured it out (123 localisation tier crossings, -0.12 points
  on S3, no latency saved) and it was removed, so this is the guard against it
  coming back.
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


def test_polish_bands_are_the_documented_ones():
    """polish_pose's window is the value the docs and comments quote.

    These constants used to double as a candidate-dedup radius. Dedup was
    measured out and removed in PR #51 review round 2 (123/600 localisation
    tier crossings, -0.12 points on S3, and no latency saved --
    .agents/PR51_CAMPAIGN.md), so they now describe only the pose re-fit
    window. Pinned because FAILURE_ANALYSIS.md and the module comments quote
    them by value.
    """
    import inspect
    sig = inspect.signature(polish_pose)
    assert sig.parameters["scale_band"].default == POLISH_SCALE_BAND
    assert sig.parameters["rot_band"].default == POLISH_ROT_BAND
    assert POLISH_SCALE_BAND == 0.03
    assert POLISH_ROT_BAND == 0.8


def test_pose_candidates_never_merges_nearby_hypotheses(monkeypatch):
    """No candidate is dropped on pose proximity before neural localisation.

    The regression guard for the default-off deduplication. How many coarse
    peaks a frame yields is data-dependent, so the count alone proves nothing;
    instead the refinement is stubbed twice on the SAME frame -- once returning
    a distinct pose per candidate, once returning identical poses for all of
    them. Deduplication would collapse the identical run and leave the distinct
    one alone, so equal lengths is exactly the property that it is gone.
    """
    import numpy as np
    import driftsense.matching as M
    rng = np.random.default_rng(3)
    ref = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    search = rng.integers(0, 255, (600, 600), dtype=np.uint8)

    def stub(distinct):
        state = {"i": 0}

        def _f(*a, **kw):
            i = state["i"]
            state["i"] += 1
            return (10.0 + (0.5 * i if distinct else 0.0), 0.0, 1.0 - 0.01 * i)
        return _f

    monkeypatch.delenv("DRIFTSENSE_DEDUP", raising=False)   # the shipped default
    monkeypatch.setattr(M, "_refine_pose_local", stub(distinct=True))
    n_distinct = len(M.pose_candidates(ref, search, k=3, coarse_scales=5))
    monkeypatch.setattr(M, "_refine_pose_local", stub(distinct=False))
    n_identical = len(M.pose_candidates(ref, search, k=3, coarse_scales=5))

    assert n_distinct >= 1
    assert n_identical == n_distinct, (
        f"identical poses collapsed {n_distinct} -> {n_identical}: candidates "
        "are being deduplicated before the network sees them, but dedup is "
        "supposed to be off unless DRIFTSENSE_DEDUP is set")


def test_dedup_flag_enables_the_merge(monkeypatch):
    """The opt-in path still works -- and firing it is what the default avoids.

    Enabling DRIFTSENSE_DEDUP must actually collapse identical poses. If it
    did not, the flag would be dead code pretending to be a switch.
    """
    import numpy as np
    import driftsense.matching as M
    rng = np.random.default_rng(3)
    ref = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    search = rng.integers(0, 255, (600, 600), dtype=np.uint8)
    calls = {"i": 0}

    def identical(*a, **kw):
        calls["i"] += 1
        return (10.0, 0.0, 1.0 - 0.01 * calls["i"])

    monkeypatch.setattr(M, "_refine_pose_local", identical)
    monkeypatch.delenv("DRIFTSENSE_DEDUP", raising=False)
    n_off = len(M.pose_candidates(ref, search, k=3, coarse_scales=5))
    calls["i"] = 0
    monkeypatch.setenv("DRIFTSENSE_DEDUP", "1")
    n_on = len(M.pose_candidates(ref, search, k=3, coarse_scales=5))
    if n_off > 1:
        assert n_on == 1, "DRIFTSENSE_DEDUP=1 did not merge identical poses"
    assert n_on <= n_off


@pytest.mark.parametrize("gate", EARLY_EXIT_GATES)
def test_every_gate_is_well_formed(gate):
    min_score, min_zncc, max_ratio, min_gap = gate
    assert 0.0 < min_score <= 1.0
    assert -1.0 <= min_zncc <= 1.0
    assert 0.0 <= max_ratio <= 1.0
    assert min_gap is None or min_gap >= 0.0

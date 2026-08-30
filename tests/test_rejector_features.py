"""Issue #6: winner-margin as a present/absent rejector feature.

The rejector spec asks for the winner's `min(score, zncc)` margin over the
runner-up as an inference-time feature. It must be attached by
locate_phase2 on every decode path (it needs no feature maps -- only the
candidate list the decode already has), so the shipped default-zncc path
records it too.
"""

from __future__ import annotations

import numpy as np
import pytest

import driftsense.matching as matching


def test_winner_margin_is_winner_minus_best_runner_up():
    cands = [
        {"score": 0.7, "zncc": 0.1},   # min = 0.1
        {"score": 0.7, "zncc": 0.9},   # min = 0.7  <- winner
        {"score": 0.7, "zncc": 0.2},   # min = 0.2  <- runner-up
    ]
    assert matching.winner_margin(cands) == pytest.approx(0.5)


def test_winner_margin_nan_for_single_candidate():
    assert np.isnan(matching.winner_margin([{"score": 0.7, "zncc": 0.9}]))


def test_winner_margin_zero_on_exact_tie():
    cands = [{"score": 0.6, "zncc": 0.6}, {"score": 0.7, "zncc": 0.6}]
    assert matching.winner_margin(cands) == pytest.approx(0.0)


def test_winner_margin_tolerates_missing_zncc():
    # A missing zncc is skipped, not penalised: candidate strengths are
    # min-of-available = 0.9 and min(0.4, 0.2) = 0.2 -> margin 0.7.
    cands = [{"score": 0.9}, {"score": 0.4, "zncc": 0.2}]
    assert matching.winner_margin(cands) == pytest.approx(0.7)


def test_locate_phase2_records_winner_margin_on_default_path(monkeypatch):
    # Same stub family as tests/test_verification.py: candidates carry zncc
    # .1/.9/.2 against a constant network score of .7, so the per-candidate
    # min(score, zncc) values are .1/.7/.2 and the winner's margin is .5.
    candidates = [(9.0, -1.0, .3), (10.0, 0.0, .4), (11.0, 1.0, .2)]
    monkeypatch.setattr(matching, "pose_candidates",
                        lambda reference, search, k, **kw: candidates[:k])
    monkeypatch.setattr(matching, "canonicalize_search",
                        lambda search, m, r: (search, np.array([[1., 0., 0.], [0., 1., 0.]])))
    monkeypatch.setattr(matching, "locate",
                        lambda *args, **kwargs: {"x": 30., "y": 30., "score": .7,
                                                 "peak_ratio": .5, "coarse": (30., 30.)})
    scores = iter([.1, .9, .2])
    monkeypatch.setattr(matching, "refine_zncc",
                        lambda search, template, cx, cy, radius: (cx + 0.25, cy - 0.25, next(scores)))
    monkeypatch.setattr(matching, "polish_pose",
                        lambda reference, search, x, y, m, r: (m, r, 1.0))

    reference = np.zeros((100, 100), dtype=np.uint8)
    search = np.zeros((60, 60), dtype=np.uint8)
    result = matching.locate_phase2(None, reference, search, None,
                                    verification="zncc", polish=False)
    assert result["winner_margin"] == pytest.approx(0.5)


def test_locate_phase2_margin_nan_under_fixed_pose(monkeypatch):
    monkeypatch.setattr(matching, "canonicalize_search",
                        lambda search, m, r: (search, np.array([[1., 0., 0.], [0., 1., 0.]])))
    monkeypatch.setattr(matching, "locate",
                        lambda *args, **kwargs: {"x": 30., "y": 30., "score": .7,
                                                 "peak_ratio": .5, "coarse": (30., 30.)})
    monkeypatch.setattr(matching, "refine_zncc",
                        lambda search, template, cx, cy, radius: (cx, cy, 0.8))
    monkeypatch.setattr(matching, "polish_pose",
                        lambda reference, search, x, y, m, r: (m, r, 1.0))
    reference = np.zeros((100, 100), dtype=np.uint8)
    search = np.zeros((60, 60), dtype=np.uint8)
    result = matching.locate_phase2(None, reference, search, None,
                                    verification="zncc", polish=False,
                                    pose=(10.0, 0.0))
    assert np.isnan(result["winner_margin"])

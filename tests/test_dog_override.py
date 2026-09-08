"""The DoG-guarded selector override: native ZNCC keeps the decision.

`scripts/verify_scores.py` measured `zncc_dog` as the strongest alternative
hypothesis selector on Set B (net +10 recovered pairs against the incumbent
ZNCC's +7) and it was never wired into inference. Adopting it wholesale is a
bad trade: DoG only helps on the *contested* decisions, and on the ~87% of
pairs that ZNCC already gets right it can only break things.

`verification="dog-override"` is the guarded form. ZNCC still owns the choice;
DoG may overturn it only when both hold:

  1. DoG's argmax differs from ZNCC's argmax, and
  2. the ZNCC margin between those two candidates is below `dog_override_margin`
     -- that is, ZNCC itself is not confident in its own winner.

These tests pin the decision rule, its boundary, and the cost: the mode must
build the DoG feature map and NOTHING else. The rank transform is the expensive
representation and dog-override must never pay for it.
"""
from __future__ import annotations

import numpy as np
import pytest

import driftsense.matching as matching
from driftsense.config import DOG_OVERRIDE_MARGIN

# Candidate poses, in the order pose_candidates ranks them. Index 0 is scale
# 9.0, index 1 is scale 10.0, index 2 is scale 11.0 -- the tests assert on the
# selected scale because it names the chosen hypothesis unambiguously.
CANDIDATES = [(9.0, -1.0, .3), (10.0, 0.0, .4), (11.0, 1.0, .2)]
SCALE_OF = {0: 9.0, 1: 10.0, 2: 11.0}


def _stub(monkeypatch, zncc, dog=None, counts=None):
    """Drive locate_phase2 with exact per-candidate zncc and dog values.

    The network score stays at 0.70 with peak_ratio 0.50 so neither
    EARLY_EXIT_GATES entry can fire (both need score >= 0.72); the selection
    rule is then the only thing under test, whatever zncc values we hand it.
    """
    monkeypatch.setattr(matching, "pose_candidates",
                        lambda reference, search, k, **kw: CANDIDATES[:k])
    monkeypatch.setattr(matching, "canonicalize_search",
                        lambda search, m, r: (search, np.array([[1., 0., 0.],
                                                                [0., 1., 0.]])))
    monkeypatch.setattr(matching, "locate",
                        lambda *a, **kw: {"x": 30., "y": 30., "score": .7,
                                          "peak_ratio": .5, "coarse": (30., 30.)})
    zncc_it = iter(zncc)
    monkeypatch.setattr(matching, "refine_zncc",
                        lambda search, template, cx, cy, radius:
                        (cx + 0.25, cy - 0.25, next(zncc_it)))
    monkeypatch.setattr(matching, "polish_pose",
                        lambda reference, search, x, y, m, r: (m, r, 1.0))
    if dog is not None:
        dog_it = iter(dog)
        monkeypatch.setattr(matching, "local_match_score",
                            lambda search, template, cx, cy: next(dog_it))
    if counts is not None:
        for name in ("rank_transform", "common_band", "dog_feature"):
            original = getattr(matching, name)

            def counted(image, _name=name, _original=original):
                counts[_name].append(image.shape)
                return _original(image)

            monkeypatch.setattr(matching, name, counted)


def _frames(seed=11):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 256, (110, 110), dtype=np.uint8),
            rng.integers(0, 256, (60, 60), dtype=np.uint8))


# subpixel_rows=False throughout. The row refinement runs after the pose
# decision is final and can only move x, so it is downstream of everything
# these tests assert -- and leaving it on makes them lie about their own
# fixture: drift_row_refine calls refine_zncc a fourth time, exhausting the
# three-value stub iterator, and the pipeline's deliberate catch-and-warn
# swallows the StopIteration. Turning it off keeps the stub honest instead of
# testing selection through a stage that is quietly failing.
SELECTION_ONLY = dict(polish=False, subpixel_rows=False)


def _run(monkeypatch, zncc, dog, **kw):
    _stub(monkeypatch, zncc, dog)
    reference, search = _frames()
    return matching.locate_phase2(None, reference, search, None,
                                  verification="dog-override",
                                  **SELECTION_ONLY, **kw)


def test_dog_override_is_an_accepted_mode():
    assert DOG_OVERRIDE_MARGIN > 0.0


def test_unknown_verification_still_rejected(monkeypatch):
    _stub(monkeypatch, [.1, .9, .2])
    reference, search = _frames()
    with pytest.raises(ValueError, match="verification must be one of"):
        matching.locate_phase2(None, reference, search, None,
                               verification="zncc_dog", **SELECTION_ONLY)


def test_agreement_leaves_the_zncc_winner_alone(monkeypatch):
    """DoG picks what ZNCC picked: nothing to override, contested or not."""
    # zncc argmax = 1 (0.82); dog argmax = 1 as well.
    result = _run(monkeypatch, zncc=[.80, .82, .10], dog=[1.0, 5.0, 0.0],
                  dog_override_margin=0.05)
    assert result["scale"] == SCALE_OF[1]


def test_contested_disagreement_hands_the_pair_to_dog(monkeypatch):
    """Margin 0.02 < eps 0.05 and DoG disagrees -- DoG's candidate wins."""
    # zncc argmax = 1 (0.82), dog argmax = 0. Margin = 0.82 - 0.80 = 0.02.
    result = _run(monkeypatch, zncc=[.80, .82, .10], dog=[5.0, 1.0, 0.0],
                  dog_override_margin=0.05)
    assert result["scale"] == SCALE_OF[0]
    # The override must hand back the OVERRIDDEN candidate's own geometry,
    # not the ZNCC winner's coordinates with a new scale label on them.
    assert result["zncc"] == pytest.approx(.80)
    assert result["theta"] == pytest.approx(-1.0)


def test_confident_zncc_is_not_overridden(monkeypatch):
    """DoG disagrees but ZNCC leads by 0.60 -- the incumbent keeps the pair."""
    result = _run(monkeypatch, zncc=[.30, .90, .10], dog=[5.0, 1.0, 0.0],
                  dog_override_margin=0.05)
    assert result["scale"] == SCALE_OF[1]
    assert result["zncc"] == pytest.approx(.90)


def test_margin_equal_to_epsilon_does_not_override(monkeypatch):
    """The gate is strict `<`. 0.75 - 0.50 is exact in binary, so this pins
    the boundary rather than a floating-point accident."""
    result = _run(monkeypatch, zncc=[.50, .75, .10], dog=[5.0, 1.0, 0.0],
                  dog_override_margin=0.25)
    assert result["scale"] == SCALE_OF[1]
    # ...and one ulp of slack the other way does override.
    result = _run(monkeypatch, zncc=[.50, .75, .10], dog=[5.0, 1.0, 0.0],
                  dog_override_margin=0.2500001)
    assert result["scale"] == SCALE_OF[0]


def test_a_zero_margin_never_overrides_when_dog_agrees(monkeypatch):
    """Tied zncc with agreeing dog: the incumbent's tie rule (first argmax)
    survives, so the mode cannot reorder ties it was not asked to touch."""
    result = _run(monkeypatch, zncc=[.90, .90, .10], dog=[5.0, 1.0, 0.0],
                  dog_override_margin=0.05)
    # zncc argmax takes the FIRST maximum -> index 0; dog also picks 0.
    assert result["scale"] == SCALE_OF[0]


def test_dog_override_builds_only_the_dog_feature_map(monkeypatch):
    """The whole point of the guard is that it is nearly free. The rank
    transform is the expensive representation; dog-override must not build it,
    nor the common band."""
    counts = {n: [] for n in ("rank_transform", "common_band", "dog_feature")}
    _stub(monkeypatch, zncc=[.80, .82, .10], counts=counts)
    reference, search = _frames()
    matching.locate_phase2(None, reference, search, None,
                           verification="dog-override", **SELECTION_ONLY)
    assert counts["rank_transform"] == []
    assert counts["common_band"] == []
    # One search frame plus one template per hypothesis.
    assert counts["dog_feature"].count(search.shape) == 1
    assert len(counts["dog_feature"]) == 4


def test_shipped_zncc_path_builds_no_feature_maps(monkeypatch):
    """Regression guard on the default decode: adding a mode must not make the
    shipped path pay for a representation it never reads."""
    counts = {n: [] for n in ("rank_transform", "common_band", "dog_feature")}
    _stub(monkeypatch, zncc=[.1, .9, .2], counts=counts)
    reference, search = _frames()
    result = matching.locate_phase2(None, reference, search, None,
                                    verification="zncc", **SELECTION_ONLY)
    assert all(v == [] for v in counts.values())
    assert result["scale"] == SCALE_OF[1]
    assert {"rank", "band", "dog"}.isdisjoint(result)


def test_winner_carries_its_dog_score_but_not_rank_or_band(monkeypatch):
    """Same contract as majority/consensus (issue #6): the SELECTED winner's
    verification statistic is recorded so the present/absent rejector can fit
    on inference-time features -- but only the one this mode computed."""
    result = _run(monkeypatch, zncc=[.80, .82, .10], dog=[5.0, 1.0, 0.0],
                  dog_override_margin=0.05)
    assert "dog" in result
    assert {"rank", "band"}.isdisjoint(result)


def test_single_hypothesis_short_circuits_without_a_dog_score(monkeypatch):
    """A pinned pose yields one candidate; there is nothing to override and
    the mode must not require a dog score to say so."""
    _stub(monkeypatch, zncc=[.42])
    reference, search = _frames()
    result = matching.locate_phase2(None, reference, search, None, pose=(10.0, 0.0),
                                    verification="dog-override", **SELECTION_ONLY)
    assert result["scale"] == SCALE_OF[1]
    assert result["zncc"] == pytest.approx(.42)


def test_default_margin_comes_from_config(monkeypatch):
    """The knob is defined once, in driftsense.config, like every other
    measured constant in this pipeline."""
    import inspect
    sig = inspect.signature(matching.locate_phase2)
    assert sig.parameters["dog_override_margin"].default == DOG_OVERRIDE_MARGIN

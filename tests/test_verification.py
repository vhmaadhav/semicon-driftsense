"""Classical Phase-2 verification representations and instrumentation."""
from __future__ import annotations

import numpy as np
import pytest

import driftsense.matching as matching
from driftsense.verification import (
    common_band, dog_feature, local_match_score, rank_transform,
)


def test_rank_transform_is_deterministic():
    image = np.arange(49, dtype=np.uint8).reshape(7, 7)
    first = rank_transform(image)
    second = rank_transform(image)
    assert np.array_equal(first, second)
    assert first.dtype == np.float32
    assert 0 <= first.min() <= first.max() <= 24


def test_rank_is_invariant_to_positive_affine_brightness():
    image = np.random.default_rng(2).normal(40, 7, (31, 29)).astype(np.float32)
    transformed = image * 2.75 + 19.0
    assert np.array_equal(rank_transform(image), rank_transform(transformed))


def test_rank_is_local_under_sparse_impulse_changes():
    image = np.random.default_rng(3).integers(20, 230, (64, 64), dtype=np.uint8)
    corrupted = image.copy()
    corrupted[32, 32] = 255
    unchanged = rank_transform(image) == rank_transform(corrupted)
    # One impulse can affect only its own 5x5 dependency neighbourhood.
    assert unchanged.mean() > 0.99


@pytest.mark.parametrize("feature", [common_band, dog_feature])
def test_filtered_feature_preserves_dimensions_and_is_float32(feature):
    image = np.zeros((43, 57), dtype=np.uint8)
    output = feature(image)
    assert output.shape == image.shape
    assert output.dtype == np.float32


def test_local_match_score_does_not_modify_coordinates_or_inputs():
    rng = np.random.default_rng(4)
    search = rng.normal(size=(80, 90)).astype(np.float32)
    template = search[28:43, 37:54].copy()
    search_before, template_before = search.copy(), template.copy()
    cx, cy = 37 + template.shape[1] / 2, 28 + template.shape[0] / 2
    score = local_match_score(search, template, cx, cy, radius=4)
    assert score > 0.99
    assert (cx, cy) == (45.5, 35.5)
    assert np.array_equal(search, search_before)
    assert np.array_equal(template, template_before)


def _stub_phase2(monkeypatch, feature_counts=None):
    candidates = [(9.0, -1.0, .3), (10.0, 0.0, .4), (11.0, 1.0, .2)]
    # **kw so the stub keeps working as pose_candidates gains parameters
    # (coarse_scales, band). The test is about hypothesis *selection*; how the
    # candidates were generated is not what it is asserting.
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
    if feature_counts is not None:
        for name in ("rank_transform", "common_band", "dog_feature"):
            original = getattr(matching, name)
            def counted(image, _name=name, _original=original):
                feature_counts[_name].append(image.shape)
                return _original(image)
            monkeypatch.setattr(matching, name, counted)


def test_verification_disabled_keeps_three_hypotheses_and_output_contract(monkeypatch):
    _stub_phase2(monkeypatch)
    reference = np.zeros((100, 100), dtype=np.uint8)
    search = np.zeros((60, 60), dtype=np.uint8)
    result = matching.locate_phase2(None, reference, search, None,
                                    verification="zncc", polish=False)
    assert result["n_hypotheses"] == 3
    assert "hypotheses" not in result
    assert result["scale"] == 10.0
    assert result["zncc"] == pytest.approx(.9)
    assert {"x", "y", "scale", "theta", "score"} <= set(result)


def test_research_feature_maps_are_computed_once_per_pair(monkeypatch):
    counts = {name: [] for name in ("rank_transform", "common_band", "dog_feature")}
    _stub_phase2(monkeypatch, counts)
    reference = np.random.default_rng(5).integers(0, 256, (100, 100), dtype=np.uint8)
    search = np.random.default_rng(6).integers(0, 256, (60, 60), dtype=np.uint8)
    result = matching.locate_phase2(None, reference, search, None, polish=False,
                                    return_hypotheses=True, verification="zncc")
    assert len(result["hypotheses"]) == 3
    for name, shapes in counts.items():
        assert shapes.count(search.shape) == 1, (name, shapes)
        assert len(shapes) == 4  # one search plus one small template per hypothesis


def test_instrumentation_does_not_move_native_zncc_coordinates(monkeypatch):
    _stub_phase2(monkeypatch)
    reference = np.random.default_rng(7).integers(0, 256, (100, 100), dtype=np.uint8)
    search = np.random.default_rng(8).integers(0, 256, (60, 60), dtype=np.uint8)
    result = matching.locate_phase2(None, reference, search, None, polish=False,
                                    return_hypotheses=True)
    for hypothesis in result["hypotheses"]:
        assert hypothesis["x"] == pytest.approx(30.25)
        assert hypothesis["y"] == pytest.approx(29.75)


def test_majority_and_consensus_only_change_the_selected_hypothesis(monkeypatch):
    _stub_phase2(monkeypatch)
    monkeypatch.setattr(matching, "local_match_score",
                        lambda search, template, cx, cy: float(template.shape[0]))
    reference = np.random.default_rng(9).integers(0, 256, (110, 110), dtype=np.uint8)
    search = np.random.default_rng(10).integers(0, 256, (60, 60), dtype=np.uint8)
    majority = matching.locate_phase2(None, reference, search, None, polish=False,
                                      verification="majority")
    assert majority["scale"] == 9.0
    # Contract since issue #6: dog stays instrumentation-only, but the WINNER's
    # rank/band are deliberately recorded under non-zncc selectors so the
    # present/absent rejector can fit on inference-time features. The default
    # zncc path computes neither (test_verification_disabled... covers it), so
    # the register.py consumer view is unchanged where it matters.
    assert "dog" not in majority
    assert {"rank", "band"} <= set(majority)
    _stub_phase2(monkeypatch)
    consensus = matching.locate_phase2(None, reference, search, None, polish=False,
                                       verification="consensus")
    assert consensus["scale"] == 9.0
    # All candidate coordinates still originate in the same native-ZNCC stub.
    assert (consensus["x"], consensus["y"]) == pytest.approx((30.25, 29.75))

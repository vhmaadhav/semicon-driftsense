"""The min(network, median-ZNCC) confidence (issue #87).

Pins the mechanism the statistic relies on -- a median restores correlation on
a noisy true match and cannot manufacture it for a mismatched reference -- and
the wiring: measured at the rigid answer and final pose, only in the "min_med3"
mode, with the legacy branch untouched.
"""
import cv2
import numpy as np
import pytest

from driftsense import matching
from driftsense.matching import denoised_zncc, make_template, refine_zncc, standardize


def _texture(seed):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(128, 60, (1000, 1000)).astype(np.float32), (0, 0), 7)
    img = (img - img.mean()) / img.std() * 40 + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def _noisy_scene(ref, seed=1, impulse=0.06):
    rng = np.random.default_rng(seed)
    frame = np.full((300, 300), 128, np.float32)
    frame[100:200, 100:200] = make_template(ref, 10.0, 0.0)
    frame += rng.normal(0, 14, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    hit = rng.random(frame.shape) < impulse
    frame[hit] = np.where(rng.random(hit.sum()) < 0.5, 0, 255).astype(np.uint8)
    return frame


def _raw_zncc(ref, frame, cx, cy):
    tpl = make_template(ref, 10.0, 0.0)
    return refine_zncc(standardize(frame / 255.0), standardize(tpl / 255.0), cx, cy, radius=4)[2]


def test_median_lifts_a_noisy_true_match_but_not_a_mismatch():
    ref, other = _texture(3), _texture(4)
    frame = _noisy_scene(ref)
    present_raw = _raw_zncc(ref, frame, 150.0, 150.0)
    present_med = denoised_zncc(ref, frame, 150.0, 150.0, 10.0, 0.0)
    absent_raw = _raw_zncc(other, frame, 150.0, 150.0)
    absent_med = denoised_zncc(other, frame, 150.0, 150.0, 10.0, 0.0)
    assert present_med > present_raw + 0.10, (present_raw, present_med)
    assert absent_med < 0.35, absent_med
    assert present_med - absent_med > present_raw - absent_raw


def test_matches_refine_zncc_on_the_filtered_frame():
    """Same window arithmetic as refine_zncc, so values measured with it carry over."""
    ref = _texture(5)
    frame = _noisy_scene(ref, seed=2)
    med = cv2.medianBlur(frame, 3)
    tpl = make_template(ref, 10.0, 0.0)
    for cx, cy in ((150.0, 150.0), (148.3, 152.6)):
        expect = refine_zncc(standardize(med / 255.0), standardize(tpl / 255.0), cx, cy, radius=4)[2]
        assert denoised_zncc(ref, frame, cx, cy, 10.0, 0.0) == pytest.approx(expect, abs=1e-6)


def test_window_leaving_the_frame_scores_zero():
    ref = _texture(6)
    frame = _noisy_scene(ref)
    assert denoised_zncc(ref, frame, 20.0, 20.0, 10.0, 0.0) == 0.0


def test_non_uint8_frames_are_accepted():
    ref = _texture(7)
    frame = _noisy_scene(ref)
    a = denoised_zncc(ref, frame, 150.0, 150.0, 10.0, 0.0)
    b = denoised_zncc(ref, frame.astype(np.float64), 150.0, 150.0, 10.0, 0.0)
    assert b == pytest.approx(a, abs=1e-3)


def _stub(monkeypatch, seen):
    monkeypatch.setattr(matching, "pose_candidates",
                        lambda reference, search, k, **kw: [(10.0, 0.0, .5)])
    monkeypatch.setattr(matching, "canonicalize_search",
                        lambda search, m, r: (search, np.array([[1., 0., 0.], [0., 1., 0.]])))
    monkeypatch.setattr(matching, "locate",
                        lambda *args, **kwargs: {"x": 30.0, "y": 31.0, "score": .8,
                                                 "peak_ratio": .5, "coarse": (30.0, 31.0)})
    monkeypatch.setattr(matching, "refine_zncc",
                        lambda search, template, cx, cy, radius: (cx + 0.25, cy, .9))
    monkeypatch.setattr(matching, "polish_pose",
                        lambda reference, search, x, y, m, r: (13.0, 6.0, 1.0))   # clipped to 12 / 5
    monkeypatch.setattr(matching, "drift_row_refine",
                        lambda search, template, cx, cy, **kw: (cx + 3.0, cy))

    def fake_dz(reference, search, cx, cy, scale, theta):
        seen.append((cx, cy, scale, theta))
        return 0.42
    monkeypatch.setattr(matching, "denoised_zncc", fake_dz)


def test_min_med3_is_measured_at_the_rigid_answer_and_final_pose(monkeypatch):
    seen = []
    _stub(monkeypatch, seen)
    monkeypatch.setattr(matching, "SHIPPED_CONFIDENCE", "min_med3")
    ref, sea = np.zeros((100, 100), np.uint8), np.zeros((60, 60), np.uint8)
    res = matching.locate_phase2(None, ref, sea, None, verification="zncc")
    assert seen == [(30.25, 31.0, 12.0, 5.0)], "rigid x (before the row move), clipped final pose"
    assert res["x"] == pytest.approx(33.25)           # the row move still reaches the answer
    assert res["zncc_med3"] == 0.42
    assert res["confidence"] == pytest.approx(min(0.8, 0.42))


def test_legacy_branch_is_unchanged(monkeypatch):
    seen = []
    _stub(monkeypatch, seen)
    monkeypatch.setattr(matching, "SHIPPED_CONFIDENCE", "legacy_min")
    ref, sea = np.zeros((100, 100), np.uint8), np.zeros((60, 60), np.uint8)
    res = matching.locate_phase2(None, ref, sea, None, verification="zncc")
    assert seen == []
    assert "zncc_med3" not in res
    assert res["confidence"] == pytest.approx(min(0.8, 0.9))

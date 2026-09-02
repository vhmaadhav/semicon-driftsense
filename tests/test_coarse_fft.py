"""driftsense.coarse_fft contract (workstream A, deliverable 3).

The spectral precomputed-search coarse scorer must reproduce, for the same
(search, template) pair, the same peak score cv2.matchTemplate's
TM_CCOEFF_NORMED reports -- that is what lets it substitute for the coarse
sweep's per-call correlation without moving any ranking. These tests pin:

* API: prepare_search(search) returns an index; index.peak_score(template)
  returns a float.
* Values: peak agrees with cv2.matchTemplate within 1e-6 on synthetic pairs
  (random noise search, planted template, and off-nominal templates).
* Robustness: flat windows (zero variance) do not produce NaN/inf; templates
  larger than the search are rejected the way matching._peak_score does
  (returns -inf).
* Memory hygiene: peak_score must not mutate the prepared index (it is
  reused across ~50 template calls per pair in the coarse sweep).

Runs on synthetic images only -- official reference material is not needed
for the value contract.
"""

import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from driftsense.coarse_fft import SpectralSearchIndex, prepare_search  # noqa: E402


def _noisy_search(rng, h=400, w=400):
    s = rng.integers(40, 210, (h, w), dtype=np.uint8)
    return cv2.GaussianBlur(s, (0, 0), 1.5).astype(np.float32)


def _planted_template(search, rng, th=32, tw=32):
    y = int(rng.integers(0, search.shape[0] - th))
    x = int(rng.integers(0, search.shape[1] - tw))
    return search[y:y + th, x:x + tw].copy(), (y, x)


def test_prepare_returns_index_with_search_dft():
    rng = np.random.default_rng(5)
    search = _noisy_search(rng)
    idx = prepare_search(search)
    assert isinstance(idx, SpectralSearchIndex)
    # the search DFT is precomputed exactly once, at construction
    assert idx.search_shape == search.shape
    assert idx.F_search is not None
    # padding is at least large enough for a linear (wrap-free) correlation
    assert idx.pad_shape[0] >= search.shape[0] + 64
    assert idx.pad_shape[1] >= search.shape[1] + 64


def test_peak_score_matches_cv2_tm_ccoeff_normed():
    rng = np.random.default_rng(6)
    search = _noisy_search(rng)
    tmpl, _ = _planted_template(search, rng)
    idx = prepare_search(search)
    got = idx.peak_score(tmpl)
    want = float(cv2.minMaxLoc(cv2.matchTemplate(
        search, tmpl, cv2.TM_CCOEFF_NORMED))[1])
    assert abs(got - want) <= 1e-6, f"spectral {got!r} vs cv2 {want!r}"


def test_peak_score_matches_cv2_across_template_sizes():
    rng = np.random.default_rng(7)
    search = _noisy_search(rng, 300, 300)
    idx = prepare_search(search)
    for th, tw in ((16, 16), (20, 24), (32, 32), (48, 40)):
        tmpl, _ = _planted_template(search, rng, th, tw)
        got = idx.peak_score(tmpl)
        want = float(cv2.minMaxLoc(cv2.matchTemplate(
            search, tmpl, cv2.TM_CCOEFF_NORMED))[1])
        assert abs(got - want) <= 1e-6, f"{th}x{tw}: {got!r} vs {want!r}"


def test_peak_score_off_nominal_template_not_planted():
    """A template that is NOT a crop (scaled/rotated content) must still
    produce the same ZNCC peak as cv2 -- the coarse sweep correlates
    synthesized templates, never planted crops."""
    rng = np.random.default_rng(8)
    search = _noisy_search(rng)
    patch, _ = _planted_template(search, rng, 40, 40)
    tmpl = cv2.warpAffine(patch, cv2.getRotationMatrix2D((20, 20), 7.0, 0.9),
                          (40, 40)).astype(np.float32)
    idx = prepare_search(search)
    got = idx.peak_score(tmpl)
    want = float(cv2.minMaxLoc(cv2.matchTemplate(
        search, tmpl, cv2.TM_CCOEFF_NORMED))[1])
    assert abs(got - want) <= 1e-6, f"{got!r} vs {want!r}"


def test_peak_score_rejects_template_larger_than_search():
    rng = np.random.default_rng(9)
    search = _noisy_search(rng, 100, 100)
    idx = prepare_search(search)
    big = np.zeros((150, 90), np.float32)
    assert idx.peak_score(big) == -np.inf
    big2 = np.zeros((90, 150), np.float32)
    assert idx.peak_score(big2) == -np.inf


def test_peak_score_is_finite_on_flat_windows():
    """A flat search region gives zero variance; the scorer must return a
    finite float, never NaN/inf, for any template."""
    rng = np.random.default_rng(10)
    flat = np.full((200, 200), 3.0, np.float32)
    idx = prepare_search(flat)
    tmpl = rng.integers(0, 255, (20, 20)).astype(np.float32)
    got = idx.peak_score(tmpl)
    assert np.isfinite(got)


def test_peak_score_does_not_mutate_the_index():
    """The coarse sweep reuses one index for ~50 templates; a scorer that
    scribbles on its cached state would corrupt later calls."""
    rng = np.random.default_rng(11)
    search = _noisy_search(rng)
    idx = prepare_search(search)
    F_before = idx.F_search.copy()
    tmpl1, _ = _planted_template(search, rng, 24, 24)
    s1 = idx.peak_score(tmpl1)
    assert np.array_equal(idx.F_search, F_before)
    # repeat call returns the identical value (no state drift)
    assert idx.peak_score(tmpl1) == s1


def test_peak_score_finds_planted_peak_location():
    """The scorer's job is ranking: its argmax must sit at the planted
    location (allowing the 1-window tolerance of an exact-value match)."""
    rng = np.random.default_rng(12)
    search = _noisy_search(rng)
    tmpl, (y, x) = _planted_template(search, rng, 32, 32)
    idx = prepare_search(search)
    # reconstruct the full map position of the peak from the scorer
    th, tw = tmpl.shape
    got = idx.peak_score(tmpl)
    want_map = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
    assert abs(got - float(want_map.max())) <= 1e-6
    wy, wx = np.unravel_index(int(want_map.argmax()), want_map.shape)
    assert (wy, wx) == (y, x)

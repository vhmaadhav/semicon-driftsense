"""Variance-stabilising transform (issue #13, driftsense/vst.py).

The load-bearing test in this file is `test_shipped_default_is_a_no_op`: the
transform is UNMEASURED, so the shipped decode must be bit-identical to the
pre-#13 path until the 5 x 200 + 2,250-pair paired bootstrap says otherwise.
"""
from __future__ import annotations

import numpy as np
import pytest

from driftsense import vst
from driftsense.config import SHIPPED_VST
from driftsense.matching import pose_candidates


@pytest.fixture(autouse=True)
def _clear_vst_env(monkeypatch):
    """Every test starts from the configured default, not an inherited flag."""
    for name in ("DRIFTSENSE_VST", "DRIFTSENSE_VST_COARSE",
                 "DRIFTSENSE_VST_VERIFY", "DRIFTSENSE_VST_REFINE"):
        monkeypatch.delenv(name, raising=False)


def test_shipped_default_is_a_no_op():
    """The default must not change the shipped decode. See module docstring."""
    assert SHIPPED_VST == "none"
    for stage in vst.STAGES:
        assert vst.resolve_mode(stage) == "none"
    img = np.arange(64, dtype=np.uint8).reshape(8, 8)
    # Identity, not merely equality: "none" must not even copy or upcast, or
    # the pre-#13 dtype path through matchTemplate would silently change.
    assert vst.apply(img, "none") is img
    ref, sea = vst.apply_pair(img, img, "coarse")
    assert ref is img and sea is img


def test_anscombe_stabilises_poisson_variance():
    """The whole point: post-transform variance must not track the mean."""
    rng = np.random.default_rng(0)
    variances = []
    for lam in (5.0, 20.0, 80.0, 200.0):
        counts = rng.poisson(np.full(40_000, lam)).astype(np.float32)
        variances.append(float(vst.anscombe(counts).var()))
    # Anscombe targets unit variance for lambda >~ 4.
    assert all(abs(v - 1.0) < 0.05 for v in variances), variances
    # And the raw data it was given genuinely had mean-tracking variance,
    # so the test is not passing on degenerate input.
    raw = [float(rng.poisson(np.full(40_000, lam)).var())
           for lam in (5.0, 200.0)]
    assert raw[1] / raw[0] > 20


def test_anscombe_is_monotone_and_finite():
    """Monotone: it re-weights noise, it does not reorder or quantise signal.

    This is the property that distinguishes it from the rank transform, which
    was measured and rejected (CITATIONS.md sec. 6) precisely because it
    discards ordering information on clean frames.
    """
    x = np.linspace(0.0, 255.0, 1000, dtype=np.float32)
    y = vst.anscombe(x)
    assert np.all(np.diff(y) > 0)
    assert np.all(np.isfinite(y))
    # Negative input is clamped rather than producing NaN.
    assert np.isfinite(vst.anscombe(np.array([-5.0], np.float32))).all()


def test_generalized_anscombe_matches_anscombe_at_unit_gain():
    x = np.linspace(0.0, 255.0, 256, dtype=np.float32)
    np.testing.assert_allclose(vst.generalized_anscombe(x, 1.0, 0.0),
                               vst.anscombe(x), rtol=1e-5)


def test_gat_stabilises_a_gained_poisson_gaussian_signal():
    """var(x) = a*E[x] + b is the model; the fit must recover enough of it.

    Plain Anscombe assumes a = 1, which 8-bit rendered images do not satisfy,
    so this is the case that motivates estimating (a, b) per image.
    """
    rng = np.random.default_rng(7)
    a_true, b_true = 3.0, 25.0
    lam = np.repeat(np.array([6.0, 20.0, 60.0], np.float32), 30_000)
    x = (a_true * rng.poisson(lam)
         + rng.normal(0.0, np.sqrt(b_true), lam.size)).astype(np.float32)

    plain = vst.anscombe(x)
    gat = vst.generalized_anscombe(x, a_true, b_true)

    def spread(y):
        parts = y.reshape(3, -1)
        v = parts.var(axis=1)
        return float(v.max() / v.min())

    # The GAT with the true parameters must flatten the variance-vs-level
    # ratio far better than the a=1 assumption does.
    assert spread(gat) < 1.15
    assert spread(gat) < spread(plain)


def test_estimate_noise_model_recovers_slope_on_synthetic_data():
    rng = np.random.default_rng(11)
    a_true, b_true = 2.5, 16.0
    # Smooth ramp of levels so blocks span a wide mean range, plus noise.
    level = np.tile(np.linspace(5.0, 120.0, 512, dtype=np.float32), (512, 1))
    img = (a_true * rng.poisson(level)
           + rng.normal(0.0, np.sqrt(b_true), level.shape)).astype(np.float32)
    a, b = vst.estimate_noise_model(img)
    assert a > 0.0
    # The lower-envelope fit is deliberately conservative (it must not be
    # inflated by structure), so this asserts the right order of magnitude,
    # not a precise recovery.
    assert 0.4 * a_true <= a <= 1.6 * a_true, (a, b)


def test_estimate_noise_model_declines_on_degenerate_input():
    """A flat image carries no mean range to regress against."""
    a, b = vst.estimate_noise_model(np.full((256, 256), 7.0, np.float32))
    assert (a, b) == (0.0, 0.0)
    # And `apply` falls back rather than raising or returning the input.
    out = vst.apply(np.full((256, 256), 7.0, np.float32), "gat")
    assert np.isfinite(out).all()


def test_unknown_mode_and_stage_raise():
    """A typo must fail loudly, not silently measure the baseline twice."""
    with pytest.raises(ValueError):
        vst.apply(np.zeros((4, 4), np.float32), "ansombe")
    with pytest.raises(ValueError):
        vst.resolve_mode("corase")


def test_stage_env_overrides_global(monkeypatch):
    monkeypatch.setenv("DRIFTSENSE_VST", "gat")
    monkeypatch.setenv("DRIFTSENSE_VST_COARSE", "anscombe")
    assert vst.resolve_mode("coarse") == "anscombe"


def test_unwired_stage_refuses_a_transform(monkeypatch):
    """A flag that cannot take effect must fail, not report a baseline run.

    `verify` and `refine` act on already-standardised (mean-centred) arrays,
    where a sqrt VST would clamp the negative half; they are deferred until
    the coarse-stage result justifies moving the seam upstream of
    `matching.standardize`. See vst.WIRED_STAGES.
    """
    for stage in ("verify", "refine"):
        assert stage not in vst.WIRED_STAGES
        monkeypatch.setenv("DRIFTSENSE_VST", "anscombe")
        with pytest.raises(NotImplementedError):
            vst.resolve_mode(stage)
        # ...but "none" on an unwired stage is fine: nothing is claimed.
        monkeypatch.setenv("DRIFTSENSE_VST", "none")
        assert vst.resolve_mode(stage) == "none"


def test_pose_candidates_unchanged_under_none(monkeypatch):
    """The seam must be inert when off, on the real code path."""
    rng = np.random.default_rng(3)
    ref = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)
    sea = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)

    baseline = pose_candidates(ref, sea, k=2)
    monkeypatch.setenv("DRIFTSENSE_VST_COARSE", "none")
    assert pose_candidates(ref, sea, k=2) == baseline


def test_pose_candidates_runs_under_each_mode(monkeypatch):
    """Shape/dtype contract holds through make_template and matchTemplate."""
    rng = np.random.default_rng(5)
    ref = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)
    sea = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)
    for mode in ("anscombe", "gat"):
        monkeypatch.setenv("DRIFTSENSE_VST_COARSE", mode)
        cands = pose_candidates(ref, sea, k=2)
        assert cands and all(len(c) == 3 for c in cands)
        assert all(np.isfinite(c[0]) and np.isfinite(c[1]) for c in cands)

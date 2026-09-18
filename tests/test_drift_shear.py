"""Raster-shear estimation and correction (issue #101).

Two kinds of test here, and the split is deliberate.

The **synthetic** tests build a periodic frame, shear it with the organizer's
own formula (``src/sem_imaging.apply_raster_drift``: sample at ``x + s``, so
content moves ``-s``) and assert the measurement responds. They pin the
physics and the sign, which is the part that silently inverts under a
refactor.

The **contract** tests pin the parts that protect the output: the null
control, the gate, the decline paths, and the fact that ``correct_x`` moves x
and nothing else. Those matter more than the accuracy numbers, which live in
``docs/PHASE3_RASTER_SHEAR.md`` because they need the generator to reproduce.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

cv2 = pytest.importorskip("cv2")

from driftsense import drift_shear                      # noqa: E402
from driftsense.matching import make_template           # noqa: E402

ROWS = 999.0


# --------------------------------------------------------------------------
# helpers

def _lattice(h=600, w=600, pitch_x=17, pitch_y=23, seed=7):
    """A periodic Manhattan frame: the structure this estimator lives on."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 40, np.float32)
    img[:, ::pitch_x] = 200
    img[::pitch_y, :] = 170
    # A little aperiodic texture, so correlation peaks are not exactly equal
    # and the site ranking is not a coin flip.
    img += rng.normal(0, 4, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _shear(img, amplitude, rows=None):
    """The generator's own raster drift, without the jitter.

    `rows` defaults to the image's own `h - 1`, matching what `measure` assumes
    when it is not told otherwise -- so an amplitude here means the same number
    of pixels it will read back.
    """
    h, w = img.shape
    if rows is None:
        rows = float(h - 1)
    shift = (amplitude * np.arange(h) / rows).astype(np.float32)
    map_x = np.arange(w, dtype=np.float32)[None, :] + shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _measurement(value, ok=True):
    return drift_shear.ShearMeasurement(value, 8, 8, 500.0, 6, ok, "")


# --------------------------------------------------------------------------
# correct_x: the only thing that touches a reported answer

def test_correct_x_adds_the_drift_back():
    # Content at row y was imaged A*y/rows to the LEFT, so the pre-drift x the
    # grader compares against is to the right.
    assert drift_shear.correct_x(100.0, ROWS, 3.0, ROWS) == pytest.approx(103.0)
    assert drift_shear.correct_x(100.0, 0.0, 3.0, ROWS) == pytest.approx(100.0)
    assert drift_shear.correct_x(100.0, ROWS / 2, 3.0, ROWS) == pytest.approx(101.5)


def test_correct_x_is_a_no_op_without_an_amplitude():
    for amp in (0.0, float("nan"), float("inf") * 0):
        assert drift_shear.correct_x(50.0, 700.0, amp, ROWS) == 50.0
    # A degenerate frame height must not divide by zero or invent a shift.
    assert drift_shear.correct_x(50.0, 700.0, 3.0, 0.0) == 50.0


def test_correct_x_scales_with_the_frame_not_a_literal():
    """The amplitude is defined against the frame's own row count."""
    assert drift_shear.correct_x(0.0, 100.0, 4.0, 199.0) == pytest.approx(2.0101, abs=1e-4)
    assert drift_shear.correct_x(0.0, 100.0, 4.0, 999.0) == pytest.approx(0.4004, abs=1e-4)


# --------------------------------------------------------------------------
# pool: the gate, the decline paths, and the null control

def test_pool_declines_below_the_minimum_batch():
    ms = [_measurement(3.0) for _ in range(drift_shear.MIN_POOL_PAIRS - 1)]
    corr = drift_shear.pool(ms)
    assert not corr.applied and corr.amplitude == 0.0
    assert "measured pair" in corr.reason


def test_pool_ignores_unmeasured_pairs():
    ok = [_measurement(3.0) for _ in range(drift_shear.MIN_POOL_PAIRS)]
    junk = [_measurement(99.0, ok=False) for _ in range(40)]
    junk += [drift_shear.NOT_MEASURED] * 10
    corr = drift_shear.pool(ok + junk)
    assert corr.n_pairs == drift_shear.MIN_POOL_PAIRS
    assert corr.batch_median == pytest.approx(3.0)


def test_pool_gate_is_what_makes_the_null_control_exact():
    """A batch reading near zero must produce EXACTLY no correction.

    This is the control docs/PHASE3_MEASUREMENT.md records this repo being
    burned twice for skipping, so it is a test and not a comment.
    """
    near_zero = drift_shear.SHEAR_OFFSET + 0.4 * drift_shear.SHEAR_GAIN
    ms = [_measurement(near_zero) for _ in range(40)]
    corr = drift_shear.pool(ms)
    assert not corr.applied
    assert corr.amplitude == 0.0            # exactly, not approximately
    assert abs(corr.estimate) < drift_shear.SHEAR_GATE
    # ... and applying it moves nothing at all.
    for y in (0.0, 500.0, 999.0):
        assert drift_shear.correct_x(123.456, y, corr.amplitude, ROWS) == 123.456


def test_pool_applies_above_the_gate_and_inverts_the_calibration():
    for truth in (2.0, 3.0, 4.0):
        raw = drift_shear.SHEAR_GAIN * truth + drift_shear.SHEAR_OFFSET
        corr = drift_shear.pool([_measurement(raw) for _ in range(40)])
        assert corr.applied
        assert corr.amplitude == pytest.approx(truth, abs=1e-6)


def test_pool_uses_the_median_so_wrong_basin_pairs_cannot_carry_it():
    raw = drift_shear.SHEAR_GAIN * 3.0 + drift_shear.SHEAR_OFFSET
    ms = [_measurement(raw) for _ in range(30)]
    ms += [_measurement(v) for v in (-500.0, 500.0, 900.0, -900.0)]
    corr = drift_shear.pool(ms)
    assert corr.amplitude == pytest.approx(3.0, abs=1e-6)


def test_pool_clips_a_pathological_batch():
    corr = drift_shear.pool([_measurement(1e6) for _ in range(40)])
    assert corr.amplitude <= drift_shear.A_BOUNDS[1]


def test_pool_refuses_a_degenerate_calibration():
    corr = drift_shear.pool([_measurement(3.0) for _ in range(40)], gain=0.0)
    assert not corr.applied and corr.amplitude == 0.0


# --------------------------------------------------------------------------
# the measurement itself: sign and response

def test_site_slope_sign_matches_the_generator():
    """A positive amplitude must read as a NEGATIVE per-row slope.

    `measure` negates it, so an inverted sign here would make the correction
    double the bias instead of removing it -- the one failure mode that looks
    like a working change.
    """
    base = _lattice()
    tpl = base[250:350, 250:350].copy()
    flat = drift_shear.site_slope(base, tpl, 300.0, 300.0)
    drifted = drift_shear.site_slope(_shear(base, 4.0), tpl, 300.0, 300.0)
    assert flat is not None and drifted is not None
    assert flat[0] == pytest.approx(0.0, abs=2e-3)
    assert drifted[0] < flat[0] - 1e-3


def test_measure_responds_monotonically_to_the_amplitude():
    """Gain is calibrated, so the contract tested here is the ORDERING."""
    base = _lattice()
    tpl = base[250:350, 250:350].copy()
    vals = []
    for amp in (0.0, 2.0, 4.0):
        m = drift_shear.measure(_shear(base, amp), tpl, centre=(300.0, 300.0))
        assert m.ok, m.reason
        vals.append(m.value)
    assert vals[0] < vals[1] < vals[2]


def test_measure_declines_on_a_frame_with_no_structure():
    blank = np.full((400, 400), 128, np.uint8)
    tpl = np.full((100, 100), 128, np.uint8)
    m = drift_shear.measure(blank, tpl)
    assert not m.ok
    assert np.isnan(m.value)


def test_measure_declines_when_the_template_does_not_fit():
    m = drift_shear.measure(np.zeros((50, 50), np.uint8),
                            np.zeros((100, 100), np.uint8))
    assert not m.ok and m.reason == "no sites"


def test_alias_sites_respects_the_radius_around_the_reported_centre():
    base = _lattice()
    tpl = base[250:350, 250:350].copy()
    wide = drift_shear.alias_sites(base, tpl)
    near = drift_shear.alias_sites(base, tpl, centre=(300.0, 300.0), radius=60.0)
    assert len(near) <= len(wide)
    for cx, cy, _v in near:
        assert abs(cx - 300.0) <= 60.0 and abs(cy - 300.0) <= 60.0


def test_alias_sites_spreads_over_row_bands():
    """Without the per-band cap the strongest peaks cluster in one mat, and a
    100-row measurement wears a 1000-row hat."""
    base = _lattice(h=600, w=600)
    tpl = base[250:350, 250:350].copy()
    sites = drift_shear.alias_sites(base, tpl, per_band=2, band_px=100)
    bands = [int(cy // 100) for _cx, cy, _v in sites]
    assert len(sites) > 0
    assert all(bands.count(b) <= 2 for b in set(bands))


# --------------------------------------------------------------------------
# end to end on a synthetic pair, through the posed-template path

def test_measure_uses_the_posed_template_path():
    """`make_template` output is what the entry point feeds in, so exercise
    that shape rather than a hand-cropped patch."""
    ref = cv2.resize(_lattice(h=300, w=300), (1000, 1000),
                     interpolation=cv2.INTER_NEAREST)
    tpl = make_template(ref, 10.0, 0.0)
    assert tpl.shape == (100, 100)
    frame = np.zeros((600, 600), np.uint8)
    frame[:] = np.tile(cv2.resize(ref, (100, 100), interpolation=cv2.INTER_AREA),
                       (6, 6))
    flat = drift_shear.measure(frame, tpl)
    sheared = drift_shear.measure(_shear(frame, 4.0), tpl)
    assert flat.ok and sheared.ok, (flat.reason, sheared.reason)
    # `measure` already negates the slope, so a positive amplitude reads
    # positive -- the same direction `pool` then inverts through the
    # calibration line.
    assert sheared.value > flat.value

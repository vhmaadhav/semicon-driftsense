"""Drift-immune rotation from vertical strip offsets (issue #88).

Pins the claim the stage rests on -- raster drift corrupts the horizontal half
of the rotation signal and leaves the vertical half alone, so a strip
regression is unbiased where a 2-D correlation fit is not -- plus the guards
and the wiring.
"""
import cv2
import numpy as np
import pytest

from driftsense.matching import (
    STRIP_ROT_SIGMA_PRIOR,
    destreak,
    make_template,
    polish_pose,
    strip_offsets,
    strip_rotation,
)

SCALE = 10.0
CENTRE = 150.0


def _texture(seed):
    rng = np.random.default_rng(seed)
    img = cv2.GaussianBlur(rng.normal(128, 60, (1000, 1000)).astype(np.float32), (0, 0), 7)
    img = (img - img.mean()) / img.std() * 40 + 128
    return np.clip(img, 0, 255).astype(np.uint8)


def _scene(ref, theta=0.0, shear_px_per_row=0.0, size=300):
    """`ref` posed at (CENTRE, CENTRE) in a flat frame, then raster-sheared.

    The shear is applied the way the imaging model applies it: each scan row is
    displaced horizontally, and nothing moves vertically.
    """
    tpl = make_template(ref, SCALE, theta)
    th, tw = tpl.shape
    frame = np.full((size, size), 128, np.float32)
    y0, x0 = int(CENTRE - th / 2), int(CENTRE - tw / 2)
    frame[y0:y0 + th, x0:x0 + tw] = tpl
    if shear_px_per_row:
        rows = np.arange(size, dtype=np.float32)[:, None]
        map_x = np.tile(np.arange(size, dtype=np.float32)[None, :], (size, 1)) \
            + shear_px_per_row * rows
        map_y = np.tile(rows, (1, size))
        frame = cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    return np.clip(frame, 0, 255).astype(np.uint8)


def _raw(ref, frame, theta_in, **kw):
    """The strip estimate itself: an unbounded prior disables the blend."""
    got = strip_rotation(ref, frame, CENTRE, CENTRE, SCALE, theta_in,
                         sigma_prior=1e9, max_delta=5.0, **kw)
    assert got is not None
    return got[0]


def test_strip_regression_recovers_a_rotation_error():
    """A template posed at the wrong angle: the strips see the residual."""
    ref = _texture(3)
    frame = _scene(ref, theta=0.0)
    for wrong in (-0.30, 0.30):
        assert abs(_raw(ref, frame, wrong)) < 0.5 * abs(wrong)


def test_raster_shear_biases_the_2d_fit_but_not_the_strip_regression():
    """The whole reason the stage exists.

    A pure horizontal shear is not a rotation, but it looks like one to a fit
    that uses both displacement components. The strip regression reads only the
    vertical component, which the shear does not touch.
    """
    ref = _texture(5)
    shear = 0.003                      # px per row: v2's 3 px over 1000 rows
    frame = _scene(ref, theta=0.0, shear_px_per_row=shear)
    apparent = abs(np.degrees(np.arctan(shear)))    # ~0.17 deg

    _, polished, _ = polish_pose(ref, frame, CENTRE, CENTRE, SCALE, 0.0)
    strip = _raw(ref, frame, 0.0)

    assert abs(polished) > 0.4 * apparent, (
        "the 2-D fit is expected to inherit the shear as apparent rotation; "
        f"got {polished:.3f} deg against an apparent {apparent:.3f} deg")
    assert abs(strip) < abs(polished), (
        f"strip regression {strip:.3f} deg should beat the 2-D fit "
        f"{polished:.3f} deg on a sheared frame")


def test_blend_shrinks_a_noisy_correction_toward_the_polish_estimate():
    """Inverse-variance blending, not replacement: the returned angle sits
    between the input and the raw regression, nearer the input the noisier the
    regression is."""
    ref = _texture(7)
    frame = _scene(ref, theta=0.0)
    theta_in = 0.30
    raw = _raw(ref, frame, theta_in)
    got = strip_rotation(ref, frame, CENTRE, CENTRE, SCALE, theta_in)
    assert got is not None
    blended, sigma = got
    assert sigma > 0
    lo, hi = sorted((theta_in, raw))
    assert lo <= blended <= hi
    expected = STRIP_ROT_SIGMA_PRIOR ** 2 / (STRIP_ROT_SIGMA_PRIOR ** 2 + sigma ** 2)
    assert blended == pytest.approx(theta_in + expected * (raw - theta_in), abs=1e-9)


def test_declines_on_featureless_strips_and_on_a_runaway():
    ref = _texture(11)
    flat = np.full((300, 300), 128, np.uint8)
    assert strip_rotation(ref, flat, CENTRE, CENTRE, SCALE, 0.0) is None

    frame = _scene(ref, theta=0.0)
    # A correction larger than max_delta is a mis-measurement, not a residual.
    assert strip_rotation(ref, frame, CENTRE, CENTRE, SCALE, 3.0,
                          max_delta=0.05) is None


def test_declines_when_the_window_leaves_the_frame():
    ref = _texture(13)
    frame = _scene(ref, theta=0.0)
    assert strip_rotation(ref, frame, 10.0, 10.0, SCALE, 0.0) is None


def test_destreak_removes_a_full_width_row_and_barely_touches_the_rest():
    """It subtracts positive row-level outliers, so the streak goes and the
    rest of the frame keeps its own values (a row that is itself an outlier is
    shaved by its own small excursion, which is the same rule, not a leak)."""
    rng = np.random.default_rng(2)
    img = rng.integers(90, 140, (200, 200)).astype(np.uint8)
    dirty = img.copy()
    dirty[70] = np.clip(dirty[70].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    cleaned = destreak(dirty)

    assert abs(int(np.median(cleaned[70])) - int(np.median(img[70]))) <= 2
    others = np.r_[0:70, 71:200]
    removed = dirty[others].astype(int) - cleaned[others].astype(int)
    assert (np.abs(removed) <= 10).all()
    assert (removed.any(axis=1)).sum() <= 5


def test_destreak_is_a_no_op_on_a_clean_frame():
    rng = np.random.default_rng(4)
    img = rng.integers(90, 140, (120, 120)).astype(np.uint8)
    assert np.array_equal(destreak(img), img)


def test_strip_offsets_measures_a_known_vertical_displacement():
    """The measurement primitive: shift the scene down, read the shift back."""
    ref = _texture(17)
    frame = _scene(ref, theta=0.0)
    tpl = make_template(ref, SCALE, 0.0)
    shifted = np.roll(frame, 2, axis=0)
    u, dy, peak = strip_offsets(shifted, tpl, CENTRE, CENTRE)
    assert len(u) >= 4 and (peak > 0.5).all()
    assert np.allclose(dy, 2.0, atol=0.25)
    assert np.allclose(u, np.sort(u))          # strips reported left to right


def test_locate_phase2_default_leaves_the_stage_off():
    """The signature default is the historical decode; register.py opts in via
    driftsense.config (tests/test_submission_parity.py pins the forwarding)."""
    import inspect

    from driftsense.matching import locate_phase2
    assert inspect.signature(locate_phase2).parameters["strip_rot"].default is False

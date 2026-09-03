"""Coordinate-frame invariants.

Every bug this file guards against is silent: the pipeline still returns a
plausible (x, y) and only the *accuracy* moves, which is exactly the class of
defect that survives eyeballing and shows up as an unexplained score drop.
"""

import cv2
import numpy as np
import pytest

from driftsense.matching import (
    _dihedral_img, _dihedral_point_inv, make_template, parabolic,
    refine_zncc, response_to_center, select_peak, standardize,
)
from driftsense.model import SCALE, STRIDE, TEMPLATE_SIZE


SHAPES = [(1000, 1000), (24, 40), (40, 24), (37, 64), (64, 37)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("t", range(8))
def test_dihedral_point_inverse_round_trips(shape, t):
    """A point mapped through a view and back must land where it started.

    Regression test: the inverse used to take a single `size` for both axes,
    which is only correct on square frames. The four odd-k rotations returned
    silently wrong coordinates on anything else -- and TTA maps *every*
    proposal through this.
    """
    h, w = shape
    for (y, x) in [(0, 0), (h - 1, w - 1), (h // 2, w // 3), (1, w - 2), (h - 2, 1)]:
        img = np.zeros((h, w), np.uint8)
        img[y, x] = 255

        transformed = _dihedral_img(img, t)
        ty, tx = (int(v) for v in np.argwhere(transformed == 255)[0])
        back_x, back_y = _dihedral_point_inv(float(tx), float(ty), (h, w), t)

        assert (round(back_x), round(back_y)) == (x, y), (
            f"t={t} shape={shape}: ({x},{y}) -> view ({tx},{ty}) -> ({back_x},{back_y})"
        )


@pytest.mark.parametrize("t", range(8))
def test_dihedral_img_preserves_content(t):
    img = np.random.default_rng(0).integers(0, 255, (32, 48), dtype=np.uint8)
    out = _dihedral_img(img, t)
    assert sorted(out.ravel()) == sorted(img.ravel())
    assert set(out.shape) == set(img.shape)


def test_make_template_downsamples_by_scale():
    """The reference is 1 nm/px and the search 10 nm/px, so the reference's
    footprint in the search frame is exactly 1/SCALE of its pixel size."""
    ref = np.random.default_rng(1).integers(0, 255, (1000, 1000), dtype=np.uint8)
    tpl = make_template(ref)
    assert tpl.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE) == (1000 // SCALE, 1000 // SCALE)


def test_make_template_handles_non_1000px_reference():
    ref = np.random.default_rng(2).integers(0, 255, (640, 800), dtype=np.uint8)
    assert make_template(ref).shape == (64, 80)


def test_response_to_center_matches_the_documented_convention():
    """Cell (i, j) means 'template top-left at search pixel (STRIDE*j,
    STRIDE*i)', so the centre is that plus half the template."""
    x, y = response_to_center(i=3, j=7, th=100, tw=100)
    assert (x, y) == (7 * STRIDE + 50.0, 3 * STRIDE + 50.0)

    # sub-cell offsets are in units of the stride
    x, y = response_to_center(i=3, j=7, th=100, tw=100, dy=0.5, dx=-0.25)
    assert x == (7 - 0.25) * STRIDE + 50.0
    assert y == (3 + 0.5) * STRIDE + 50.0


def test_parabolic_is_zero_on_a_symmetric_peak():
    assert parabolic(0.5, 1.0, 0.5) == 0.0
    assert parabolic(1.0, 1.0, 1.0) == 0.0          # degenerate -> no shift
    assert parabolic(0.9, 1.0, 0.5) < 0             # heavier on the left
    assert parabolic(0.5, 1.0, 0.9) > 0


def test_select_peak_breaks_ties_toward_the_centre():
    """The problem statement's rule: where several regions match equally well,
    return the one nearest the centre of the search image."""
    prob = np.zeros((100, 100), np.float32)
    prob[5, 5] = 0.90        # far corner
    prob[50, 50] = 0.895     # near centre, within the 4% tie tolerance
    i, j, _ = select_peak(prob, search_hw=(1000, 1000), template_hw=(100, 100))
    assert (i, j) == (50, 50)


def test_select_peak_takes_the_strongest_when_not_tied():
    prob = np.zeros((100, 100), np.float32)
    prob[5, 5] = 0.90
    prob[50, 50] = 0.50      # well outside the tie tolerance
    i, j, _ = select_peak(prob, search_hw=(1000, 1000), template_hw=(100, 100))
    assert (i, j) == (5, 5)


def test_refine_zncc_recovers_a_planted_offset():
    """Plant a template at a known location and check the snap finds it to
    sub-pixel. This is the stage that carries the final precision."""
    rng = np.random.default_rng(3)
    search = rng.normal(0, 1, (400, 400)).astype(np.float32)
    tpl = search[150:250, 120:220].copy()          # true centre (170, 200)

    true_cx, true_cy = 120 + 50.0, 150 + 50.0
    # start 3 px off, inside the +/-4 px search radius
    rx, ry, score = refine_zncc(search, tpl, true_cx + 3, true_cy - 2, radius=4)

    assert abs(rx - true_cx) < 0.05
    assert abs(ry - true_cy) < 0.05
    assert score > 0.99


def test_refine_zncc_declines_gracefully_at_the_frame_edge():
    """Too little room for the window: return the input unchanged rather than
    crashing or wrapping."""
    search = np.zeros((100, 100), np.float32)
    tpl = np.zeros((100, 100), np.float32)
    rx, ry, score = refine_zncc(search, tpl, 50.0, 50.0, radius=4)
    assert (rx, ry, score) == (50.0, 50.0, 0.0)


def test_standardize_is_zero_mean_unit_variance():
    x = np.random.default_rng(4).integers(0, 255, (50, 50)).astype(np.uint8)
    z = standardize(x)
    assert abs(float(z.mean())) < 1e-5
    assert abs(float(z.std()) - 1.0) < 1e-3


def test_standardize_survives_a_constant_image():
    z = standardize(np.full((10, 10), 128, np.uint8))
    assert np.isfinite(z).all()


# --- reference-resolution robustness ---------------------------------------
#
# The graders run infer.py on their own pairs with no manual edits, so the
# reference may not arrive at the 1000 px the local generator emits. A fixed
# /10 downsample turns a 100 px reference into a 10x10 template and the match
# becomes noise -- silently, with a plausible-looking coordinate returned.

from driftsense.matching import choose_factor, template_hypotheses


@pytest.mark.parametrize("ref_px,expected", [
    (1000, 10.0),   # 1 nm/px reference: the usual case, one hypothesis
    (500, 5.0),
    (200, 2.0),
    (100, 1.0),     # reference already at the search's pixel size
])
def test_template_hypotheses_cover_the_footprint_reading(ref_px, expected):
    ref = np.zeros((ref_px, ref_px), np.uint8)
    assert expected in template_hypotheses(ref)


def test_the_usual_reference_size_costs_no_extra_hypothesis():
    """1000 px: both readings agree, so nothing extra is evaluated."""
    assert template_hypotheses(np.zeros((1000, 1000), np.uint8)) == [10.0]


def test_hypotheses_never_upscale_the_reference():
    """A reference smaller than its own footprint must not be blown up."""
    assert all(f >= 1.0 for f in template_hypotheses(np.zeros((40, 40), np.uint8)))


@pytest.mark.parametrize("ref_px", [1000, 500, 200, 100])
def test_choose_factor_recovers_a_planted_pattern_at_any_reference_size(ref_px):
    """Plant a known 100x100 footprint, hand the reference over at four
    different resolutions, and check the chosen factor reproduces it."""
    rng = np.random.default_rng(21)
    search = cv2.GaussianBlur(
        rng.integers(0, 255, (1000, 1000), dtype=np.uint8), (0, 0), 1.5)
    footprint = search[300:400, 500:600]
    reference = cv2.resize(footprint, (ref_px, ref_px), interpolation=cv2.INTER_NEAREST)

    factor = choose_factor(reference, search)
    tpl = make_template(reference, factor)
    assert tpl.shape == (100, 100), f"ref {ref_px}px -> template {tpl.shape}"


def test_make_template_still_defaults_to_the_fixed_ratio():
    """The default must stay backwards-compatible: /SCALE, as before."""
    ref = np.zeros((640, 800), np.uint8)
    assert make_template(ref).shape == (64, 80)
    assert make_template(ref, factor=8.0).shape == (80, 100)

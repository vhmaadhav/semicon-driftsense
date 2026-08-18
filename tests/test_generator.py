"""Dataset-generator degradations named by the problem statement.

The spec requires the generator to model independent sensor noise,
edge-brightening, blur, rotation and scaling variations. The first two of
those five were absent; these tests pin the additions down, and pin the
default path to its previous byte-for-byte behaviour so the shipped splits
stay reproducible.
"""

import cv2
import numpy as np
import pytest

from driftsense.generate import (
    BOX_PX, PoseParams, SEARCH_SIZE_PX, apply_affine_point,
    apply_edge_brightening, image_search_traced, search_affine,
)
from src.pipeline import GenerationParams

CLEAN = dict(dose_search=1e9, detector_noise_sigma_search=0.0,
             beam_spot_size_nm=0.1, shear_amplitude_px=0.0,
             drift_jitter_px=0.0, barrel_distortion_k=0.0)


def _canvas(size=4000, marker=(1200, 900), r=40):
    img = np.zeros((size, size), np.uint8)
    cv2.circle(img, marker, r, 255, -1)
    return img


def _blob_centroid(img):
    ys, xs = np.nonzero(img > img.max() * 0.4)
    w = img[ys, xs].astype(np.float64)
    return float(np.average(xs, weights=w)), float(np.average(ys, weights=w))


# --- edge brightening ------------------------------------------------------

def test_edge_brightening_is_a_no_op_at_zero():
    img = np.random.default_rng(0).integers(0, 255, (64, 64), dtype=np.uint8)
    assert np.array_equal(apply_edge_brightening(img, 0.0), img)


def test_edge_brightening_leaves_a_flat_field_alone():
    """No gradient, no secondary-electron edge response."""
    flat = np.full((64, 64), 120, np.uint8)
    assert np.array_equal(apply_edge_brightening(flat, 0.5), flat)


def test_edge_brightening_lifts_edges_above_their_neighbours():
    step = np.zeros((64, 64), np.uint8)
    step[:, 32:] = 200
    out = apply_edge_brightening(step, 0.5)
    edge, interior = int(out[32, 31]), int(out[32, 5])
    assert edge > int(step[32, 31])
    assert edge > interior


def test_edge_brightening_cannot_overflow():
    bright = np.full((32, 32), 250, np.uint8)
    bright[:, 16:] = 0
    out = apply_edge_brightening(bright, 5.0)
    assert out.dtype == np.uint8 and out.max() <= 255


# --- the affine: rotation and magnification --------------------------------

def test_affine_maps_canvas_centre_to_search_centre():
    for mag, rot in [(10.0, 0.0), (9.0, 2.0), (11.0, -2.0)]:
        M = search_affine(4000, SEARCH_SIZE_PX, mag, rot)
        x, y = apply_affine_point(M, 1999.5, 1999.5)
        assert abs(x - 499.5) < 1e-6 and abs(y - 499.5) < 1e-6


@pytest.mark.parametrize("mag,rot", [(9.0, 0.0), (11.0, 0.0), (10.0, 2.0),
                                     (10.0, -2.0), (9.5, 1.5), (11.0, -2.0)])
def test_posed_label_lands_where_the_pattern_renders(mag, rot):
    """The matrix that renders the frame is the matrix that maps the label, so
    the two must agree to well under a pixel."""
    C, marker = 4000, (1200, 900)
    p = GenerationParams(**CLEAN)
    img, _, _ = image_search_traced(_canvas(C, marker), p,
                                    np.random.default_rng(0),
                                    PoseParams(rotation_deg=rot, magnification=mag))
    lx, ly = apply_affine_point(search_affine(C, SEARCH_SIZE_PX, mag, rot), *marker)
    ax, ay = _blob_centroid(img)
    assert np.hypot(ax - lx, ay - ly) < 0.6, f"label ({lx},{ly}) vs actual ({ax},{ay})"


def test_posed_frames_are_always_the_declared_search_size():
    p = GenerationParams(**CLEAN)
    img, _, _ = image_search_traced(_canvas(), p, np.random.default_rng(0),
                                    PoseParams(rotation_deg=2.0, magnification=9.0))
    assert img.shape == (SEARCH_SIZE_PX, SEARCH_SIZE_PX)


# --- the default path must not move ----------------------------------------

def test_nominal_pose_reproduces_the_upstream_path_exactly():
    """PoseParams() must be indistinguishable from passing no pose at all --
    this is what keeps every previously generated split reproducible."""
    p = GenerationParams(**CLEAN)
    a, ra, ka = image_search_traced(_canvas(), p, np.random.default_rng(3))
    b, rb, kb = image_search_traced(_canvas(), p, np.random.default_rng(3), PoseParams())
    assert np.array_equal(a, b) and np.array_equal(ra, rb) and ka == kb


def test_pose_actually_changes_the_image():
    p = GenerationParams(**CLEAN)
    a, _, _ = image_search_traced(_canvas(), p, np.random.default_rng(3))
    b, _, _ = image_search_traced(_canvas(), p, np.random.default_rng(3),
                                  PoseParams(rotation_deg=2.0, magnification=9.0))
    assert not np.array_equal(a, b)

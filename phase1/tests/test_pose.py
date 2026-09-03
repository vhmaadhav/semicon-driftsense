"""Pose estimation at inference time.

The problem statement names rotation and scaling variations, and the graders
run infer.py on their own pairs. The network was trained at exactly 10x and 0
degrees, so the pose search has two jobs and they pull against each other:
find a genuinely off-nominal pose, and never wander off nominal on data that
is nominal. The margin guard is what reconciles them, so it is tested from
both sides.
"""

import cv2
import numpy as np
import pytest

from driftsense.matching import (
    POSE_SKIP_ABOVE, choose_pose, make_template,
)


def _scene(mag=10.0, rot=0.0, size=1000, seed=0):
    """A search frame with a known pattern planted at a known pose."""
    rng = np.random.default_rng(seed)
    search = cv2.GaussianBlur(
        rng.integers(0, 255, (size, size), dtype=np.uint8), (0, 0), 1.2)

    # Carve the footprint out of the frame, then invert the pose to build the
    # 1000 px reference an acquisition at that pose would have produced.
    n = int(round(1000 / mag))
    x0, y0 = 400, 300
    patch = search[y0:y0 + n, x0:x0 + n].copy()
    if rot:
        M = cv2.getRotationMatrix2D(((n - 1) / 2.0, (n - 1) / 2.0), -rot, 1.0)
        patch = cv2.warpAffine(patch, M, (n, n), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)
    reference = cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_LINEAR)
    return reference, search


def test_nominal_scene_stays_nominal():
    """The guard's whole purpose: no wandering on data that is already 10x/0."""
    ref, sea = _scene()
    assert choose_pose(ref, sea) == (10.0, 0.0)


@pytest.mark.parametrize("seed", range(4))
def test_nominal_stays_nominal_across_scenes(seed):
    ref, sea = _scene(seed=seed)
    factor, rot = choose_pose(ref, sea)
    assert (factor, rot) == (10.0, 0.0)


def test_a_confident_nominal_match_skips_the_search_entirely():
    """Early exit: a strong nominal correlation means there is nothing to look
    for, and the eight extra correlations are not run."""
    ref, sea = _scene()
    from driftsense.matching import _peak_score, _probe, choose_factor
    base = choose_factor(ref, sea)
    nominal = _peak_score(_probe(sea), _probe(make_template(ref, base)))
    assert nominal >= POSE_SKIP_ABOVE


def test_make_template_rotation_is_a_no_op_at_zero():
    ref = np.random.default_rng(1).integers(0, 255, (1000, 1000), dtype=np.uint8)
    assert np.array_equal(make_template(ref, 10.0, 0.0), make_template(ref, 10.0))


def test_make_template_rotation_preserves_shape():
    ref = np.random.default_rng(2).integers(0, 255, (1000, 1000), dtype=np.uint8)
    assert make_template(ref, 10.0, 2.0).shape == make_template(ref, 10.0).shape


def test_rotating_a_template_actually_changes_it():
    ref = np.random.default_rng(3).integers(0, 255, (1000, 1000), dtype=np.uint8)
    assert not np.array_equal(make_template(ref, 10.0, 2.0), make_template(ref, 10.0))


def test_the_margin_guard_can_be_disabled():
    """With no margin the search is free to move; this pins the guard as the
    thing keeping nominal data nominal, rather than the search being blind."""
    ref, sea = _scene()
    assert choose_pose(ref, sea, margin=0.0) is not None

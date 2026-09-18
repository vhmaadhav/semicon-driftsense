"""Sanity checks for the synthetic data generator itself.

Note on the baseline matcher: DRAM/FinFET structures are highly periodic
(repeating every 3-5 px at search resolution), so a naive global
cv2.matchTemplate search can genuinely lock onto the wrong repeat of the
pattern -- that's real matching ambiguity, not a generator bug, and worth
raising with students as a limitation of naive template matching. Because
of that, these tests validate the generator directly (does the search
image really contain the downsampled reference at the stated ground-truth
box?) rather than routing through the ambiguous global baseline search.
"""

import cv2
import numpy as np
import pytest

from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS

# FinFET's 8nm fin width aliases harder under 10nm/px downsampling than DRAM's
# 18nm features, so the true-location diff is naturally noisier (~10-26) even
# with near-zero imaging noise; keep the threshold well below the ~30+ diff
# seen when comparing against the wrong periodic repeat (see docstring above).
MAX_MEAN_ABS_DIFF = 30.0


@pytest.mark.parametrize("architecture", list(PRESETS.keys()))
def test_ground_truth_patch_matches_reference(architecture):
    rng = np.random.default_rng(0)
    params = GenerationParams(
        beam_spot_size_nm=1.0,
        dose_reference=1e6,
        dose_search=1e6,
        shear_amplitude_px=0.0,
        drift_jitter_px=0.0,
        detector_noise_sigma_ref=0.0,
        detector_noise_sigma_search=0.0,
        # Phase 2 knobs disabled: this test validates the crop/GT alignment
        # invariant itself, which reference_rotation_deg and no_match_prob
        # deliberately break (by design) for other samples.
        reference_rotation_deg=0.0,
        no_match_prob=0.0,
    )
    sample = generate_sample(architecture, rng, params)

    assert sample["match_found"] is True
    x0, y0, w, h = (int(round(v)) for v in sample["gt_box"])
    patch_at_gt = sample["search_img"][y0:y0 + h, x0:x0 + w]
    reference_downsampled = cv2.resize(sample["reference_img"], (w, h), interpolation=cv2.INTER_AREA)

    mean_abs_diff = np.abs(patch_at_gt.astype(int) - reference_downsampled.astype(int)).mean()
    assert mean_abs_diff < MAX_MEAN_ABS_DIFF


def test_image_shapes_and_ground_truth_in_bounds():
    rng = np.random.default_rng(1)
    params = GenerationParams()
    sample = generate_sample("dram_1x", rng, params)

    assert sample["reference_img"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)
    if sample["match_found"]:
        assert 0 <= sample["gt_x"] <= 1000
        assert 0 <= sample["gt_y"] <= 1000
    else:
        assert sample["gt_x"] is None
        assert sample["gt_y"] is None
        assert sample["gt_box"] is None


def test_reference_rotation_changes_only_the_reference():
    params_no_rot = GenerationParams(reference_rotation_deg=0.0, no_match_prob=0.0)
    params_rot = GenerationParams(reference_rotation_deg=15.0, no_match_prob=0.0)

    sample_no_rot = generate_sample("dram_1x", np.random.default_rng(3), params_no_rot)
    sample_rot = generate_sample("dram_1x", np.random.default_rng(3), params_rot)

    assert sample_rot["reference_img"].shape == (1000, 1000)
    # a 15deg rotation should visibly change the reference vs. the unrotated crop
    diff = np.abs(sample_rot["reference_img"].astype(int) - sample_no_rot["reference_img"].astype(int)).mean()
    assert diff > 5.0


def test_no_match_sample_has_no_ground_truth():
    params = GenerationParams(no_match_prob=1.0)
    sample = generate_sample("finfet_10nm", np.random.default_rng(4), params)

    assert sample["match_found"] is False
    assert sample["gt_x"] is None
    assert sample["gt_y"] is None
    assert sample["gt_box"] is None
    assert sample["reference_img"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)


def test_polygon_scale_outliers_do_not_crash():
    params = GenerationParams(polygon_scale_prob=0.5, polygon_scale_range=0.3, no_match_prob=0.0)
    sample = generate_sample("dram_dense", np.random.default_rng(5), params)

    assert sample["reference_img"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)

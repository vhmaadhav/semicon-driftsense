"""Sanity checks for the Phase 3 CAD-reference pipeline.

Mirrors tests/test_pipeline.py's approach: validate the generator directly
(does the Search image really contain the Reference CAD's footprint at the
stated ground truth?) rather than routing through a matcher.
"""

import cv2
import numpy as np
import pytest

from src.cad_pipeline import CadGenerationParams, generate_cad_sample, build_cad_geometry

MAX_MEAN_ABS_DIFF = 30.0


def _overlaps_any_strip(x0, y0, size, strip_rects):
    return any(
        x0 < sx + sw and x0 + size > sx and y0 < sy + sh and y0 + size > sy
        for sx, sy, sw, sh in strip_rects
    )


@pytest.mark.parametrize("kind", ["dram", "finfet"])
def test_gds_reference_matches_search_at_gt_box(kind):
    rng = np.random.default_rng(0)
    params = CadGenerationParams(no_match_prob=0.0)
    sample = generate_cad_sample(kind, rng, params)

    assert sample["match_found"] is True
    x0, y0, w, h = (int(round(v)) for v in sample["gt_box"])
    patch_at_gt = sample["search_img"][y0:y0 + h, x0:x0 + w]
    ref_downsampled = cv2.resize(sample["reference_preview"], (w, h), interpolation=cv2.INTER_AREA)

    mean_abs_diff = np.abs(patch_at_gt.astype(int) - ref_downsampled.astype(int)).mean()
    assert mean_abs_diff < MAX_MEAN_ABS_DIFF


def test_reference_gds_has_expected_layers():
    rng = np.random.default_rng(1)
    sample = generate_cad_sample("dram", rng, CadGenerationParams(no_match_prob=0.0))

    layers_present = {p.layer for p in sample["reference_cell"].get_polygons()}
    assert layers_present.issubset(set(range(sample["num_layers"])))
    assert len(layers_present) > 1  # a real multi-layer design, not a single flat layer


def test_no_match_sample_has_no_ground_truth():
    rng = np.random.default_rng(2)
    sample = generate_cad_sample("finfet", rng, CadGenerationParams(no_match_prob=1.0))

    assert sample["match_found"] is False
    assert sample["gt_x"] is None
    assert sample["gt_y"] is None
    assert sample["gt_box"] is None
    assert sample["reference_preview"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)


@pytest.mark.parametrize("kind", ["dram", "finfet"])
def test_reference_crop_always_straddles_a_separator(kind):
    for seed in range(5):
        rng = np.random.default_rng(seed)
        geom = build_cad_geometry(kind, rng, CadGenerationParams(no_match_prob=0.0))
        assert _overlaps_any_strip(geom["x0"], geom["y0"], 1000, geom["strip_rects"]), (
            f"seed={seed}: reference crop at ({geom['x0']}, {geom['y0']}) does not overlap any strip"
        )


def test_image_shapes():
    rng = np.random.default_rng(3)
    sample = generate_cad_sample("dram", rng, CadGenerationParams())
    assert sample["reference_preview"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)

"""The varied-parameter wrapper: recorded rotation is the one applied."""

import csv
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from generate_cad_varied import GUI_BANDS, draw_params, generate_sample
from src.cad_pipeline import FINE_CANVAS_SIZE_PX, PADDED_CANVAS_SIZE_PX, REFERENCE_SIZE_PX, SCALE_FACTOR


def _expected_gt(geometry, angle):
    """render_cad_sample's own gt mapping, for a given angle."""
    m = (PADDED_CANVAS_SIZE_PX - FINE_CANVAS_SIZE_PX) // 2
    c = PADDED_CANVAS_SIZE_PX / 2.0
    M = cv2.getRotationMatrix2D((c, c), angle, 1.0)
    fx = geometry["x0"] + REFERENCE_SIZE_PX / 2.0 + m
    fy = geometry["y0"] + REFERENCE_SIZE_PX / 2.0 + m
    return ((M[0, 0] * fx + M[0, 1] * fy + M[0, 2] - m) / SCALE_FACTOR,
            (M[1, 0] * fx + M[1, 1] * fy + M[1, 2] - m) / SCALE_FACTOR)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_recorded_rotation_reproduces_the_ground_truth(seed):
    rng = np.random.default_rng(seed)
    s = generate_sample("dram" if seed % 2 else "finfet", rng, max_rotation_deg=8.0, no_match_prob=0.0)
    assert s["match_found"]
    assert abs(s["gt_theta"]) <= s["params"]["search_rotation_deg"]
    ex, ey = _expected_gt(s["geometry"], s["gt_theta"])
    assert abs(ex - s["gt_x"]) < 1e-9 and abs(ey - s["gt_y"]) < 1e-9
    # And a perturbed angle does not: the check can actually fail.
    wx, wy = _expected_gt(s["geometry"], s["gt_theta"] + 0.05)
    assert max(abs(wx - s["gt_x"]), abs(wy - s["gt_y"])) > 1e-3


def test_draws_stay_inside_the_gui_bands():
    rng = np.random.default_rng(3)
    for _ in range(200):
        p = draw_params(rng, max_rotation_deg=8.0, no_match_prob=0.08).as_dict()
        for k, (lo, hi) in GUI_BANDS.items():
            assert lo <= p[k] <= hi, k


def test_absent_sample_has_zero_pose():
    rng = np.random.default_rng(4)
    s = generate_sample("dram", rng, max_rotation_deg=8.0, no_match_prob=1.0)
    assert not s["match_found"] and s["gt_theta"] == 0.0


def test_cli_writes_a_manifest_with_the_pose_columns(tmp_path):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "generate_cad_varied.py", "--num-samples", "2", "--split", "s",
                    "--output-dir", str(tmp_path), "--seed", "5"], cwd=here, check=True,
                   capture_output=True)
    rows = list(csv.DictReader(open(tmp_path / "s" / "manifest.csv")))
    assert len(rows) == 2 and {"gt_theta", "gt_scale", "search_rotation_deg"} <= set(rows[0])
    for r in rows:
        assert (tmp_path / "s" / r["reference_gds_path"]).exists()
        assert (tmp_path / "s" / r["search_path"]).exists()

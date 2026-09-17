"""Evaluation tooling for the Phase 2 v2 validation set (issue #85).

compare_phase2_v2.components re-implements the rubric in vectorised form for
the bootstrap, so it is pinned to the reference scripts/grade_emulation.rubric.
register_parallel.py must write exactly what one register.py process writes.
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.compare_phase2_v2 import components  # noqa: E402
from scripts.grade_emulation import LOC_TIERS, ROT_TIERS, SCALE_TIERS, rubric, tier  # noqa: E402


def _random_frame(seed, n=60):
    rng = np.random.default_rng(seed)
    sets = rng.choice(["A", "B", "C"], size=n, p=[0.4, 0.4, 0.2])
    present = (sets != "C").astype(int)
    gt_x, gt_y = rng.uniform(100, 900, n), rng.uniform(100, 900, n)
    err = rng.choice([0.3, 1.5, 2.5, 4.0, 7.0], size=n)
    ang = rng.uniform(0, 2 * np.pi, n)
    score = np.round(rng.uniform(0, 1, n), 2)              # rounding makes ties
    df = pd.DataFrame({
        "pair_id": [f"q{i:03d}" for i in range(n)], "set": sets, "gt_found": present, "score": score,
        "gt_x": gt_x, "gt_y": gt_y, "x": gt_x + err * np.cos(ang), "y": gt_y + err * np.sin(ang),
        "gt_scale": rng.uniform(8, 12, n), "gt_rot": rng.uniform(-5, 5, n)})
    df["scale"] = df.gt_scale * (1 + rng.choice([0.005, 0.015, 0.04, 0.08], size=n))
    df["theta"] = df.gt_rot + rng.choice([0.1, 0.4, 0.8, 1.5], size=n)
    p = df.gt_found == 1
    df["err"] = np.where(p, np.hypot(df.x - df.gt_x, df.y - df.gt_y), np.nan)
    df["loc_raw"] = [tier(e, LOC_TIERS) if q else 0.0 for e, q in zip(df.err, p)]
    df["sc_raw"] = [tier(abs(s - g) / g, SCALE_TIERS) if q else 0.0 for s, g, q in zip(df.scale, df.gt_scale, p)]
    df["rc_raw"] = [tier(abs(t - g), ROT_TIERS) if q else 0.0 for t, g, q in zip(df.theta, df.gt_rot, p)]
    return df


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("threshold", [0.0, 0.18, 0.5, 0.95])
def test_vectorised_rubric_matches_the_reference(seed, threshold):
    df = _random_frame(seed)
    ref = rubric(df, threshold)
    got = components(df, np.arange(len(df)), threshold)
    assert got["locA"] == pytest.approx(ref["loc_A"])
    assert got["locB"] == pytest.approx(ref["loc_B"])
    assert got["f1"] == pytest.approx(15 * ref["f1_reject"])
    assert got["auc"] == pytest.approx(10 * ref["auc"])
    if np.isfinite(ref["total"]):
        assert got["total"] == pytest.approx(ref["total"])
    else:
        assert not np.isfinite(got["total"])


def test_register_parallel_matches_a_single_process_run(tmp_path):
    rng = np.random.default_rng(9)
    lines = ["pair_id,reference_path,search_path"]
    for i in range(5):
        ref = cv2.GaussianBlur(rng.integers(0, 255, (1000, 1000), dtype=np.uint8), (0, 0), 6)
        frame = np.full((260, 260), 128, np.uint8)
        y, x = rng.integers(20, 140, 2)
        frame[y:y + 100, x:x + 100] = cv2.resize(ref, (100, 100), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(tmp_path / f"r{i}.png"), ref)
        cv2.imwrite(str(tmp_path / f"s{i}.png"), frame)
        lines.append(f"P{i},r{i}.png,s{i}.png")
    (tmp_path / "pairs.csv").write_text("\n".join(lines) + "\n")
    flags = ["--weights", str(tmp_path / "missing.pt"), "--allow-fallback", "--threshold", "0.0"]

    single = tmp_path / "single.csv"
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "register.py"), "--input", str(tmp_path / "pairs.csv"),
                    "--output", str(single), "--threads", "1", "--quiet", *flags],
                   check=True, capture_output=True, timeout=600)
    parallel = tmp_path / "out" / "parallel.csv"
    subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "register_parallel.py"),
                    "--input", str(tmp_path / "pairs.csv"), "--output", str(parallel),
                    "--jobs", "2", "--threads", "1", "--", *flags],
                   check=True, capture_output=True, timeout=600)
    assert parallel.read_text() == single.read_text()

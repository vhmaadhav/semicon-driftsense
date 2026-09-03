"""Regression coverage for issue #36: the classical no-weights fallback must
search the disclosed Phase 2 pose space and be scored on its own calibrated
threshold, not silently narrow to a Phase-1-shaped search or reuse the
network-calibrated SHIPPED_THRESHOLD on a raw ZNCC score.

Before this fix: `infer.zncc_fallback` searched only ~9x-11x with no
rotation search at all, then `register.py` hard-coded scale=10.0/theta=0.0
on top of that and applied SHIPPED_THRESHOLD (calibrated for the network's
statistic) to the raw ZNCC score -- so a packaging/runtime failure that
triggers this path could silently write confident, wrong, perfectly
formatted predictions across all five graded Phase 2 dimensions.
"""

import csv
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

CANVAS_PX = 9000


def _rotated_scaled_pair(z: float, theta: float):
    """A search frame with a real 1000x1000 canvas crop planted at a known
    (z, theta) pose -- same affine construction as
    tests/test_pose_rotation_ranking.py::_fixture (proven correct there),
    same procedural dram_1x canvas so the content actually survives the
    downsample (unlike plain blurred noise, which aliases badly under a
    ~z-to-1 resample and correlates with nothing)."""
    import cv2
    from generator.src.patterns.dram import generate_dram_canvas
    from generator.src.presets import get_preset

    rng = np.random.default_rng(26)
    canvas = generate_dram_canvas(CANVAS_PX, get_preset("dram_1x"), 10.0, rng)

    t = np.deg2rad(theta)
    A = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) / z
    M = np.zeros((2, 3))
    M[:, :2] = A
    M[:, 2] = np.array([500.0, 500.0]) - A @ np.array([4500.0, 4500.0])
    search = cv2.warpAffine(canvas, M, (1000, 1000), flags=cv2.INTER_LINEAR)
    reference = canvas[4000:5000, 4000:5000]
    return reference, search, (500.0, 500.0)


def test_zncc_fallback_recovers_off_grid_scale_and_rotation():
    """The old fallback searched ~9x-11x with no rotation at all. z=8.0 is
    the disclosed range's own lower endpoint, well outside that window, and
    theta=3.0 could not be found by a translation-only search regardless of
    scale -- both must now be recoverable since the fix searches the full
    disclosed pose space."""
    pytest.importorskip("cv2")
    import infer as I

    reference, search, (gx, gy) = _rotated_scaled_pair(z=8.0, theta=3.0)

    res = I.zncc_fallback(reference, search)
    assert np.hypot(res["x"] - gx, res["y"] - gy) < 5.0, (
        f"fallback missed the planted pose: got ({res['x']:.1f}, {res['y']:.1f}), "
        f"expected near ({gx:.1f}, {gy:.1f})")
    assert abs(res["scale"] - 8.0) <= 0.5, (
        f"scale {res['scale']} not close to the planted 8.0 -- fallback is "
        "not actually searching the disclosed [8,12] range")
    assert abs(res["theta"] - 3.0) <= 1.0, (
        f"theta {res['theta']} not close to the planted 3.0 -- fallback is "
        "not searching rotation at all")


def test_zncc_fallback_reports_real_pose_not_hardcoded_defaults():
    """Direct regression for the removed `res.setdefault("scale", 10.0)` /
    `res.setdefault("theta", 0.0)` in register.py: the dict zncc_fallback
    itself returns must already carry a real scale/theta key, not rely on a
    caller to paper over their absence with Phase-1-shaped defaults."""
    pytest.importorskip("cv2")
    import infer as I

    reference, search, _ = _rotated_scaled_pair(z=11.0, theta=-2.0)
    res = I.zncc_fallback(reference, search)
    assert "scale" in res and "theta" in res
    assert isinstance(res["scale"], float) and isinstance(res["theta"], float)


def test_legacy_fallback_threshold_is_a_distinct_calibrated_value():
    """The fallback's raw-ZNCC score and the network's calibrated statistic
    are different unit systems; gating both with the same SHIPPED_THRESHOLD
    (0.18) was the second half of issue #36."""
    from driftsense.config import LEGACY_FALLBACK_THRESHOLD, SHIPPED_THRESHOLD
    assert LEGACY_FALLBACK_THRESHOLD == pytest.approx(0.55)
    assert LEGACY_FALLBACK_THRESHOLD != SHIPPED_THRESHOLD


def test_register_uses_legacy_threshold_only_on_the_fallback_path(tmp_path, monkeypatch):
    """End-to-end: force register.main() onto the no-weights path and prove
    it is gated by LEGACY_FALLBACK_THRESHOLD, not the caller's --threshold
    (which the CLI help now documents as learned-path-only). Requires
    --allow-fallback explicitly, since the default (below) is now to fail
    closed on a model-load failure rather than decode with it."""
    pytest.importorskip("cv2")
    import cv2

    reference, search, _ = _rotated_scaled_pair(z=10.0, theta=0.0)
    rp, sp = str(tmp_path / "reference.png"), str(tmp_path / "search.png")
    cv2.imwrite(rp, reference)
    cv2.imwrite(sp, search)

    csv_path = tmp_path / "pairs.csv"
    out_path = tmp_path / "predictions.csv"
    csv_path.write_text(f"pair_id,reference,search\nP0001,{rp},{sp}\n")

    import register
    import infer as I

    monkeypatch.setattr(I, "load_model", lambda *a, **k: None)
    # An absurdly high fallback threshold must decline even a strong match,
    # proving register.py reads LEGACY_FALLBACK_THRESHOLD for this path
    # rather than the caller's (unset, so SHIPPED_THRESHOLD=0.18) --threshold.
    monkeypatch.setattr(register, "LEGACY_FALLBACK_THRESHOLD", 1.01)

    argv = ["register.py", "--input", str(csv_path), "--output", str(out_path),
            "--quiet", "--allow-fallback"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        register.main()
    finally:
        sys.argv = old_argv

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["found"]) == 0, (
        "an unreachable LEGACY_FALLBACK_THRESHOLD=1.01 should decline every "
        "pair on the fallback path -- found=1 means register.py is not "
        "actually reading this constant for the no-weights path")


def test_register_fails_closed_by_default_on_model_load_failure(tmp_path, monkeypatch):
    """The strategic fix: the organizer's required command
    (`register.py --input ... --output ...`, no --allow-fallback) must abort
    loudly rather than silently decode the batch with a materially different
    algorithm. predictions.csv must not exist afterward -- a partial/wrong
    file being present is itself a hazard if something downstream globs for
    it."""
    pytest.importorskip("cv2")
    import cv2

    reference, search, _ = _rotated_scaled_pair(z=10.0, theta=0.0)
    rp, sp = str(tmp_path / "reference.png"), str(tmp_path / "search.png")
    cv2.imwrite(rp, reference)
    cv2.imwrite(sp, search)

    csv_path = tmp_path / "pairs.csv"
    out_path = tmp_path / "predictions.csv"
    csv_path.write_text(f"pair_id,reference,search\nP0001,{rp},{sp}\n")

    import register
    import infer as I

    monkeypatch.setattr(I, "load_model", lambda *a, **k: None)

    argv = ["register.py", "--input", str(csv_path), "--output", str(out_path),
            "--quiet"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        with pytest.raises(SystemExit):
            register.main()
    finally:
        sys.argv = old_argv

    assert not out_path.exists(), (
        "register.py must fail BEFORE creating predictions.csv when the "
        "model can't load and --allow-fallback was not passed -- finding "
        "the file here means it fell through to the fallback silently")


def test_register_allow_fallback_flag_opts_back_into_the_old_behaviour(tmp_path, monkeypatch):
    """--allow-fallback is the explicit, local-only escape hatch: with it
    set, a model-load failure must still produce a complete predictions.csv
    via the classical fallback rather than aborting."""
    pytest.importorskip("cv2")
    import cv2

    reference, search, _ = _rotated_scaled_pair(z=10.0, theta=0.0)
    rp, sp = str(tmp_path / "reference.png"), str(tmp_path / "search.png")
    cv2.imwrite(rp, reference)
    cv2.imwrite(sp, search)

    csv_path = tmp_path / "pairs.csv"
    out_path = tmp_path / "predictions.csv"
    csv_path.write_text(f"pair_id,reference,search\nP0001,{rp},{sp}\n")

    import register
    import infer as I

    monkeypatch.setattr(I, "load_model", lambda *a, **k: None)

    argv = ["register.py", "--input", str(csv_path), "--output", str(out_path),
            "--quiet", "--allow-fallback"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        register.main()
    finally:
        sys.argv = old_argv

    with open(out_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1

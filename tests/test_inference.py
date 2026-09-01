"""Model and CLI contract.

`infer.py` is the deliverable interface: the graders run it, so its stdout
contract matters as much as the accuracy behind it.
"""

import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np
import pytest
import torch

from driftsense.matching import locate, locate_phase2, locate_tta, zncc_only
from driftsense.model import DriftSenseNet, net_from_checkpoint

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO_ROOT, "weights", "driftsense.pt")
INFER = os.path.join(REPO_ROOT, "infer.py")


@pytest.fixture(scope="module")
def model():
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    m = net_from_checkpoint(ckpt)
    m.load_state_dict(ckpt.get("model", ckpt))
    return m.eval()


@pytest.fixture(scope="module")
def pair(tmp_path_factory):
    """A search frame with the reference's own content planted in it, so the
    correct answer is known without invoking the generator."""
    rng = np.random.default_rng(11)
    search = rng.integers(40, 210, (1000, 1000), dtype=np.uint8)
    search = cv2.GaussianBlur(search, (0, 0), 2.0)

    # Take a 100x100 patch and treat it as the 10x-finer reference of that site.
    x0, y0 = 430, 260
    patch = search[y0:y0 + 100, x0:x0 + 100]
    reference = cv2.resize(patch, (1000, 1000), interpolation=cv2.INTER_NEAREST)

    d = tmp_path_factory.mktemp("pair")
    rp, sp = str(d / "reference.png"), str(d / "search.png")
    cv2.imwrite(rp, reference)
    cv2.imwrite(sp, search)
    return rp, sp, (x0 + 50.0, y0 + 50.0)


# --- model ----------------------------------------------------------------

def test_forward_pass_shapes(model):
    t = torch.zeros(1, 1, 25 * 4, 25 * 4)
    s = torch.zeros(1, 1, 1000, 1000)
    out = model(t, s)
    # valid correlation over a stride-4 encoding: (1000/4 - 100/4 + 1) = 226
    assert out["logit"].shape == (1, 1, 226, 226)
    assert out["offset"].shape == (1, 2, 226, 226)


def test_locate_is_deterministic(model, pair):
    rp, sp, _ = pair
    ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
    a = locate(model, ref, sea, torch.device("cpu"))
    b = locate(model, ref, sea, torch.device("cpu"))
    assert (a["x"], a["y"]) == (b["x"], b["y"])


def test_locate_returns_a_point_inside_the_frame(model, pair):
    rp, sp, _ = pair
    ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
    r = locate(model, ref, sea, torch.device("cpu"))
    assert 0 <= r["x"] <= sea.shape[1] - 1
    assert 0 <= r["y"] <= sea.shape[0] - 1


def test_phase2_explicit_zncc_reproduces_default(model, pair):
    rp, sp, _ = pair
    ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
    default = locate_phase2(model, ref, sea, torch.device("cpu"), polish=False)
    explicit = locate_phase2(model, ref, sea, torch.device("cpu"), polish=False,
                             verification="zncc")
    for key in ("x", "y", "scale", "theta", "score", "zncc"):
        assert explicit[key] == pytest.approx(default[key], abs=1e-7)


@pytest.mark.parametrize("shape", [(1000, 1000), (800, 1200), (1200, 800)])
def test_locate_and_tta_handle_non_square_frames(model, shape):
    """The docstring promises frames that are not exactly 1000 px still
    localise. TTA rotates the frame, so this is where it used to break."""
    rng = np.random.default_rng(12)
    h, w = shape
    sea = cv2.GaussianBlur(rng.integers(40, 210, (h, w), dtype=np.uint8), (0, 0), 2.0)
    x0, y0 = 120, 90
    ref = cv2.resize(sea[y0:y0 + 100, x0:x0 + 100], (1000, 1000),
                     interpolation=cv2.INTER_NEAREST)
    dev = torch.device("cpu")

    for r in (locate(model, ref, sea, dev), locate_tta(model, ref, sea, dev)):
        assert 0 <= r["x"] <= w - 1
        assert 0 <= r["y"] <= h - 1
        assert np.isfinite(r["x"]) and np.isfinite(r["y"])


def test_tta_reports_its_vote_structure(model, pair):
    rp, sp, _ = pair
    ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
    r = locate_tta(model, ref, sea, torch.device("cpu"))
    assert r["n_views"] == 8
    assert 1 <= r["votes"] <= 8
    assert 0.0 < r["agreement"] <= 1.0


def test_zncc_fallback_finds_a_planted_pattern(pair):
    """The no-torch path must still work; it is what runs if the grader's
    environment cannot load the weights."""
    rp, sp, (gx, gy) = pair
    ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
    r = zncc_only(ref, sea)
    assert np.hypot(r["x"] - gx, r["y"] - gy) < 5.0


# --- CLI contract ---------------------------------------------------------

def _run(args):
    return subprocess.run([sys.executable, INFER] + args,
                          capture_output=True, text=True, cwd=REPO_ROOT)


def test_cli_prints_exactly_one_coordinate_line(pair):
    rp, sp, _ = pair
    p = _run(["-r", rp, "-s", sp])
    assert p.returncode == 0, p.stderr
    lines = p.stdout.strip().splitlines()
    assert len(lines) == 1, f"stdout must be one line, got {lines}"
    assert re.fullmatch(r"-?\d+\.\d{2},-?\d+\.\d{2}", lines[0]), lines[0]


def test_cli_accepts_positional_paths(pair):
    rp, sp, _ = pair
    p = _run([rp, sp])
    assert p.returncode == 0, p.stderr
    assert re.fullmatch(r"-?\d+\.\d{2},-?\d+\.\d{2}", p.stdout.strip())


def test_cli_json_mode_is_parseable(pair):
    rp, sp, _ = pair
    p = _run(["-r", rp, "-s", sp, "--json"])
    assert p.returncode == 0, p.stderr
    blob = json.loads(p.stdout)
    assert {"x", "y", "score", "method"} <= set(blob)
    assert "heatmap" not in blob          # numpy arrays must not leak into json


def test_cli_no_tta_agrees_with_the_default_path_in_format(pair):
    rp, sp, _ = pair
    p = _run(["-r", rp, "-s", sp, "--no-tta"])
    assert p.returncode == 0, p.stderr
    assert re.fullmatch(r"-?\d+\.\d{2},-?\d+\.\d{2}", p.stdout.strip())


def test_cli_keeps_warnings_off_stdout(pair):
    """A missing checkpoint must degrade to the classical matcher and still
    emit a clean coordinate -- the warning belongs on stderr."""
    rp, sp, _ = pair
    p = _run(["-r", rp, "-s", sp, "-w", "/nonexistent/weights.pt"])
    assert p.returncode == 0, p.stderr
    assert re.fullmatch(r"-?\d+\.\d{2},-?\d+\.\d{2}", p.stdout.strip())
    assert "warn" in p.stderr.lower()


def test_cli_fails_loudly_on_a_missing_image(tmp_path):
    p = _run(["-r", str(tmp_path / "nope.png"), "-s", str(tmp_path / "nope2.png")])
    assert p.returncode != 0
    assert p.stdout.strip() == ""

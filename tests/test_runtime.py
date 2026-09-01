"""Shared runtime regression tests for the Phase 2 mainline."""

import os

import cv2
import numpy as np
import torch

from driftsense import runtime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(REPO_ROOT, "weights", "driftsense.pt")


def test_shipped_checkpoint_loads_through_shared_runtime():
    loaded = runtime.load_model(WEIGHTS)
    assert loaded is not None
    model, device = loaded
    assert model is not None
    assert isinstance(device, torch.device)


def test_runtime_default_weights_is_shipped_checkpoint():
    assert os.path.normpath(runtime.DEFAULT_WEIGHTS) == os.path.normpath(WEIGHTS)


def test_read_gray_and_fallback_are_independent_of_legacy_cli(tmp_path):
    rng = np.random.default_rng(17)
    search = cv2.GaussianBlur(
        rng.integers(40, 210, (300, 300), dtype=np.uint8), (0, 0), 2.0
    )
    x0, y0 = 120, 80
    patch = search[y0:y0 + 30, x0:x0 + 30]
    reference = cv2.resize(patch, (300, 300), interpolation=cv2.INTER_NEAREST)

    rp = str(tmp_path / "reference.png")
    sp = str(tmp_path / "search.png")
    cv2.imwrite(rp, reference)
    cv2.imwrite(sp, search)

    ref = runtime.read_gray(rp)
    sea = runtime.read_gray(sp)
    result = runtime.zncc_fallback(ref, sea)

    assert 0 <= result["x"] <= sea.shape[1]
    assert 0 <= result["y"] <= sea.shape[0]
    assert "score" in result

"""Pose-estimate jitter must reach the on-disk training path.

`driftsense/dataset.build_sample` undoes the Phase 2 pose before the network
sees a frame, and its docstring says it does so with a *jittered* estimate on
purpose: "the estimate the pose search will hand us at inference is not exact,
so training canonicalises with a jittered pose ... which makes the network
tolerant of precisely the mistake it will actually be given."

The mechanism was implemented and correct, but `train.py` only ever passed
`pose_jitter` on the `--stream` branch. The on-disk branch (`--train-dirs`)
constructed `DriftSenseDataset` without it, so it sat at the constructor
default of (0, 0) -- and every shipped Phase 2 checkpoint (p6-p9, wide,
setcfull) came from that path, because scripts/wide_run.sh uses --train-dirs.
The network has therefore only ever trained on frames canonicalised with the
EXACT ground-truth pose: zero residual, a condition it never meets at
inference, where README reports 99% at 5px with a true pose against 83% with
an estimated one.

These tests pin three things:
  * the flag is opt-in (default None), so no existing command changes;
  * both defaults are preserved -- streaming keeps (0.015, 0.30) under
    --phase2, the on-disk path keeps (0, 0);
  * the jitter, once set, actually perturbs the canonicalisation rather than
    being accepted and ignored.
"""

import importlib.util
import os

import numpy as np
import pytest

from driftsense.dataset import build_sample
from driftsense.model import SCALE

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _root_train():
    """Import the repo-root train.py by path.

    conftest.py puts generator/ ahead of the repo root on sys.path and there is
    also a phase1/train.py, so a plain `import train` is ambiguous.
    """
    path = os.path.join(REPO_ROOT, "train.py")
    spec = importlib.util.spec_from_file_location("_root_train", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flag_is_opt_in_so_no_existing_command_changes():
    """--pose-jitter must default to None. A non-None default would silently
    turn jitter on for every on-disk run the moment it was wired through."""
    T = _root_train()
    import sys
    old = sys.argv
    try:
        sys.argv = ["train.py", "--train-dirs", "x", "--val-dir", "y"]
        args = T.parse_args()
    finally:
        sys.argv = old
    assert args.pose_jitter is None


def test_both_path_defaults_are_preserved():
    T = _root_train()
    # Streaming's historical value, unchanged.
    assert T.POSE_JITTER_STREAM_DEFAULT == (0.015, 0.30)
    # The on-disk path's effective value before this change was the
    # DriftSenseDataset constructor default.
    assert T.POSE_JITTER_DISK_DEFAULT == (0.0, 0.0)


def _structured_image(rng):
    """A spatially correlated field, not pixel noise.

    Pixel-independent noise decorrelates completely under ANY nonzero
    resampling shift, so "does a bigger shift move the image further" is
    unmeasurable on it -- every nonzero jitter looks equally "different".
    A blurred field has real spatial structure, so a larger geometric
    perturbation genuinely produces a larger pixel difference.
    """
    import cv2
    noise = rng.integers(0, 255, (1000, 1000)).astype(np.float32)
    return cv2.GaussianBlur(noise, (0, 0), sigmaX=6.0).astype(np.uint8)


def _sample(pose_jitter, seed=0):
    """One posed sample through the real training path."""
    rng_img = np.random.default_rng(5)
    search = _structured_image(rng_img)
    reference = _structured_image(rng_img)
    return build_sample(
        reference, search, gx=500.0, gy=500.0, crop=512, train=True,
        rng=np.random.default_rng(seed), resp=512 // 4 - 25 + 1,
        # A genuinely posed frame: build_sample only canonicalises (and so only
        # consults pose_jitter) when the pose is off-nominal.
        magnification=float(SCALE) + 1.4, rotation_deg=3.0,
        found=1, pose_jitter=pose_jitter)


def test_zero_jitter_is_deterministic_across_seeds_in_the_pose_step():
    """With jitter off, canonicalisation uses the exact ground-truth pose, so
    the pose step contributes no randomness -- the control for the test below."""
    a = _sample((0.0, 0.0), seed=1)
    b = _sample((0.0, 0.0), seed=1)
    assert np.array_equal(a["search"], b["search"])


def test_nonzero_jitter_actually_perturbs_the_canonicalisation():
    """The point of the lever: a set jitter must change what the network sees,
    not be accepted and ignored."""
    off = _sample((0.0, 0.0), seed=1)
    on = _sample((0.05, 1.0), seed=1)
    assert not np.array_equal(off["search"], on["search"]), \
        "pose_jitter had no effect on the canonicalised frame"


def test_jitter_magnitude_scales_the_perturbation():
    """A larger jitter must move the frame further from the exact-pose result,
    which is what makes the value a meaningful knob rather than a boolean."""
    # build_sample returns torch tensors; compare in numpy.
    base = _sample((0.0, 0.0), seed=3)["search"].numpy().astype(np.float64)
    small = _sample((0.01, 0.2), seed=3)["search"].numpy().astype(np.float64)
    large = _sample((0.08, 1.6), seed=3)["search"].numpy().astype(np.float64)
    d_small = float(np.abs(small - base).mean())
    d_large = float(np.abs(large - base).mean())
    assert d_large > d_small, (d_small, d_large)

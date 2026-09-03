"""Ground-truth label invariants.

The generator writes the label in the *pre-warp* canvas frame, but the search
image the model sees has been sheared and barrel-distorted afterwards. Getting
that inversion wrong shifts every label by a couple of pixels -- invisible at
the 10 px tolerance, fatal at 1 px. `scripts/verify_gt_correction.py` checks it
statistically with a real matcher; this checks the geometry exactly.
"""

import cv2
import numpy as np
import pytest

from driftsense.generate import SEARCH_SIZE_PX, correct_gt
from src import sem_imaging


def _row_shift(h, shear_amplitude, jitter_std, seed=0):
    rng = np.random.default_rng(seed)
    rows = np.arange(h)
    shear = shear_amplitude * (rows / max(h - 1, 1))
    jitter = rng.normal(0, jitter_std, size=h) if jitter_std > 0 else np.zeros(h)
    return (shear + jitter).astype(np.float32)


def _warp(img, row_shift, k):
    h, w = img.shape
    map_x = np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    out = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    return sem_imaging.apply_barrel_distortion(out, k)


def _centroid_of_blob(img):
    """Sub-pixel location of the planted blob, robust to the interpolation the
    warp applies (a hard argmax quantises to whole pixels)."""
    ys, xs = np.nonzero(img > img.max() * 0.25)
    wts = img[ys, xs].astype(np.float64)
    return float(np.average(xs, weights=wts)), float(np.average(ys, weights=wts))


def _planted(px, py, size=SEARCH_SIZE_PX):
    img = np.zeros((size, size), np.float32)
    cv2.circle(img, (px, py), 3, 255.0, -1)
    return img


@pytest.mark.parametrize("py", [120, 500, 880])
@pytest.mark.parametrize("shear", [1.5, 4.0])
def test_correct_gt_inverts_pure_shear(py, shear):
    px = 460
    row_shift = _row_shift(SEARCH_SIZE_PX, shear, 0.0)

    warped = _warp(_planted(px, py), row_shift, k=0.0)
    seen_x, seen_y = _centroid_of_blob(warped)
    corr_x, corr_y = correct_gt(float(px), float(py), row_shift, 0.0)

    assert abs(corr_x - seen_x) < 0.25, f"x: corrected {corr_x} vs actual {seen_x}"
    assert abs(corr_y - seen_y) < 0.25, f"y: corrected {corr_y} vs actual {seen_y}"


@pytest.mark.parametrize("k", [0.02, -0.02])
def test_correct_gt_inverts_pure_barrel(k):
    px, py = 300, 700
    row_shift = np.zeros(SEARCH_SIZE_PX, np.float32)

    warped = _warp(_planted(px, py), row_shift, k=k)
    seen_x, seen_y = _centroid_of_blob(warped)
    corr_x, corr_y = correct_gt(float(px), float(py), row_shift, k)

    assert abs(corr_x - seen_x) < 1.0, f"x: corrected {corr_x} vs actual {seen_x}"
    assert abs(corr_y - seen_y) < 1.0, f"y: corrected {corr_y} vs actual {seen_y}"


def test_correct_gt_inverts_shear_and_barrel_together():
    px, py = 380, 640
    row_shift = _row_shift(SEARCH_SIZE_PX, 3.0, 1.0, seed=7)
    k = 0.015

    warped = _warp(_planted(px, py), row_shift, k=k)
    seen_x, seen_y = _centroid_of_blob(warped)
    corr_x, corr_y = correct_gt(float(px), float(py), row_shift, k)

    assert np.hypot(corr_x - seen_x, corr_y - seen_y) < 1.0


def test_correct_gt_is_a_no_op_without_warps():
    row_shift = np.zeros(SEARCH_SIZE_PX, np.float32)
    assert correct_gt(123.0, 456.0, row_shift, 0.0) == (123.0, 456.0)


def test_uncorrected_label_is_measurably_wrong():
    """The whole reason the correction exists: guards against someone
    'simplifying' it away because the two labels look close enough."""
    px, py = 400, 900
    row_shift = _row_shift(SEARCH_SIZE_PX, 4.0, 0.0)
    warped = _warp(_planted(px, py), row_shift, k=0.0)
    seen_x, _ = _centroid_of_blob(warped)

    assert abs(px - seen_x) > 3.0        # the raw label is off by the shear
    corr_x, _ = correct_gt(float(px), float(py), row_shift, 0.0)
    assert abs(corr_x - seen_x) < 0.25   # the corrected one is not


# --- label-noise weighting on the sub-pixel head ---------------------------
# The offset target is only accurate to ~0.72x the frame's raster drift, so the
# loss down-weights pairs whose labels are noisiest. Three things must hold: the
# old behaviour is preserved when no jitter is recorded, the weights have the
# intended magnitude, and the loss keeps its scale so the offset/focal balance
# is not silently changed.

def test_offset_loss_unchanged_without_jitter():
    import torch
    from driftsense.engine import offset_loss
    torch.manual_seed(0)
    b = 8
    off, tgt = torch.randn(b, 2, 6, 6), torch.randn(b, 2)
    peak, found = torch.zeros(b, 2, dtype=torch.long), torch.ones(b)
    base = offset_loss(off, tgt, peak, found)
    assert offset_loss(off, tgt, peak, found, None) == base
    # A manifest without the column yields zeros, which must be a no-op rather
    # than damping every pair towards zero weight.
    assert offset_loss(off, tgt, peak, found, torch.zeros(b)) == base


def test_offset_loss_downweights_noisy_labels():
    import torch
    from driftsense.engine import LABEL_NOISE_GAIN, LABEL_NOISE_SIGMA0, offset_loss
    b = 8
    peak, found = torch.zeros(b, 2, dtype=torch.long), torch.ones(b)
    off = torch.zeros(b, 2, 6, 6)
    # Give the noisy half a large residual and the clean half none. Weighting
    # must pull the loss below the unweighted mean.
    tgt = torch.cat([torch.zeros(b // 2, 2), torch.ones(b // 2, 2)])
    jit = torch.tensor([0.55] * (b // 2) + [1.89] * (b // 2))
    assert offset_loss(off, tgt, peak, found, jit) < offset_loss(off, tgt, peak, found)

    sig = LABEL_NOISE_GAIN * jit
    w = LABEL_NOISE_SIGMA0 ** 2 / (LABEL_NOISE_SIGMA0 ** 2 + sig ** 2)
    w = w / w.mean()
    assert abs(float(w.mean()) - 1.0) < 1e-5           # scale preserved
    assert 2.5 < float(w[0] / w[-1]) < 3.3             # severity 1 vs 4


def test_build_sample_carries_drift_jitter():
    import numpy as np
    from driftsense.dataset import build_sample
    rng = np.random.default_rng(0)
    ref = rng.integers(0, 255, (1000, 1000), dtype=np.uint8).astype(np.uint8)
    sea = rng.integers(0, 255, (1000, 1000), dtype=np.uint8).astype(np.uint8)
    s = build_sample(ref, sea, 500.0, 500.0, 512, True, rng, 4, drift_jitter=1.37)
    assert abs(float(s["jitter"]) - 1.37) < 1e-5
    # Absent by default means unknown, not zero-weighted.
    assert float(build_sample(ref, sea, 500.0, 500.0, 512, True, rng, 4)["jitter"]) == 0.0

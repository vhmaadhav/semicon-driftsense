"""Whole-frame estimators: rotation and the label row's drift over the full width.

Pins the two claims the stages rest on, on synthetic frames whose geometry is
known exactly:

* raster drift moves content only horizontally, so the angle at which the
  layout's horizontal edges line up across the frame is the rotation itself --
  shear, per-row jitter and charging streaks do not move it;
* a scan row's drift is recoverable from the row's vertical continuity with
  its neighbours, over the whole frame width, and the label's correction is
  that row's sample relative to the template rows' mean.
"""
import cv2
import numpy as np
import pytest

from driftsense.matching import (
    destreaked_y,
    full_width_refine,
    global_rotation,
    row_shift_band,
)

SIZE = 1000


def _drift(img, shift):
    """The imaging model's raster drift: row y samples x + shift[y]."""
    h, w = img.shape
    mx = np.arange(w, dtype=np.float32)[None, :] + shift.astype(np.float32)[:, None]
    my = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    return cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _manhattan(seed, size=1600):
    """Axis-aligned layout with fine horizontal and vertical lines at irregular
    pitch (2-8 px) and a few wide horizontal strips."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 90.0, np.float32)
    y = 0.0
    while y < size:
        y += rng.uniform(2.5, 8.0)
        img[int(y):int(y) + 1, :] += rng.uniform(40, 90)
    x = 0.0
    while x < size:
        x += rng.uniform(3.0, 9.0)
        img[:, int(x):int(x) + 2] += rng.uniform(20, 60)
    for _ in range(6):
        y0 = int(rng.uniform(0, size - 40))
        img[y0:y0 + int(rng.uniform(15, 35)), :] = 60.0
    return cv2.GaussianBlur(img, (0, 0), 0.7)


def _search_frame(theta, seed=1, jitter=0.6, shear=3.0, noise=6.0, streaks=0):
    rng = np.random.default_rng(seed + 100)
    canvas = _manhattan(seed)
    c = (canvas.shape[1] - 1) / 2.0
    # Canvas rotated by +theta (CCW, y-down), as the Phase 2 generator poses it.
    M = cv2.getRotationMatrix2D((c, c), theta, 1.0)
    M[:, 2] += (SIZE - 1) / 2.0 - c
    img = cv2.warpAffine(canvas, M, (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    shift = shear * np.arange(SIZE) / (SIZE - 1) + rng.normal(0, jitter, SIZE)
    img = _drift(img, shift)
    img = img + rng.normal(0, noise, img.shape)
    for _ in range(streaks):
        r = int(rng.integers(0, SIZE - 4))
        img[r:r + 3, :] += 45.0
    return np.clip(img, 0, 255).astype(np.uint8), shift


@pytest.mark.parametrize("theta", [-4.3, -0.4, 0.0, 1.7, 4.8])
def test_global_rotation_recovers_theta_through_drift_and_streaks(theta):
    frame, _ = _search_frame(theta, streaks=12)
    got = global_rotation(frame, theta + 0.3)        # start 0.3 deg off
    assert abs(got - theta) < 0.05


def test_global_rotation_ignores_raster_shear():
    """3 px of shear across the frame tilts vertical edges by ~0.17 deg; the
    horizontal edges the estimator reads are not moved at all."""
    for shear in (0.0, 3.0, -3.0):
        frame, _ = _search_frame(1.2, jitter=0.0, shear=shear)
        assert abs(global_rotation(frame, 1.0) - 1.2) < 0.03


def _vertical_texture(seed, size=SIZE):
    """Layout with long vertical continuity (the property the row reading
    relies on) but aperiodic along x."""
    rng = np.random.default_rng(seed)
    img = np.full((size, size), 100.0, np.float32)
    for _ in range(420):
        x = rng.uniform(0, size)
        y0 = rng.uniform(-200, size)
        img[int(max(y0, 0)):int(y0 + rng.uniform(60, 400)), int(x):int(x) + rng.integers(1, 5)] \
            += rng.uniform(-60, 80)
    return cv2.GaussianBlur(img, (0, 0), 0.8)


def test_row_shift_band_recovers_the_per_row_jitter():
    rng = np.random.default_rng(7)
    layout = _vertical_texture(3)
    shift = 2.5 * np.arange(SIZE) / (SIZE - 1) + rng.normal(0, 0.8, SIZE)
    frame = np.clip(_drift(layout, shift) + rng.normal(0, 4, (SIZE, SIZE)), 0, 255).astype(np.uint8)
    rows, s = row_shift_band(frame, 500)
    truth = shift[rows] - np.polyval(np.polyfit(rows, shift[rows], 1), rows)
    inner = slice(4, len(rows) - 4)                  # band ends see one side only
    # The ridge that pins the null space also shrinks drift components smoother
    # than ~10 rows (measured as the better trade on noisy frames), so the
    # per-row pattern is recovered, not every slow wander of it.
    assert np.corrcoef(s[inner], truth[inner])[0, 1] > 0.9
    assert np.sqrt(np.mean((s - truth)[inner] ** 2)) < 0.25


@pytest.mark.parametrize("convention", ["edge", "center"])
def test_full_width_refine_moves_x_by_the_label_rows_own_sample(convention):
    """The rigid match recovers the template rows' mean shift; the label
    carries its own row's. The correction is the difference."""
    rng = np.random.default_rng(11)
    layout = _vertical_texture(5)
    shift = rng.normal(0, 0.9, SIZE)
    # Lower the layout by a quarter pixel, so the template's centre sits at
    # y = 520.25 (pixel-edge) and 519.75 (pixel-centre): row 520 either way,
    # a quarter pixel from any rounding boundary.
    lowered = cv2.warpAffine(layout, np.float32([[1, 0, 0], [0, 1, 0.25]]), (SIZE, SIZE),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    frame = np.clip(_drift(lowered, shift) + rng.normal(0, 4, (SIZE, SIZE)), 0, 255).astype(np.uint8)
    th = tw = 100
    template = layout[470:570, 430:530].astype(np.float32)
    cx, cy = 480.0, 520.25                            # the rigid pixel-edge match
    got = full_width_refine(frame, template, cx, cy, label_convention=convention)
    assert got is not None
    trows = np.arange(470, 470 + th)
    expected = cx - (shift[520] - shift[trows].mean())
    assert abs(got[0] - expected) < 0.2
    assert abs(got[1] - cy) < 0.25


def test_destreaked_y_is_not_pulled_by_a_charging_streak():
    rng = np.random.default_rng(2)
    layout = _manhattan(4, size=SIZE)
    frame = np.clip(layout + rng.normal(0, 4, layout.shape), 0, 255)
    frame[503:506, :] += 60.0                         # a streak inside the template window
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    th = tw = 100
    cx, cy = 500.0, 500.0
    template = layout[450:550, 450:550]
    got = destreaked_y(frame, template, cx, cy)
    assert got is not None and abs(got - cy) < 0.15


def test_full_width_refine_rejects_an_unknown_convention():
    frame = np.zeros((SIZE, SIZE), np.uint8)
    with pytest.raises(ValueError):
        full_width_refine(frame, np.zeros((100, 100), np.float32), 500.0, 500.0,
                          label_convention="corner")

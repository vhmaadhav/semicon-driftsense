import numpy as np
from scripts.probe_global_scanline import relative_shift


def test_scanline_shift_direction():
    rng = np.random.default_rng(2)
    row = rng.integers(0, 255, 200, dtype=np.uint8)
    image = np.repeat(row[None], 9, axis=0)
    image[4] = np.roll(row, 2)
    assert abs(relative_shift(image, 100, 4, 70) - 2) < 0.05


def test_uninformative_scanline_is_unsupported():
    assert relative_shift(np.ones((9, 200), np.uint8), 100, 4, 70) is None

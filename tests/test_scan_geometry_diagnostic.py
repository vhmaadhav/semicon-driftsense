import numpy as np
from scripts.diagnose_scan_geometry import radial, undistort


def test_radial_forward_inverse_agree_at_both_distortion_signs():
    x = np.array([1.0, 125.0, 499.5, 888.0, 999.0])
    y = np.array([25.0, 899.0, 499.5, 400.0, 999.0])
    for k in [-0.02, 0.0, 0.02]:
        sx, sy = radial(x, y, k, (1000, 1000))
        rx, ry = radial(sx, sy, k, (1000, 1000), inverse=True)
        np.testing.assert_allclose(rx, x, atol=1e-7)
        np.testing.assert_allclose(ry, y, atol=1e-7)


def test_zero_distortion_does_not_resample():
    a = np.arange(100, dtype=np.uint8).reshape(10, 10)
    assert undistort(a, 0) is a

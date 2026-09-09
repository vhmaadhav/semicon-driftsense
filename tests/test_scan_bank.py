import numpy as np
from driftsense.scan_bank import candidates, radial


def test_identity_survives_all_unsupported_hypotheses():
    ref = np.zeros((1000, 1000), np.uint8)
    search = np.zeros((100, 100), np.uint8)
    x, f, valid = candidates(ref, search, 1.0, 2.0, 10.0, 0.0)
    assert x.shape == (10,) and f.shape == (10, 12)
    assert valid[0] and not valid[1:].any()
    assert np.all(x == 1.0) and np.isfinite(f).all()


def test_production_coordinate_map_roundtrips():
    x, y = radial(123.0, 780.0, -0.015, (1000, 1000))
    rx, ry = radial(x, y, -0.015, (1000, 1000), inverse=True)
    np.testing.assert_allclose([rx, ry], [123.0, 780.0], atol=1e-7)

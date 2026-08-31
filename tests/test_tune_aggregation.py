"""Regression tests for scripts/tune_aggregation.py.

C-04 of the static audit: the cache builder passed `sea.shape[0]` (an int)
where `_dihedral_point_inv` requires an `(h, w)` tuple, so cache construction
crashed before aggregation tuning could run -- and even conceptually could
not invert non-square frames. These tests pin the round trip through the
extracted `dihedral_proposals` helper on non-square images.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.matching import _dihedral_img  # noqa: E402


def _load_script():
    path = os.path.join(REPO_ROOT, "scripts", "tune_aggregation.py")
    spec = importlib.util.spec_from_file_location("tune_aggregation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _scene_with_marker(h=60, w=100, x=37, y=11):
    """Non-square search image whose only bright pixel sits at (x, y)."""
    sea = np.zeros((h, w), dtype=np.float32)
    sea[y, x] = 1.0
    return sea


def _fake_locate_factory():
    """Stand-in for `locate`: returns the marker's position in whatever
    (dihedral-transformed) frame it is handed."""
    def fake_locate(model, reference, search, device, refine=False, **kw):
        ty, tx = search.shape[-2:]
        idx = int(np.argmax(search))
        y, x = np.unravel_index(idx, search.shape[-2:])
        return {"x": float(x), "y": float(y), "score": 0.9}
    return fake_locate


class _NullModel:
    pass


@pytest.mark.parametrize("t", range(8))
def test_dihedral_proposals_round_trip_non_square(t):
    mod = _load_script()
    h, w, x0, y0 = 60, 100, 37, 11
    sea = _scene_with_marker(h, w, x0, y0)
    ref = np.zeros((10, 10), dtype=np.float32)

    mod.locate = _fake_locate_factory()
    props = mod.dihedral_proposals(_NullModel(), ref, sea, device="cpu")

    assert len(props) == 8
    px, py, score = props[t]
    assert (round(px), round(py)) == (x0, y0), (
        f"t={t}: proposal ({px:.2f}, {py:.2f}) did not round-trip the marker "
        f"at ({x0}, {y0}) through the dihedral inverse")
    assert score == pytest.approx(0.9)


def test_dihedral_proposals_shape_is_tuple():
    """The helper must hand `_dihedral_point_inv` the full (h, w) shape."""
    import inspect
    mod = _load_script()
    src = inspect.getsource(mod.dihedral_proposals)
    assert "shape[:2]" in src

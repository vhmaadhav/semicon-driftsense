"""Phase 3 GDS read path (``driftsense/gds.py``).

What is pinned here:

1. **The yield ladder is derived, not hard-coded.** Brightness must come from a
   layer's position in the stack, because a GDS layer is only an integer -- the
   file carries no material information. If this regresses to per-name
   constants the matcher silently stops inverting the generator's model.

2. **Fail closed.** Every failure mode (missing dependency, unreadable file, no
   polygons, empty file) raises. None of them may degrade into an empty or
   background-only raster, because downstream that becomes a declined row: a
   well-formed predictions.csv that scores zero and exits 0.

3. **Layer handling is real.** Layers present in the file but absent from a
   requested range must not shift the numbering, and ``min_layer`` must drop
   exactly the layers below the index.

These run against real gdstk-written files, not mocks, so the read path is
exercised the way the graded run would exercise it.
"""

import os

import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")

import sys  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense import gds  # noqa: E402


def _write_gds(tmp_path, layers, name="ref.gds", size=1000.0, stagger=None):
    """Write a real GDSII file with one rectangle per (layer, index).

    stagger : float | None
        When given, layer ``i`` (in sorted order) is shifted right by
        ``stagger * i`` so the layers occupy distinct x ranges instead of
        covering each other.
    """
    cell = gdstk.Cell("REF")
    for pos, layer in enumerate(sorted(layers)):
        count = layers[layer]
        dx = 0.0 if stagger is None else stagger * float(pos)
        for i in range(count):
            lo = (i * 40.0 + dx, i * 30.0)
            hi = (lo[0] + 300.0, lo[1] + 200.0)
            cell.add(gdstk.rectangle(lo, hi, layer=layer))
    lib = gdstk.Library()
    lib.add(cell)
    path = str(tmp_path / name)
    lib.write_gds(path)
    assert os.path.exists(path)
    return path


# --------------------------------------------------------------------------
# yield ladder
# --------------------------------------------------------------------------

def test_layer_yield_increases_with_stack_position():
    n = 8
    yields = [gds.layer_yield(i, n) for i in range(n)]
    assert yields == sorted(yields), "yield must be monotonic in layer index"
    assert yields[0] == pytest.approx(gds.BASE_YIELD)
    assert yields[-1] == pytest.approx(gds.TOP_YIELD)


def test_single_layer_stack_returns_top_yield():
    assert gds.layer_yield(0, 1) == pytest.approx(gds.TOP_YIELD)


def test_background_is_darker_than_every_layer():
    """The field must not read as bright as a drawn layer, or a thin design
    becomes invisible against its own background."""
    for i in range(8):
        assert gds.yield_to_intensity(gds.layer_yield(i, 8)) > gds.background_intensity()


def test_yield_to_intensity_clamps_to_8bit():
    assert gds.yield_to_intensity(-1.0) == 0
    assert gds.yield_to_intensity(2.0) == 255


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def test_read_returns_one_entry_per_present_layer(tmp_path):
    path = _write_gds(tmp_path, {0: 2, 3: 1, 7: 4})
    by_layer, num_layers = gds.read_gds_layers(path)
    assert num_layers == 8, "num_layers must span to the highest present layer"
    assert sorted(by_layer) == [0, 3, 7]
    assert len(by_layer[0]) == 2 and len(by_layer[7]) == 4


def test_read_rejects_a_file_with_no_polygons(tmp_path):
    cell = gdstk.Cell("EMPTY")
    lib = gdstk.Library()
    lib.add(cell)
    path = str(tmp_path / "empty.gds")
    lib.write_gds(path)
    with pytest.raises(gds.GdsError, match="datatype-0 polygons"):
        gds.read_gds_layers(path)


def test_read_rejects_a_missing_file(tmp_path):
    with pytest.raises(gds.GdsError):
        gds.read_gds_layers(str(tmp_path / "nope.gds"))


def test_read_rejects_a_non_gds_file(tmp_path):
    p = tmp_path / "not.gds"
    p.write_bytes(b"this is not a gds file at all")
    with pytest.raises(gds.GdsError):
        gds.read_gds_layers(str(p))


# --------------------------------------------------------------------------
# rasterizing
# --------------------------------------------------------------------------

def test_rasterize_paints_the_requested_layer_intensity(tmp_path):
    path = _write_gds(tmp_path, {0: 1})
    polys, n = gds.read_gds_layers(path)
    img = gds.rasterize_layers(polys, n, size=256,
                               layer_intensities={0: 200})
    assert img.shape == (256, 256) and img.dtype == np.uint8
    assert img.max() == 200, "the requested intensity must be what gets painted"
    assert (img == 200).sum() > 0


def test_rasterize_defaults_to_the_derived_yield(tmp_path):
    path = _write_gds(tmp_path, {0: 1})
    polys, n = gds.read_gds_layers(path)
    img = gds.rasterize_layers(polys, n, size=256)
    assert img.max() == gds.yield_to_intensity(gds.layer_yield(0, n))


def test_higher_layer_overwrites_lower_where_they_overlap(tmp_path):
    """Painter's algorithm: a real SEM sees only the top surface."""
    cell = gdstk.Cell("STACK")
    # identical footprint on two layers, so overlap is total
    cell.add(gdstk.rectangle((10, 10), (200, 200), layer=0))
    cell.add(gdstk.rectangle((10, 10), (200, 200), layer=5))
    lib = gdstk.Library()
    lib.add(cell)
    path = str(tmp_path / "stack.gds")
    lib.write_gds(path)

    polys, n = gds.read_gds_layers(path)
    img = gds.rasterize_layers(polys, n, size=256,
                               layer_intensities={0: 50, 5: 210})
    assert img.max() == 210
    assert (img == 50).sum() == 0, "the lower layer should be fully covered"


def test_min_layer_drops_the_buried_layers(tmp_path):
    # stagger so the two layers occupy different x ranges and both stay visible
    # inside the rendered window
    path = _write_gds(tmp_path, {0: 1, 6: 1}, stagger=350.0)
    polys, n = gds.read_gds_layers(path)
    full = gds.rasterize_layers(polys, n, size=800,
                                layer_intensities={0: 40, 6: 220})
    cut = gds.rasterize_layers(polys, n, size=800,
                               layer_intensities={0: 40, 6: 220}, min_layer=6)
    assert full.max() == 220 and cut.max() == 220
    assert (full == 40).sum() > 0, "layer 0 must be visible when it is not covered"
    assert (cut == 40).sum() == 0, "min_layer must exclude the lower layer entirely"


def test_absent_layers_do_not_shift_numbering(tmp_path):
    """A file with layers 0 and 7 only is still an 8-layer stack."""
    path = _write_gds(tmp_path, {0: 1, 7: 1})
    polys, n = gds.read_gds_layers(path)
    assert n == 8
    img = gds.rasterize_layers(polys, n, size=128)
    assert img.max() == gds.yield_to_intensity(gds.layer_yield(7, 8))


def test_rasterize_rejects_a_bad_size(tmp_path):
    path = _write_gds(tmp_path, {0: 1})
    polys, n = gds.read_gds_layers(path)
    with pytest.raises(gds.GdsError):
        gds.rasterize_layers(polys, n, size=0)


def test_offset_shifts_geometry(tmp_path):
    path = _write_gds(tmp_path, {0: 1})
    polys, n = gds.read_gds_layers(path)
    a = gds.rasterize_layers(polys, n, size=256, layer_intensities={0: 200})
    b = gds.rasterize_layers(polys, n, size=256, layer_intensities={0: 200},
                             offset=(-9.0, -9.0))
    assert not np.array_equal(a, b), "offset must actually move the geometry"


def test_render_reference_is_the_documented_entry_point(tmp_path):
    path = _write_gds(tmp_path, {0: 2, 4: 2})
    img = gds.render_reference(path, size=512)
    assert img.shape == (512, 512) and img.dtype == np.uint8
    assert img.min() >= 0 and img.max() <= 255
    assert img.max() > gds.background_intensity(), "the design must be visible"


# --------------------------------------------------------------------------
# fail-closed contract
# --------------------------------------------------------------------------

def test_gds_error_is_a_runtime_error():
    """Callers catch RuntimeError-family; it must not be a bare Exception."""
    assert issubclass(gds.GdsError, RuntimeError)


def test_missing_dependency_message_names_the_fix(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "gdstk":
            raise ImportError("simulated absence")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(gds.GdsError, match="gdstk"):
        gds._require_gdstk()

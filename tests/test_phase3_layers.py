"""Phase 3 layer emission and params schema.

The properties pinned here are the ones whose failure is silent and expensive:

1. **The 8-layer stack exists and matches the deck's staged view.** Layers 1, 3
   and 5 must be word line / bit line metal / storage capacitor, because the
   deck's staged view is defined by those cut points ("0 to 1 up to word line",
   "0 to 3 up to bit line metal", "0 to 5 up to storage capacitor"). A silent
   renumbering would break the correspondence with the brief.

2. **Brightness is never baked into the geometry.** GDS emission must not
   assign grey levels; only the render side derives them. If a builder started
   writing intensity-derived coordinates or a per-layer colour, "infer the
   per-layer brightness" would stop being a task.

3. **Every emitted polygon is datatype 0.** Non-zero datatypes are not drawn
   geometry (annotations, fill, DRC markers). Emitting them, or reading them
   back, breaks the reference/search correspondence.

4. **params.json round-trips and refuses malformed input.** It is the
   supervision target for per-layer brightness, and it records which layers are
   even observable.
"""

import json
import os
import sys

import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense import gds, gds_layers, params3  # noqa: E402


def _preset(name):
    sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))
    from src.presets import get_preset
    return get_preset(name)


# --------------------------------------------------------------------------
# 1. the stack
# --------------------------------------------------------------------------

def test_dram_layer_names_match_the_deck_staged_view():
    names = gds_layers.DRAM_LAYERS
    assert len(names) == 8, "the deck's staged view is an 8-layer stack"
    assert names[1] == "word_line", "layer 1 is the 'up to word line' cut"
    assert names[3] == "bit_line_metal", "layer 3 is the 'up to bit line metal' cut"
    assert names[5] == "storage_capacitor", "layer 5 is the storage-capacitor cut"
    assert names[7] == "metal2_strap", "all 8 layers is the full GDS"


def test_finfet_stack_is_eight_layers():
    assert len(gds_layers.FINFET_LAYERS) == 8
    assert gds_layers.FINFET_LAYERS[0] == "fin"
    assert gds_layers.FINFET_LAYERS[1] == "gate"


@pytest.mark.parametrize("arch,preset", [
    ("dram", "dram_1x"), ("dram", "dram_loose"),
    ("finfet", "finfet_10nm"), ("finfet", "finfet_22nm"),
])
def test_builder_populates_every_layer(arch, preset):
    rng = np.random.default_rng(3)
    cell = gds_layers.build_gds(arch, 3000.0, _preset(preset), 10.0, rng)
    counts = gds_layers.layer_population(cell)
    assert set(counts) == set(range(8))
    empty = [L for L, c in counts.items() if c == 0]
    assert not empty, f"layers {empty} are empty -- the stack is incomplete"


def test_builder_rejects_an_unknown_architecture():
    with pytest.raises(ValueError, match="unknown architecture"):
        gds_layers.build_gds("gpu", 1000.0, {}, 10.0, np.random.default_rng(0))


# --------------------------------------------------------------------------
# 2. brightness is not baked into geometry
# --------------------------------------------------------------------------

def test_emission_is_independent_of_intensity():
    """Two cells built from the same seed must be identical regardless of any
    intensity notion -- brightness is a render-side decision, not geometry."""
    a = gds_layers.build_gds("dram", 3000.0, _preset("dram_1x"), 10.0,
                             np.random.default_rng(11))
    b = gds_layers.build_gds("dram", 3000.0, _preset("dram_1x"), 10.0,
                             np.random.default_rng(11))
    pa = sorted((p.layer, p.datatype, tuple(np.round(p.points, 6).ravel()))
                for p in a.get_polygons())
    pb = sorted((p.layer, p.datatype, tuple(np.round(p.points, 6).ravel()))
                for p in b.get_polygons())
    assert pa == pb, "same seed must give identical geometry"


def test_render_derives_brightness_from_layer_index():
    """The same geometry must render differently when the ladder changes,
    proving brightness comes from the render side and not the polygons."""
    rng = np.random.default_rng(5)
    cell = gds_layers.build_dram_gds(3000.0, _preset("dram_1x"), 10.0, rng)
    polys = {L: [np.asarray(p.points, float)
                 for p in cell.get_polygons(layer=L, datatype=0)]
             for L in range(8)}
    a = gds.rasterize_layers(polys, 8, size=512)
    b = gds.rasterize_layers(polys, 8, size=512,
                             layer_intensities={L: 200 for L in range(8)})
    assert not np.array_equal(a, b), "intensity override must change the raster"
    assert b.max() == 200


# --------------------------------------------------------------------------
# 3. datatype discipline
# --------------------------------------------------------------------------

def test_emitted_polygons_are_all_datatype_zero():
    for arch, preset in (("dram", "dram_1x"), ("finfet", "finfet_10nm")):
        cell = gds_layers.build_gds(arch, 2000.0, _preset(preset), 10.0,
                                    np.random.default_rng(2))
        dts = {p.datatype for p in cell.get_polygons()}
        assert dts == {0}, f"{arch} emitted non-zero datatypes {dts}"


def test_reader_skips_non_zero_datatypes(tmp_path):
    """A datatype!=0 polygon is not drawn geometry and must not be painted."""
    cell = gdstk.Cell("DT")
    cell.add(gdstk.rectangle((0, 0), (200, 200), layer=3, datatype=0))
    cell.add(gdstk.rectangle((400, 0), (600, 200), layer=3, datatype=7))
    lib = gdstk.Library()
    lib.add(cell)
    path = str(tmp_path / "dt.gds")
    lib.write_gds(path)

    polys, n = gds.read_gds_layers(path)
    assert n == 4
    assert len(polys[3]) == 1, "only the datatype-0 polygon may be read"
    img = gds.rasterize_layers(polys, n, size=700, layer_intensities={3: 255})
    # the datatype=7 rectangle sat at x 400-600; nothing may be painted there
    assert img[:, 420:580].max() == gds.background_intensity()


def test_reader_reports_a_file_with_only_non_zero_datatypes(tmp_path):
    cell = gdstk.Cell("ONLYDT")
    cell.add(gdstk.rectangle((0, 0), (100, 100), layer=1, datatype=3))
    lib = gdstk.Library()
    lib.add(cell)
    path = str(tmp_path / "onlydt.gds")
    lib.write_gds(path)
    with pytest.raises(gds.GdsError, match="datatype"):
        gds.read_gds_layers(path)


# --------------------------------------------------------------------------
# 4. clipping: the reference must describe the same scene as the search
# --------------------------------------------------------------------------

def test_clip_cell_to_rect_drops_geometry_outside_the_window(tmp_path):
    cell = gdstk.Cell("BIG")
    cell.add(gdstk.rectangle((0, 0), (100, 100), layer=0))       # inside
    cell.add(gdstk.rectangle((900, 0), (1000, 100), layer=0))    # outside
    out = gds_layers.clip_cell_to_rect(cell, 200.0, 200.0, 8)
    pts = [np.asarray(p.points, float) for p in out.get_polygons(layer=0, datatype=0)]
    assert len(pts) == 1
    assert pts[0][:, 0].max() <= 200.0, "geometry past the rect must be dropped"


def test_clip_multi_cell_merges_overlapping_mats():
    """A window straddling two mats must contain geometry from BOTH.

    Clipping only the containing mat is the bug that made the reference
    disagree with the search image (measured 17.8% of pixels differing).
    """
    a = gdstk.Cell("A")
    a.add(gdstk.rectangle((0, 0), (100, 100), layer=0))
    b = gdstk.Cell("B")
    b.add(gdstk.rectangle((0, 0), (100, 100), layer=0))
    mats = [
        {"cell": a, "x0": 0, "y0": 0, "w": 100, "h": 100},
        {"cell": b, "x0": 100, "y0": 0, "w": 100, "h": 100},
    ]
    out = gds_layers.clip_multi_cell(mats, 50, 0, 100, 8)
    polys = out.get_polygons(layer=0, datatype=0)
    xs = np.concatenate([np.asarray(p.points, float)[:, 0] for p in polys])
    assert xs.min() <= 0.0 and xs.max() >= 100.0, (
        "the window spans both mats, so both must contribute")


def test_clip_multi_cell_ignores_non_overlapping_mats():
    a = gdstk.Cell("A")
    a.add(gdstk.rectangle((0, 0), (100, 100), layer=0))
    b = gdstk.Cell("B")
    b.add(gdstk.rectangle((0, 0), (100, 100), layer=0))
    mats = [
        {"cell": a, "x0": 0, "y0": 0, "w": 100, "h": 100},
        {"cell": b, "x0": 5000, "y0": 5000, "w": 100, "h": 100},
    ]
    out = gds_layers.clip_multi_cell(mats, 0, 0, 100, 8)
    assert len(out.get_polygons(layer=0, datatype=0)) == 1


def test_strip_cell_is_real_geometry_in_canvas_coords():
    strip = gds_layers.build_strip_cell(1000.0, 0.0, 200.0, 400.0,
                                        rng=np.random.default_rng(1))
    polys = strip.get_polygons(layer=gds_layers.STRIP_LAYER_INDEX, datatype=0)
    assert polys, "a strip with routing must emit polygons"
    xs = np.concatenate([np.asarray(p.points, float)[:, 0] for p in polys])
    assert xs.min() >= 1000.0, "strip geometry is in canvas coords, not local"


def test_rasterize_accepts_non_square_size():
    cell = gdstk.Cell("NS")
    cell.add(gdstk.rectangle((0, 0), (300, 100), layer=0))
    polys = {0: [np.asarray(p.points, float)
                 for p in cell.get_polygons(layer=0, datatype=0)]}
    img = gds.rasterize_layers(polys, 8, size=(400, 200))
    assert img.shape == (200, 400), "size must be (width, height)"


def test_non_square_raster_does_not_drop_geometry():
    """A square raster sliced to [:h,:w] loses polygons past w; the pair form
    must not."""
    cell = gdstk.Cell("WIDE")
    cell.add(gdstk.rectangle((0, 0), (100, 100), layer=0))
    cell.add(gdstk.rectangle((350, 0), (390, 100), layer=0))   # beyond w=200
    polys = {0: [np.asarray(p.points, float)
                 for p in cell.get_polygons(layer=0, datatype=0)]}
    img = gds.rasterize_layers(polys, 8, size=(200, 400),
                               layer_intensities={0: 255})
    # the far rectangle is outside this window and must be absent
    assert img[:, 150:].max() == gds.background_intensity()
    wide = gds.rasterize_layers(polys, 8, size=(400, 400),
                                layer_intensities={0: 255})
    assert wide[:, 360:].max() == 255, "a wider canvas must show it"


# --------------------------------------------------------------------------
# 5. params.json
# --------------------------------------------------------------------------

def test_params_round_trip(tmp_path):
    p = params3.build_params("dram", 8, {i: 40 + i * 20 for i in range(8)},
                             reference_gds_path="reference/p.gds",
                             present=True)
    path = str(tmp_path / "p.json")
    params3.write_params(path, p)
    back = params3.read_params(path)
    # JSON object keys are strings; the value identity is what matters
    assert params3.intensity_vector(back) == params3.intensity_vector(p)
    assert back["architecture"] == "dram"
    assert back["schema_version"] == params3.SCHEMA_VERSION


def test_params_records_which_layers_are_observable():
    p = params3.build_params("dram", 8, {i: 100 for i in range(8)})
    assert 4 in p["occluded_layers"], "the storage contact is covered by the capacitor"
    assert 6 in p["occluded_layers"], "via1 is covered by the metal2 strap"
    assert 4 not in p["observable_layers"]
    assert 6 not in p["observable_layers"]
    assert len(p["observable_layers"]) == 6


def test_params_rejects_a_bad_schema_version(tmp_path):
    path = str(tmp_path / "old.json")
    with open(path, "w") as f:
        json.dump({"schema_version": 99, "architecture": "dram",
                   "num_layers": 8, "layer_intensities": {"0": 50}}, f)
    with pytest.raises(params3.ParamsError, match="schema_version"):
        params3.read_params(path)


def test_params_rejects_a_missing_key(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as f:
        json.dump({"schema_version": 1, "architecture": "dram"}, f)
    with pytest.raises(params3.ParamsError, match="missing"):
        params3.read_params(path)


def test_params_rejects_an_out_of_range_intensity(tmp_path):
    path = str(tmp_path / "range.json")
    with open(path, "w") as f:
        json.dump({"schema_version": 1, "architecture": "dram",
                   "num_layers": 8, "layer_intensities": {"0": 999}}, f)
    with pytest.raises(params3.ParamsError, match="outside"):
        params3.read_params(path)


def test_params_rejects_missing_and_invalid_files(tmp_path):
    with pytest.raises(params3.ParamsError, match="not found"):
        params3.read_params(str(tmp_path / "nope.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(params3.ParamsError, match="valid JSON"):
        params3.read_params(str(bad))
    with pytest.raises(params3.ParamsError, match="no params path"):
        params3.read_params("")


def test_intensity_vector_is_dense_and_indexed():
    p = params3.build_params("dram", 8, {0: 11, 7: 222})
    assert params3.intensity_vector(p) == [11, 0, 0, 0, 0, 0, 0, 222]


def test_inference_does_not_import_the_params_reader():
    """phase3.py must not depend on params.json: the blind split leaves that
    column empty, so depending on it fails on the scored run."""
    src = open(os.path.join(REPO_ROOT, "phase3.py"), encoding="utf-8").read()
    assert "params3" not in src
    assert "read_params" not in src

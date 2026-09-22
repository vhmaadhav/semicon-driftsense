"""Phase 3 dataset generator (`generate_phase3_dataset.py`).

The central invariant, and the reason this file exists: **the reference GDS and
the search image must describe the same scene at the ground-truth location.**

That is easy to state and easy to break. Three separate defects were found by
testing it and none of them by reading the code:

  * clipping the reference from only the mat that contains the window, so a
    window straddling a mat/strip boundary was missing the neighbour's geometry
    (17.8% of pixels differed at the true match, NCC 0.42 instead of ~0.9);
  * rasterizing a non-square mat onto a square ``max(w, h)`` canvas and slicing
    ``[:h, :w]``, dropping every polygon past the slice;
  * painting strip routing into the search raster only, so a reference over a
    strip contained none of it -- the design cannot express paint that was never
    emitted as geometry.

So the tests below check the invariant directly (render the reference, render
the canvas, compare the window) rather than only checking shapes and counts.
"""

import csv
import os
import subprocess
import sys

import numpy as np
import pytest

gdstk = pytest.importorskip("gdstk")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import generate_phase3_dataset as G  # noqa: E402
from driftsense import gds, gds_layers, params3  # noqa: E402
from driftsense.pairs3 import PHASE3_FIELDS  # noqa: E402

NL = gds_layers.NUM_LAYERS


def _flat_intensities():
    """A fixed ladder, so tests are independent of the sampling in main()."""
    return {i: gds.yield_to_intensity(gds.layer_yield(i, NL)) for i in range(NL)}


def _render_reference_cell(cell, size=1000, intens=None):
    polys = {L: [np.asarray(p.points, float)
                 for p in cell.get_polygons(layer=L, datatype=0)]
             for L in range(NL)}
    return gds.rasterize_layers(polys, NL, size=size,
                                layer_intensities=intens or _flat_intensities())


def _run(args):
    return subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "generate_phase3_dataset.py"),
         *args],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=1800)


# --------------------------------------------------------------------------
# the core invariant
# --------------------------------------------------------------------------

@pytest.mark.parametrize("seed,arch,preset", [
    (7, "dram", "dram_1x"),
    (11, "finfet", "finfet_10nm"),
    (23, "dram", "dram_loose"),
    (5, "finfet", "finfet_28nm"),
])
def test_reference_matches_the_search_canvas_at_the_crop(seed, arch, preset):
    """The reference raster must equal the canvas window at the crop.

    This is the property that makes the ground truth true. A near-match is
    acceptable only for single-pixel rasteriser aliasing on polygon borders;
    real missing geometry shows up as several percent of pixels.
    """
    rng = np.random.default_rng(seed)
    _m, mats, strips, ref_cell, x0, y0 = G.build_site(arch, preset, rng)
    intens = _flat_intensities()

    canvas = G.render_search_canvas(mats, strips, NL, intens)
    window = canvas[y0:y0 + G.REFERENCE_SIZE_PX, x0:x0 + G.REFERENCE_SIZE_PX]
    ref = _render_reference_cell(ref_cell, intens=intens)

    assert window.shape == ref.shape == (1000, 1000)
    diff = np.abs(ref.astype(int) - window.astype(int))
    frac = float((diff > 0).mean())
    assert frac < 0.005, (
        f"{100*frac:.2f}% of pixels differ between the reference and the search "
        f"canvas at the ground-truth crop -- the two sides do not describe the "
        f"same scene")


def test_crop_always_allows_a_full_reference_window():
    """A crop origin near the canvas edge cannot hold a full reference, so GT
    would point outside the search frame."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        _m, _mats, _s, _c, x0, y0 = G.build_site("dram", "dram_1x", rng)
        assert 0 <= x0 <= G.FINE_CANVAS_SIZE_PX - G.REFERENCE_SIZE_PX
        assert 0 <= y0 <= G.FINE_CANVAS_SIZE_PX - G.REFERENCE_SIZE_PX
        # and the GT centre must land inside the 1000x1000 search frame
        cx = (x0 + G.REFERENCE_SIZE_PX / 2.0) / 10.0
        cy = (y0 + G.REFERENCE_SIZE_PX / 2.0) / 10.0
        assert 50.0 <= cx <= 950.0, f"gt_x {cx} outside the search frame"
        assert 50.0 <= cy <= 950.0, f"gt_y {cy} outside the search frame"


def test_zone_grid_is_irregular():
    """A regular mat grid is itself periodic, and a periodic layout aliases
    against itself: with a fixed period the reference matches every period
    equally well (measured as a consistent +240 px offset)."""
    rng = np.random.default_rng(4)
    spans = G._zone_grid(G.FINE_CANVAS_SIZE_PX, rng)
    widths = [e - s for is_mat, s, e in spans if is_mat]
    assert len(set(widths)) > 1, "mat widths must vary, not repeat one period"


def test_every_layer_is_present_in_a_generated_reference(tmp_path):
    for arch, preset in (("dram", "dram_1x"), ("finfet", "finfet_10nm")):
        _m, _mats, _s, ref_cell, _x, _y = G.build_site(
            arch, preset, np.random.default_rng(9))
        counts = gds_layers.layer_population(ref_cell)
        empty = [L for L, c in counts.items() if c == 0]
        assert not empty, f"{arch} reference is missing layers {empty}"


# --------------------------------------------------------------------------
# the CLI and its artefacts
# --------------------------------------------------------------------------

def test_cli_writes_all_five_artefacts(tmp_path):
    out = tmp_path / "ds"
    r = _run(["--num-pairs", "3", "--output-dir", str(out), "--seed", "7"])
    assert r.returncode == 0, r.stderr
    root = out / "train"
    for sub in ("reference", "search", "reference_sem", "params"):
        assert (root / sub).is_dir(), f"missing directory {sub}"
    assert (root / "pairs.csv").is_file()
    assert (root / "ground_truth.csv").is_file()
    gds_files = sorted((root / "reference").glob("*.gds"))
    assert len(gds_files) == 3

    for g in gds_files:
        cell = gdstk.read_gds(str(g)).top_level()[0]
        assert cell.get_polygons(), f"{g} has no polygons"
        dts = {p.datatype for p in cell.get_polygons()}
        assert dts == {0}, f"{g} contains non-zero datatypes {dts}"


def test_pairs_csv_uses_the_phase3_layout(tmp_path):
    out = tmp_path / "ds"
    r = _run(["--num-pairs", "3", "--output-dir", str(out), "--seed", "7"])
    assert r.returncode == 0, r.stderr
    with open(out / "train" / "pairs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == list(PHASE3_FIELDS), "schema must be exact"
    assert len(rows) == 3
    for row in rows:
        assert row["reference_sem_path"], "training split fills this"
        assert row["params_json_path"], "training split fills this"


def test_blind_split_empties_the_withheld_columns(tmp_path):
    out = tmp_path / "ds"
    r = _run(["--num-pairs", "3", "--output-dir", str(out), "--seed", "7",
              "--blind"])
    assert r.returncode == 0, r.stderr
    with open(out / "train" / "pairs.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert row["reference_sem_path"] == ""
        assert row["params_json_path"] == ""
        # everything else must still be present
        assert row["search_path"] and row["reference_gds_path"]


def test_ground_truth_is_real_and_matches_the_crop(tmp_path):
    out = tmp_path / "ds"
    r = _run(["--num-pairs", "2", "--output-dir", str(out), "--seed", "7",
              "--absent-frac", "0"])
    assert r.returncode == 0, r.stderr
    root = out / "train"
    with open(root / "pairs.csv", newline="") as f:
        pairs = {x["pair_id"]: x for x in csv.DictReader(f)}
    with open(root / "ground_truth.csv", newline="") as f:
        gt = {x["pair_id"]: x for x in csv.DictReader(f)}

    assert list(next(iter(gt.values())).keys()) == [
        "pair_id", "present", "x", "y", "theta", "scale"]

    for pid, g in gt.items():
        assert g["present"] == "1"
        cx, cy = float(g["x"]), float(g["y"])
        assert 0.0 < cx < 1000.0 and 0.0 < cy < 1000.0, "gt inside the search frame"
        # the recorded crop must reproduce the recorded ground truth
        p = params3.read_params(os.path.join(root, pairs[pid]["params_json_path"]))
        gen = p["generation"]
        assert abs(cx - (gen["crop_x0"] + 500) / 10.0) < 0.01
        assert abs(cy - (gen["crop_y0"] + 500) / 10.0) < 0.01


def test_absent_pairs_are_zero_filled_in_ground_truth(tmp_path):
    out = tmp_path / "ds"
    r = _run(["--num-pairs", "4", "--output-dir", str(out), "--seed", "3",
              "--absent-frac", "1.0"])
    assert r.returncode == 0, r.stderr
    with open(out / "train" / "ground_truth.csv", newline="") as f:
        for row in csv.DictReader(f):
            assert row["present"] == "0"
            assert row["x"] == row["y"] == "0"


def test_absent_rate_defaults_to_phase3_not_phase2():
    """Phase 3 discloses ~1 site in 12 (~8.3%), not Phase 2's 20%.

    Checked on the parsed default rather than by grepping source, so a
    reformatted line cannot make the test pass vacuously.
    """
    import contextlib
    import io

    # --help exits 0 and prints the parser's defaults
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit) as e:
        G.main(["--help"])
    assert e.value.code == 0
    help_text = buf.getvalue()
    assert "--absent-frac" in help_text
    # the default itself
    m = [ln for ln in help_text.splitlines() if "--absent-frac" in ln]
    assert m, "absent-frac must be documented in --help"
    assert "0.08" in help_text, (
        "the default absent rate must be Phase 3's ~8%, not Phase 2's 0.20")


def test_params_json_is_valid_and_records_observability(tmp_path):
    out = tmp_path / "ds"
    r = _run(["--num-pairs", "3", "--output-dir", str(out), "--seed", "7"])
    assert r.returncode == 0, r.stderr
    pdir = out / "train" / "params"
    files = sorted(pdir.glob("*.json"))
    assert len(files) == 3
    for f in files:
        p = params3.read_params(str(f))
        assert p["num_layers"] == 8
        assert len(p["layer_intensities"]) == 8
        for v in p["layer_intensities"].values():
            assert gds.background_intensity() < v <= 255, (
                "a drawn layer must be brighter than the background, or the "
                "design vanishes into the field")
        assert p["observable_layers"], "observability must be recorded"


def test_absent_rate_is_respected_statistically(tmp_path):
    out = tmp_path / "ds"
    r = _run(["--num-pairs", "40", "--output-dir", str(out), "--seed", "13",
              "--absent-frac", "0.5"])
    assert r.returncode == 0, r.stderr
    root = out / "train"
    with open(root / "ground_truth.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    absent = sum(1 for x in rows if x["present"] == "0")
    assert 10 <= absent <= 30, f"expected ~20 absent of 40, got {absent}"


def test_generator_does_not_import_torch():
    """The generator is a data tool; pulling torch in would make it unusable
    where only the light dependencies exist."""
    src = open(os.path.join(REPO_ROOT, "generate_phase3_dataset.py"),
               encoding="utf-8").read()
    assert "import torch" not in src

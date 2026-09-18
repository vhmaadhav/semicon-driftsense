"""CAD-anchored Phase 3 registration (driftsense.cad_anchor).

Unit tests pin each step on synthetic geometry; the end-to-end tests generate
real pairs with the organizer's own CAD pipeline (through
generator_i4c/generate_cad_varied.py, run as a subprocess because conftest
puts the older generator's `src` package first on sys.path) and check the
answer against that pipeline's ground truth.
"""

import csv
import json
import math
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from driftsense import cad_anchor as CA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
I4C = os.path.join(REPO, "generator_i4c")


# ---------------------------------------------------------------------------
# Label arithmetic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.0, 3.7, -8.2])
def test_design_to_search_is_the_generators_label_mapping(theta):
    # render_cad_sample: padded canvas 12500, margin 1250, rotation about the
    # padded centre, design point = crop origin + 500 (+ margin), then / 10.
    x0, y0, pad, m = 3217.0, 6120.0, 12500, 1250
    M = cv2.getRotationMatrix2D((pad / 2.0, pad / 2.0), theta, 1.0)
    fx, fy = x0 + 500 + m, y0 + 500 + m
    gx = (M[0, 0] * fx + M[0, 1] * fy + M[0, 2] - m) / 10.0
    gy = (M[1, 0] * fx + M[1, 1] * fy + M[1, 2] - m) / 10.0
    x, y = CA.design_to_search(x0 + 500, y0 + 500, theta, (1000, 1000))
    assert abs(x - gx) < 1e-9 and abs(y - gy) < 1e-9


# ---------------------------------------------------------------------------
# Rasterization helpers
# ---------------------------------------------------------------------------

def _rect(x0, y0, w, h):
    return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]], np.float64)


def test_visible_fractions_partition_every_pixel_top_layer_first():
    # cv2's anti-aliased fill bleeds ~half a pixel past an edge, so the
    # background probe sits well clear of both rectangles.
    polys = {0: [_rect(0, 0, 60, 60)], 1: [_rect(30, 30, 40, 40)]}
    m = CA.layer_masks(polys, 2, (10, 10), 10.0)
    v = CA.visible_fractions(m)
    assert np.allclose(v.sum(axis=0), 1.0, atol=1e-5)
    assert v[2, 4, 4] > 0.99 and v[1, 4, 4] < 0.01          # layer 1 hides layer 0
    assert v[1, 1, 1] > 0.99 and v[0, 9, 9] > 0.99


def test_bboxes_match_a_per_polygon_loop():
    rng = np.random.default_rng(0)
    polys = {0: [rng.uniform(0, 100, (int(rng.integers(3, 9)), 2)) for _ in range(50)]}
    got = CA._bboxes(polys)[0]
    want = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in polys[0]])
    assert np.array_equal(got, want)


# ---------------------------------------------------------------------------
# Step 1: CAD-to-CAD
# ---------------------------------------------------------------------------

def _lattice(extent, pitch, size, marker_at):
    """A periodic contact array with one odd-sized marker, so exactly one
    offset matches everything."""
    polys = [_rect(x, y, size, size) for x in np.arange(5, extent - size, pitch)
             for y in np.arange(5, extent - size, pitch)]
    polys.append(_rect(marker_at[0], marker_at[1], 2 * size + 7, size))
    return polys


def _clip(polys, x0, y0, side):
    out = []
    for p in polys:
        if p[:, 0].min() >= x0 and p[:, 1].min() >= y0 and p[:, 0].max() <= x0 + side and p[:, 1].max() <= y0 + side:
            out.append(p - (x0, y0))
    return out


def test_exact_offset_recovers_the_integer_origin_and_rejects_elsewhere():
    search_polys = {0: _lattice(4000, 40.0, 16.0, (2003.0, 1507.0))}
    search = CA.SearchCad(polys=search_polys, num_layers=1, bboxes=CA._bboxes(search_polys),
                          masks=CA.layer_masks(search_polys, 1, (400, 400), 10.0))
    ref = {0: _clip(search_polys[0], 1613, 1109, 1000)}
    origin, support, n_int, _ = CA.locate_reference(ref, 1, search)
    assert origin == (1613.0, 1109.0)
    assert support > 0.95 and n_int > 100
    # Same lattice, marker moved: the lattice aliases match, the marker does
    # not, so the best offset explains fewer than all interior polygons --
    # and a reference from an unrelated layout explains almost none.
    other = {0: [p + (3.0, 11.0) for p in _lattice(1000, 37.0, 13.0, (400.0, 300.0))]}
    _, support_other, _, _ = CA.locate_reference(other, 1, search)
    assert support_other < 0.2


# ---------------------------------------------------------------------------
# Step 2: rotation
# ---------------------------------------------------------------------------

def _scene(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((n, n), np.float32)
    for x in range(0, n, 9):
        img[:, x:x + 3] = 120
    for _ in range(40):                                  # non-periodic blocks
        x, y = rng.integers(0, n - 60, 2)
        img[y:y + int(rng.integers(20, 60)), x:x + int(rng.integers(20, 60))] = 220
    return cv2.GaussianBlur(img, (0, 0), 0.8)


@pytest.mark.parametrize("theta", [-7.3, 2.15])
def test_coarse_then_fine_rotation_recovers_the_angle(theta):
    model = _scene()
    img = CA._rotate(model, theta) + np.random.default_rng(1).normal(0, 8, model.shape).astype(np.float32)
    th0 = CA.coarse_rotation(img, model, 10.0)
    assert abs(th0 - theta) < 0.3
    th, *_ = CA.fine_rotation(img, model, th0)
    assert abs(th - theta) < 0.02


def test_fine_rotation_ignores_horizontal_raster_drift():
    model = _scene(seed=2)
    img = CA._rotate(model, 1.5)
    # Progressive per-row shear plus jitter, horizontal only -- as sem_imaging.
    rng = np.random.default_rng(3)
    shift = (2.5 * np.arange(1000) / 999 + rng.normal(0, 1.0, 1000)).astype(np.float32)
    mx = np.arange(1000, dtype=np.float32)[None, :] + shift[:, None]
    my = np.tile(np.arange(1000, dtype=np.float32)[:, None], (1, 1000))
    img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    th, *_ = CA.fine_rotation(img, model, 1.4)
    assert abs(th - 1.5) < 0.02


# ---------------------------------------------------------------------------
# End to end, on the organizer's own CAD pipeline
# ---------------------------------------------------------------------------

def _generate(out, split, n, seed, *extra):
    subprocess.run([sys.executable, "generate_cad_varied.py", "--num-samples", str(n), "--split", split,
                    "--output-dir", str(out), "--seed", str(seed), *extra],
                   cwd=I4C, check=True, capture_output=True)
    d = out / split
    return d, list(csv.DictReader(open(d / "manifest.csv")))


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    out = tmp_path_factory.mktemp("cad")
    present = _generate(out, "present", 2, 11, "--no-match-prob", "0", "--max-rotation-deg", "10")
    absent = _generate(out, "absent", 1, 12, "--no-match-prob", "1")
    return present, absent


def test_present_pairs_are_located_to_a_fraction_of_a_pixel(generated):
    (d, rows), _ = generated
    for r in rows:
        img = cv2.imread(str(d / r["search_path"]), cv2.IMREAD_GRAYSCALE)
        res = CA.register(str(d / r["reference_gds_path"]), str(d / r["search_gds_path"]), img)
        assert res.found and res.support > 0.9
        assert math.hypot(res.x - float(r["gt_x"]), res.y - float(r["gt_y"])) < 0.25
        assert abs(res.theta - float(r["gt_theta"])) < 0.03
        assert abs(res.scale - 10.0) < 0.1


def test_an_absent_reference_is_declined_by_the_cad(generated):
    _, (d, rows) = generated
    r = rows[0]
    img = cv2.imread(str(d / r["search_path"]), cv2.IMREAD_GRAYSCALE)
    res = CA.register(str(d / r["reference_gds_path"]), str(d / r["search_gds_path"]), img)
    assert not res.found and res.support < 0.2 and res.score < 0.2


def test_a_useless_search_cad_raises_so_the_caller_can_fall_back(generated, tmp_path):
    import gdstk
    (d, rows), _ = generated
    lib = gdstk.Library()
    cell = lib.new_cell("EMPTY")
    cell.add(gdstk.rectangle((-500, -500), (-400, -400), layer=0))   # outside the frame
    path = tmp_path / "empty.gds"
    lib.write_gds(str(path))
    img = cv2.imread(str(d / rows[0]["search_path"]), cv2.IMREAD_GRAYSCALE)
    with pytest.raises(CA.CadAnchorUnavailable):
        CA.register(str(d / rows[0]["reference_gds_path"]), str(path), img)


def test_phase3_cli_answers_from_the_cad_on_a_blind_split(generated, tmp_path):
    (d, rows), (da, rows_a) = generated
    pairs = tmp_path / "pairs.csv"
    with open(pairs, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "search_path", "reference_gds_path", "search_gds_path",
                    "reference_sem_path", "params_json_path"])
        for tag, dd, rr in (("p", d, rows), ("a", da, rows_a)):
            for r in rr:
                w.writerow([f"{tag}{r['id']}", str(dd / r["search_path"]), str(dd / r["reference_gds_path"]),
                            str(dd / r["search_gds_path"]), "", ""])
    out = tmp_path / "pred.csv"
    subprocess.run([sys.executable, os.path.join(REPO, "phase3.py"), "--input", str(pairs),
                    "--output", str(out), "--quiet"], check=True, capture_output=True)
    pred = {r["pair_id"]: r for r in csv.DictReader(open(out))}
    for r in rows:
        p = pred[f"p{r['id']}"]
        assert p["found"] == "1"
        assert math.hypot(float(p["x"]) - float(r["gt_x"]), float(p["y"]) - float(r["gt_y"])) < 0.25
    assert pred[f"a{rows_a[0]['id']}"]["found"] == "0"
    assert float(pred[f"a{rows_a[0]['id']}"]["score"]) < min(float(pred[f"p{r['id']}"]["score"]) for r in rows)

"""Pixel convention of the reported (x, y) and of the drift-row label (issue #86).

The Phase 2 v2 extension generator writes pixel-centre labels (pixel i spans
[i-0.5, i+0.5]); our generator and the original Phase 2 generator write
pixel-edge labels (pixel i spans [i, i+1)). The decoder works in pixel-edge
coordinates throughout, so "center" must (a) report the same point 0.5 px
up-left and (b) read the raster-drift sample from the row a pixel-centre label
is defined on. "edge" must reproduce the historical output exactly.
"""
import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

from driftsense import matching
from driftsense.matching import LABEL_CONVENTIONS, drift_row_refine, label_row, make_template
from test_subpixel_drift import _apply_row_shift, _pattern, _scene

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------- label_row

@pytest.mark.parametrize("cy, edge_row, center_row", [
    (150.2, 150, 150),     # pixel-centre y 149.7 -> row 150: both agree
    (150.7, 151, 150),     # pixel-centre y 150.2 -> row 150; pixel-edge 150.7 -> 151
    (150.9, 151, 150),
    (151.4, 151, 151),
])
def test_label_row_matches_each_generators_rounding(cy, edge_row, center_row):
    # generate.correct_gt: row_shift[round(py)] with pixel-edge py.
    assert label_row(cy, "edge") == edge_row
    # phase2_pipeline.drift_forward_pt: row_shift[round(y)] with pixel-centre
    # y = cy - 0.5.
    assert label_row(cy, "center") == center_row


def test_label_row_rejects_an_unknown_convention():
    with pytest.raises(ValueError):
        label_row(150.0, "centre")


# ---------------------------------------------------------- drift_row_refine

def _two_row_scene():
    """Nominal-pose scene whose rows 150 and 151 carry opposite drift samples.

    Same construction as test_subpixel_drift._scene, with the two rows either
    side of the label pinned so the row choice is unmistakable: content in a
    row moves by -shift, so row 150 reads -1.2 px and row 151 reads +1.2 px.
    """
    ref_full = _pattern(21)
    search_clean = cv2.resize(ref_full, (100, 100), interpolation=cv2.INTER_AREA)
    canvas = np.full((300, 300), 128, np.uint8)
    canvas[100:200, 100:200] = search_clean
    shift = np.random.default_rng(5).normal(0, 0.2, size=canvas.shape[0])
    shift[150], shift[151] = 1.2, -1.2
    return ref_full, _apply_row_shift(canvas, shift)


def test_center_convention_reads_the_pixel_centre_row():
    ref, search = _two_row_scene()
    tpl = make_template(ref, 10.0, 0.0)
    cx, cy = 150.0, 150.7          # pixel-edge: round -> 151, pixel-centre 150.2 -> 150
    # shrink_sigma=None: this test is about WHICH row is read, so it wants the
    # row's raw offset. The shipped shrinkage (issue #89) scales a correction
    # by how much drift the pair shows overall, and this scene pins two outlier
    # rows into an otherwise quiet frame, so it would report a small fraction
    # of 1.2 px -- correct behaviour, but it would hide the row choice here.
    kw = dict(shrink_sigma=None)
    centre = drift_row_refine(search, tpl, cx, cy, label_convention="center", **kw)
    edge = drift_row_refine(search, tpl, cx, cy, label_convention="edge", **kw)
    assert centre is not None and edge is not None
    assert centre[0] - cx == pytest.approx(-1.2, abs=0.35), "must read row 150"
    assert edge[0] - cx == pytest.approx(+1.2, abs=0.35), "must read row 151"


def test_edge_is_the_default_and_reproduces_the_historical_row():
    ref, search, cx, _, _ = _scene(jitter_sd=1.2, seed=3)
    tpl = make_template(ref, 10.0, 0.0)
    for cy in (150.0, 150.3, 150.7):
        default = drift_row_refine(search, tpl, cx, cy)
        explicit = drift_row_refine(search, tpl, cx, cy, label_convention="edge")
        assert default == explicit


def test_drift_row_refine_rejects_an_unknown_convention():
    ref, search, cx, cy, _ = _scene(jitter_sd=1.0, seed=3)
    tpl = make_template(ref, 10.0, 0.0)
    with pytest.raises(ValueError):
        drift_row_refine(search, tpl, cx, cy, label_convention="pixel")


# ------------------------------------------------------------- locate_phase2

def _stub_decode(monkeypatch, seen):
    """Model-free locate_phase2, same stub family as test_rejector_features."""
    monkeypatch.setattr(matching, "pose_candidates",
                        lambda reference, search, k, **kw: [(10.0, 0.0, .5)])
    monkeypatch.setattr(matching, "canonicalize_search",
                        lambda search, m, r: (search, np.array([[1., 0., 0.], [0., 1., 0.]])))
    monkeypatch.setattr(matching, "locate",
                        lambda *args, **kwargs: {"x": 30.25, "y": 29.75, "score": .7,
                                                 "peak_ratio": .5, "coarse": (30.25, 29.75)})
    monkeypatch.setattr(matching, "refine_zncc",
                        lambda search, template, cx, cy, radius: (cx + 0.125, cy - 0.375, .8))
    monkeypatch.setattr(matching, "polish_pose",
                        lambda reference, search, x, y, m, r: (m, r, 1.0))

    def fake_rows(search, template, cx, cy, **kw):
        seen.append(kw.get("label_convention"))
        return cx + 0.0625, cy

    monkeypatch.setattr(matching, "drift_row_refine", fake_rows)


@pytest.mark.parametrize("subpixel_rows", [False, True])
def test_center_reports_the_edge_answer_half_a_pixel_up_left(monkeypatch, subpixel_rows):
    seen = []
    _stub_decode(monkeypatch, seen)
    reference = np.zeros((100, 100), dtype=np.uint8)
    search = np.zeros((60, 60), dtype=np.uint8)
    kw = dict(verification="zncc", polish=False, subpixel_rows=subpixel_rows)
    edge = matching.locate_phase2(None, reference, search, None, **kw)
    default = matching.locate_phase2(None, reference, search, None, **kw)
    centre = matching.locate_phase2(None, reference, search, None,
                                    label_convention="center", **kw)
    assert (default["x"], default["y"]) == (edge["x"], edge["y"])
    assert centre["x"] == edge["x"] - 0.5
    assert centre["y"] == edge["y"] - 0.5
    for key in ("scale", "theta", "score", "zncc", "confidence"):
        assert centre[key] == edge[key]
    # The row reader is told which convention the label uses, explicitly.
    assert seen == (["edge", "edge", "center"] if subpixel_rows else [])


def test_locate_phase2_rejects_an_unknown_convention():
    with pytest.raises(ValueError):
        matching.locate_phase2(None, np.zeros((100, 100), np.uint8),
                               np.zeros((60, 60), np.uint8), None, label_convention="middle")


def test_signature_default_is_edge_for_internal_evaluators():
    """engine.evaluate and scripts/eval_ext.py score our own pixel-edge data by
    calling locate_phase2 with this default; the submission passes the shipped
    convention explicitly (pinned in test_submission_parity)."""
    import inspect
    assert inspect.signature(matching.locate_phase2).parameters["label_convention"].default == "edge"
    assert inspect.signature(drift_row_refine).parameters["label_convention"].default == "edge"
    assert set(LABEL_CONVENTIONS) == {"edge", "center"}


# ------------------------------------------------- register.py fallback path

def test_register_fallback_path_writes_the_same_convention(tmp_path):
    """The no-weights ZNCC fallback reports pixel-edge coordinates; register.py
    must re-express them exactly like the model path, so the output file never
    mixes conventions."""
    rng = np.random.default_rng(3)
    ref = cv2.GaussianBlur(rng.integers(0, 255, (1000, 1000), dtype=np.uint8), (0, 0), 6)
    search = cv2.resize(ref, (100, 100), interpolation=cv2.INTER_AREA)
    frame = np.full((300, 300), 128, np.uint8)
    frame[80:180, 120:220] = search
    cv2.imwrite(str(tmp_path / "r.png"), ref)
    cv2.imwrite(str(tmp_path / "s.png"), frame)
    (tmp_path / "pairs.csv").write_text("pair_id,reference,search\nP1,r.png,s.png\n")

    rows = {}
    for conv in ("edge", "center"):
        out = tmp_path / f"pred_{conv}.csv"
        subprocess.run([sys.executable, os.path.join(REPO_ROOT, "register.py"),
                        "--input", str(tmp_path / "pairs.csv"), "--output", str(out),
                        "--weights", str(tmp_path / "missing.pt"), "--allow-fallback",
                        "--threshold", "0.0", "--label-convention", conv, "--quiet"],
                       check=True, capture_output=True, timeout=300)
        import csv
        with open(out, newline="") as f:
            rows[conv] = next(csv.DictReader(f))
    assert rows["edge"]["found"] == rows["center"]["found"] == "1"
    assert float(rows["center"]["x"]) == pytest.approx(float(rows["edge"]["x"]) - 0.5, abs=1e-4)
    assert float(rows["center"]["y"]) == pytest.approx(float(rows["edge"]["y"]) - 0.5, abs=1e-4)
    assert rows["center"]["score"] == rows["edge"]["score"]

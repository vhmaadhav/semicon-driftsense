"""Submission-parity tests (reviewer-required, PR #24/#25 review round 2).

Two properties are pinned here, both born from the same review finding:
eval_ext.py had drifted from register.py (threshold 0.2018 vs 0.18, band ON
vs OFF) while its help text claimed alignment -- so the evaluator was not
measuring the shipped system.

(a) **Defaults parity.** register.py and scripts/eval_ext.py must derive
    their defaults from the ONE shared definition (driftsense.config), and
    locate_phase2's signature default must mirror the shipped band setting.

(b) **End-to-end parity.** The SAME synthetic pair run through the batch
    submission path (register.main() writing predictions.csv with the real
    weights and the shipped decode settings) and through the evaluator's
    decode path (locate_phase2 with identical settings, then eval_ext's
    threshold/masking semantics) must agree on the found decision, the
    score, and the pose. The model decode is used on both sides (the
    classical fallback ignores band/verification entirely, so parity
    through it would be vacuous); the decode settings are identical by
    construction -- both sides call locate_phase2 with refine=True,
    verification=SHIPPED_VERIFICATION, band=SHIPPED_BAND and every other
    parameter left at its default.
"""

import argparse
import contextlib
import importlib.util
import inspect
import io
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# The vendored generator is a namespace package rooted at generator/ (see
# tests/conftest.py); the synthetic-pair builder imports src.* from there.
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

from driftsense.config import (SHIPPED_BAND, SHIPPED_THRESHOLD,
                               SHIPPED_VERIFICATION, SHIPPED_CONFIDENCE)


def _load_eval_ext():
    path = os.path.join(REPO_ROOT, "scripts", "eval_ext.py")
    spec = importlib.util.spec_from_file_location("eval_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argparse_defaults(main_fn):
    """Run a script's main() far enough to capture its argparse parser, and
    return the parser so the EFFECTIVE defaults can be read via get_default
    (not restated by hand in this test -- that would just re-drift)."""
    parsers = []
    orig = argparse.ArgumentParser.parse_args

    def spy(self, args=None, namespace=None):
        parsers.append(self)
        raise SystemExit(0)

    argparse.ArgumentParser.parse_args = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            with pytest.raises(SystemExit):
                main_fn()
    finally:
        argparse.ArgumentParser.parse_args = orig
    assert parsers, "main() never reached parse_args"
    return parsers[0]


# ---------------------------------------------------------------------------
# (a) defaults parity
# ---------------------------------------------------------------------------

def test_register_threshold_default_is_the_shared_shipped_value():
    import register
    assert register.DEFAULT_FOUND_THRESHOLD == SHIPPED_THRESHOLD
    # Units change (2026-09-03): the score column is now the fused 6-feature
    # calibrated P(present) (driftsense.calibration.calibrate_shipped), so the
    # threshold lives in probability units, not the legacy min() scale of 0.18
    # (kept only for the no-weights ZNCC fallback; see driftsense/config.py).
    assert SHIPPED_THRESHOLD == pytest.approx(0.4870)
    assert SHIPPED_CONFIDENCE == "fused6"


def test_eval_ext_effective_defaults_match_register():
    ev = _load_eval_ext()
    ev_parser = _argparse_defaults(ev.main)
    assert ev_parser.get_default("threshold") == SHIPPED_THRESHOLD
    # band: --band is opt-in, so the effective default is False == the
    # shipped decode. --no-band still parses (backward-compatible no-op).
    assert ev_parser.get_default("band") is False
    assert ev_parser.get_default("band") == SHIPPED_BAND
    assert ev_parser.get_default("verification") == SHIPPED_VERIFICATION


def test_register_effective_defaults_match_shared_config():
    pytest.importorskip("torch")
    import register  # noqa: F401  (imports infer -> torch at module level)
    reg_parser = _argparse_defaults(register.main)
    assert reg_parser.get_default("threshold") == SHIPPED_THRESHOLD
    assert reg_parser.get_default("verification") == SHIPPED_VERIFICATION


def test_locate_phase2_signature_band_default_is_shipped():
    from driftsense.matching import locate_phase2
    default = inspect.signature(locate_phase2).parameters["band"].default
    assert default is False
    assert default == SHIPPED_BAND


# ---------------------------------------------------------------------------
# (b) end-to-end parity fixture
# ---------------------------------------------------------------------------

Z_TRUE = 9.0
THETA_TRUE = 2.0
SOURCE_CENTRE = (4500.0, 4500.0)
SEARCH_CENTRE = (500.0, 500.0)
CANVAS_PX = 9000


def _affine_centring_source_at(z, theta_deg):
    """Same invertible construction as tests/test_scale_semantics.py:
    p_search = (1/z) R(theta) (p_canvas - c_canvas) + c_search, with the
    translation solved so SOURCE_CENTRE lands at SEARCH_CENTRE."""
    t = np.deg2rad(theta_deg)
    A = np.array([[np.cos(t), np.sin(t)],
                  [-np.sin(t), np.cos(t)]]) / z
    M = np.zeros((2, 3), dtype=float)
    M[:, :2] = A
    M[:, 2] = (np.array(SEARCH_CENTRE, dtype=float)
               - A @ np.array(SOURCE_CENTRE, dtype=float))
    return M


def _make_synthetic_pair(tmp_path):
    """A DRAM-canvas pair with a known planted pose: the reference is the
    source crop whose image the affine plants at the search centre."""
    cv2 = pytest.importorskip("cv2")
    from src.patterns.dram import generate_dram_canvas
    from src.presets import get_preset

    rng = np.random.default_rng(26)
    canvas = generate_dram_canvas(CANVAS_PX, get_preset("dram_1x"), 10.0, rng)

    M = _affine_centring_source_at(Z_TRUE, THETA_TRUE)
    search = cv2.warpAffine(canvas, M, (1000, 1000), flags=cv2.INTER_LINEAR)

    half = 500
    cx, cy = int(SOURCE_CENTRE[0]), int(SOURCE_CENTRE[1])
    reference = canvas[cy - half:cy + half, cx - half:cx + half]

    rp = str(tmp_path / "reference.png")
    sp = str(tmp_path / "search.png")
    cv2.imwrite(rp, reference)
    cv2.imwrite(sp, search)
    return rp, sp


def _run_register_batch(tmp_path, rp, sp):
    """Run the real batch submission path: register.main() over a pairs.csv
    pointing at the pair, shipping defaults (threshold/band/verification
    from driftsense.config via register.py). Returns the predictions row."""
    csv_path = tmp_path / "pairs.csv"
    out_path = tmp_path / "predictions.csv"
    csv_path.write_text(
        "pair_id,reference,search\nP0001," + rp + "," + sp + "\n")

    import register
    argv = ["register.py", "--input", str(csv_path), "--output", str(out_path),
            "--quiet"]
    old_argv = sys.argv
    sys.argv = argv
    try:
        register.main()
    finally:
        sys.argv = old_argv

    import csv as _csv
    with open(out_path, newline="") as f:
        rows = list(_csv.DictReader(f))
    assert len(rows) == 1
    return rows[0]


def _eval_decode(rp, sp):
    """The evaluator's decode path with the shipped settings: locate_phase2
    with exactly the arguments register.py passes and every other parameter
    at its (now parity-pinned) default."""
    pytest.importorskip("torch")
    import cv2
    import infer as I
    from driftsense.matching import locate_phase2

    loaded = I.load_model(os.path.join(REPO_ROOT, "weights", "driftsense.pt"))
    assert loaded is not None, (
        "infer.load_model returned None for weights/driftsense.pt: with torch "
        "present the real ship checkpoint is expected to instantiate (a None "
        "here hides a broken ship path, so this is a failure, not a skip)")
    model, device = loaded

    ref = cv2.imread(rp, cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(sp, cv2.IMREAD_GRAYSCALE)
    return locate_phase2(model, ref, sea, device, refine=True,
                         verification=SHIPPED_VERIFICATION, band=SHIPPED_BAND)


def test_end_to_end_submission_parity(tmp_path):
    """The same pair through both paths: identical found decision, identical
    score, identical pose (to float tolerance), plus the zero-fill semantics
    on a decline and eval_ext's pred_found agreement."""
    ev = _load_eval_ext()
    rp, sp = _make_synthetic_pair(tmp_path)

    row = _run_register_batch(tmp_path, rp, sp)
    res = _eval_decode(rp, sp)

    # Both sides must report the SHIPPED confidence statistic (the fused
    # 6-feature calibrated P(present); formerly min(network score, native
    # ZNCC)) -- the column register.py writes and eval_ext's _worker records.
    # Parity holds by construction: both read res["confidence"], which
    # locate_phase2 fills via driftsense.config.SHIPPED_CONFIDENCE.
    score_eval = float(res.get("confidence", res.get("score")))
    found_eval = int(score_eval >= SHIPPED_THRESHOLD)
    found_csv = int(row["found"])
    assert found_csv == found_eval, (
        "found decision diverged: register wrote %s, evaluator derives %s "
        "(score %.6f vs threshold %s)" % (found_csv, found_eval, score_eval,
                                          SHIPPED_THRESHOLD))

    # The CSV score is written with %.6f; compare at that precision.
    assert float(row["score"]) == pytest.approx(score_eval, abs=1e-6)

    if found_csv == 1:
        for col, key in (("x", "x"), ("y", "y"), ("theta", "theta"),
                         ("scale", "scale")):
            assert float(row[col]) == pytest.approx(float(res[key]), abs=1e-3), \
                "pose column %s diverged: %s vs %s" % (col, row[col], res[key])
    else:
        # Zero-fill contract (slide 5): a declined answer zeroes every
        # pose/location field, which is exactly what eval_ext's masking
        # models when it computes credit.
        for col in ("x", "y", "theta", "scale"):
            assert float(row[col]) == 0.0

    # eval_ext's threshold/masking semantics agree with the CSV decision:
    # one row, set A, gt = the planted truth, scored by eval_ext.score().
    df = pd.DataFrame([{
        "set": "A", "gt_found": 1,
        "score": float(row["score"]),
        "x": float(row["x"]), "y": float(row["y"]),
        "gt_x": SEARCH_CENTRE[0], "gt_y": SEARCH_CENTRE[1],
        "scale": float(row["scale"]), "theta": float(row["theta"]),
        "gt_scale": Z_TRUE, "gt_rot": THETA_TRUE,
    }])
    _, masked = ev.score(df, SHIPPED_THRESHOLD, quiet=True)
    assert int(masked.pred_found.iloc[0]) == found_csv

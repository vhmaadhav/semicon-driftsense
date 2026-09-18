#!/usr/bin/env python3
"""Pin the evaluation report's arithmetic to the official rubric scorer.

`evaluation/ap25_score.py` recomputes every rubric component from the
frozen per-pair CSV rather than transcribing the scorer's output, so the PDF's
numbers are independently derived. That is only a virtue if the two agree --
otherwise the document quietly reports a different score than the one the
judging harness computes.

This test runs `judging/score_rubric.py::score` (the implementation that
produced the Phase 2 campaign's ranking table) and this repo's recomputation
over the same frozen inputs, and asserts they match to floating-point
tolerance. It runs whenever `judging/` sits beside this checkout, and skips
cleanly when it does not (CI, a fresh clone).

    python -m pytest tests/test_eval_report_parity.py -q
"""
from __future__ import annotations

import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGING = "/Users/sachin/Development/semicon/judging"
REPORT = os.path.join(HERE, "evaluation", "ap25_score.py")
DATA = os.path.join(HERE, "evaluation", "ap25")


def _load_report_module():
    """The pure scoring core -- no reportlab/matplotlib needed, so this runs in CI."""
    spec = importlib.util.spec_from_file_location("ap25_score", REPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_official_scorer():
    spec = importlib.util.spec_from_file_location(
        "score_rubric", os.path.join(JUDGING, "score_rubric.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pytestmark = pytest.mark.skipif(
    not os.path.exists(os.path.join(JUDGING, "score_rubric.py")),
    reason="judging/score_rubric.py is not present beside this checkout")


def test_report_matches_official_scorer():
    import pandas as pd

    rep = _load_report_module()
    official = _load_official_scorer()

    per, _comp, _base, _ab = rep.load()
    ours = rep.score_of(per)

    # --- rebuild the official scorer's frame the way score_rubric.main does
    pred = pd.read_csv(os.path.join(DATA, "predictions.csv"))
    gt = pd.read_csv(os.path.join(DATA, "ground_truth.csv"))
    man = pd.read_csv(os.path.join(DATA, "manifest_jury.csv"))
    gt = gt.rename(columns={"theta": "gt_rot"})
    gt["gt_found"] = gt["present"]
    df = pred.merge(man[["pair_id", "set"]], on="pair_id", how="left")
    df = df.merge(gt[["pair_id", "gt_found", "x", "y", "gt_rot", "scale"]],
                  on="pair_id", how="left", suffixes=("", "_gt"))
    df["gt_x"], df["gt_y"], df["gt_scale"] = df["x_gt"], df["y_gt"], df["scale_gt"]
    for c in ("x", "y", "theta", "scale", "score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    res, _df, subtotal = official.score(df, None, quiet=True)

    pairs = [
        ("setA_loc_credit", ours["a"], res["setA_loc_credit"]),
        ("setB_loc_credit", ours["b"], res["setB_loc_credit"]),
        # score() returns (value, points) tuples for the weighted components.
        ("localisation", ours["loc"], res["localisation"][0]),
        ("scale", ours["sc"], res["scale"][0]),
        ("rotation", ours["rc"], res["rotation"][0]),
        ("rejection", ours["f1"], res["rejection"][0]),
        ("calibration", ours["auc"], res["calibration"][0]),
        ("setD_credit", ours["d"], res["setD_credit"][0]
         if isinstance(res["setD_credit"], tuple) else res["setD_credit"]),
        ("subtotal85", ours["subtotal"], subtotal),
    ]
    for name, mine, theirs in pairs:
        assert float(mine) == pytest.approx(float(theirs), abs=1e-9), (
            f"{name}: report {mine} != official {theirs}")

    assert ours["bonus6"] == res["bonus6"]
    assert ours["bonus4"] == res["bonus4"]
    assert ours["tp"] == res["rej_tp"]
    assert ours["fp"] == res["rej_fp"]
    assert ours["fn"] == res["rej_fn"]

    official_total = subtotal + 6 * res["bonus6"] + 4 * res["bonus4"]
    assert ours["total"] == pytest.approx(official_total, abs=1e-9)


def test_headline_is_the_one_the_report_prints():
    """The number a reader sees at the top of the PDF is the scorer's total."""
    rep = _load_report_module()
    per, _c, _b, _ab = rep.load()
    sc = rep.score_of(per)
    assert sc["total"] == pytest.approx(82.71111111111111, abs=1e-6)
    assert sc["bonus6"] is True
    assert sc["bonus4"] is False


def test_pose_credit_excludes_set_d():
    """Set D is bonus-only: it must never enter the 10+10 pose blocks."""
    rep = _load_report_module()
    per, _c, _b, _ab = rep.load()
    sc = rep.score_of(per)
    assert all(s != "D" for s in sc["ok"]["set"]), (
        "Set D leaked into the pose-credit pool")

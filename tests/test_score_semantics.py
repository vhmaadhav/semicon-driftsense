"""Issue #22 P0: eval_ext.score() must match optimize_threshold.points()
semantics -- a present pair that the decode DECLINES (score < threshold)
earns zero localisation and zero pose credit, because register.py writes
zero fields for declined answers and the grader scores what is submitted.

Regression fixture: four pairs covering accepted-present, declined-present,
accepted-absent, declined-absent.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def _load():
    path = os.path.join(REPO_ROOT, "scripts", "eval_ext.py")
    spec = importlib.util.spec_from_file_location("eval_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _row(set_, gt_found, score, perfect=True):
    return {"pair_id": f"{set_}{gt_found}{score:.2f}", "set": set_,
            "gt_found": gt_found,
            "x": 500.0 if perfect else 800.0, "y": 500.0 if perfect else 800.0,
            "gt_x": 500.0, "gt_y": 500.0,
            "scale": 10.0, "gt_scale": 10.0,
            "theta": 0.0, "gt_rot": 0.0,
            "score": score}


def _df():
    return pd.DataFrame([
        _row("A", 1, 0.9),    # present, accepted, perfect  -> full loc + pose
        _row("B", 1, 0.1),    # present, DECLINED, perfect coords -> must score ZERO
        _row("A", 0, 0.9),    # absent, accepted            -> false negative (fn)
        _row("B", 0, 0.1),    # absent, declined            -> true rejection (tp)
    ])


def _subtotal(out):
    return float([l for l in out.splitlines() if "SUBTOTAL" in l][0].split()[-1])


def test_declined_present_pair_earns_zero_loc_and_pose(capsys):
    mod = _load()
    mod.score(_df(), 0.2018)
    out = capsys.readouterr().out
    total = _subtotal(out)
    # Corrected semantics, by hand:
    #   loc: A present accepted = 1, B present declined = 0 -> L = 0.45*1 + 0.55*0 = 0.45 -> 18
    #   pose: only the accepted-present A row -> S = R = 1 -> 20
    #   F1(reject): tp=1 (B absent), fp=1 (B present declined), fn=1 (A absent) -> 0.5 -> 7.5
    #   AUC: correct present (score .9) vs incorrect (.9, .1): >,= -> 0.5 -> 5
    assert total == pytest.approx(18 + 20 + 7.5 + 5, abs=0.01)


def test_accepted_present_pair_still_scores_full(capsys):
    """Guard against over-masking: accepted present pairs keep full credit."""
    mod = _load()
    df = pd.DataFrame([_row("A", 1, 0.9), _row("B", 1, 0.9),
                       _row("A", 0, 0.9), _row("B", 0, 0.1)])
    mod.score(df, 0.2018)
    out = capsys.readouterr().out
    total = _subtotal(out)
    # loc: both present accepted perfect -> L = 1 -> 40; pose -> 20
    # F1: tp=1, fp=0, fn=1 -> 2/3 -> 10; AUC: correct (.9) vs incorrect (.9,.1) -> 0.75 -> 7.5
    assert total == pytest.approx(40 + 20 + 10 + 7.5, abs=0.01)

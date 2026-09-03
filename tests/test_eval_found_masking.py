"""Regression tests for the found-mask in eval_ext.score() (issue #22 P0).

register.py writes zero pose/location fields for a declined answer, so a
pair the system DECLINED but that is actually PRESENT must earn zero
localisation credit -- and therefore zero pose credit. The scorer used to
credit such pairs with their geometric credit as if the answer had been
given, inflating every reported total.

Four fixtures, each with hand-computed expected totals pinned as literals
(derived from the published tier tables in eval_ext.py, NOT from running
the implementation):

  LOC_TIERS:    err <= 1px -> 1.0, <= 2px -> 0.8, <= 3px -> 0.6, <= 5px -> 0.4
  SCALE_TIERS:  s_err <= 0.01 -> 1.0, <= 0.02 -> 0.6, <= 0.05 -> 0.3
  ROT_TIERS:    r_err <= 0.25 -> 1.0, <= 0.50 -> 0.6, <= 1.00 -> 0.3
  Localisation: 0.45 * mean(credit A) + 0.55 * mean(credit B), x40 points
  Pose:         mean tier over present pairs with credit > 0, 10 pts each

Threshold used throughout: the shipped operating point, imported from
driftsense.config (the ONE shared definition register.py and eval_ext.py both
consume -- see tests/test_submission_parity.py).
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.config import SHIPPED_THRESHOLD as THRESHOLD


def _load():
    path = os.path.join(REPO_ROOT, "scripts", "eval_ext.py")
    spec = importlib.util.spec_from_file_location("eval_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pair(set_, gt_found, score, x, y, gt_x, gt_y, scale, theta,
          gt_scale, gt_rot):
    return {
        "set": set_, "gt_found": gt_found, "score": score,
        "x": x, "y": y, "gt_x": gt_x, "gt_y": gt_y,
        "scale": scale, "theta": theta, "gt_scale": gt_scale, "gt_rot": gt_rot,
    }


def _mk_df(rows):
    return pd.DataFrame(rows)


def _score(mod, rows):
    df = _mk_df(rows)
    res, _ = mod.score(df, THRESHOLD, quiet=True)
    return res


def test_accepted_present_earns_localisation_and_pose():
    """Both present pairs said 'found' (score >= threshold): geometric
    credit flows through untouched.

    Pair 1 (set A): exact hit -> err 0px -> credit 1.0;
                    s_err 0.000 -> 1.0; r_err 0.00 -> 1.0.
    Pair 2 (set B): err 2px  -> credit 0.8;
                    s_err 0.008 -> 1.0; r_err 0.50 -> 0.6.
                    (s_err 0.008 chosen instead of 0.010 because the latter
                    is 0.010000000000000009 in binary FP, a hair above the
                    <=0.01 tier bound, and would legitimately earn 0.6.)

    loc credit = 0.45*1.0 + 0.55*0.8 = 0.89          -> 35.6 pts
    pose ok    = both pairs
    scale      = (1.0 + 1.0)/2 = 1.0                 -> 10.0 pts
    rotation   = (1.0 + 0.6)/2 = 0.8                 ->  8.0 pts
    """
    mod = _load()
    res = _score(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.75,  52.0,  80.0,  50.0,  80.0, 1.008, 0.50, 1.0, 0.0),
    ])
    assert res["localisation"][1] == pytest.approx(35.6, abs=1e-9)
    assert res["scale"][1] == pytest.approx(10.0, abs=1e-9)
    assert res["rotation"][1] == pytest.approx(8.0, abs=1e-9)


def test_declined_present_earns_zero_localisation_and_zero_pose():
    """The P0 case: both present pairs are perfect geometrically but the
    system DECLINED them (score < threshold). Their raw geometric credit
    (1.0 each) must be zeroed BEFORE any subset is taken, so:

      loc credit = 0.45*0.0 + 0.55*0.0 = 0.0         ->  0.0 pts
      pose ok    = empty (no present pair kept credit) -> nan points

    Under the pre-mask scorer this frame scored 40.0 localisation and
    20.0 pose -- the inflation this test pins to zero.
    """
    mod = _load()
    res = _score(mod, [
        _pair("A", 1, 0.10, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.15,  50.0,  80.0,  50.0,  80.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert res["localisation"][1] == 0.0
    assert np.isnan(res["scale"][1])
    assert np.isnan(res["rotation"][1])


def test_accepted_absent_hurts_rejection_only():
    """An ABSENT pair said 'found' must not touch localisation/pose (it is
    not in 'present'), but it must zero the reject-positive F1.

    Present pairs as in test_accepted_present (loc 35.6, pose 10+8).
    gray F1(reject) @ fixed threshold: pf = [1, 1, 1] over
    gt_found = [1, 1, 0] -> tp=0, fp=0, fn=1 -> F1 = 0.0 -> 0.0 pts.
    """
    mod = _load()
    res = _score(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.75,  52.0,  80.0,  50.0,  80.0, 1.008, 0.50, 1.0, 0.0),
        _pair("C", 0, 0.90,   0.0,   0.0,  99.0,  99.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert res["localisation"][1] == pytest.approx(35.6, abs=1e-9)
    assert res["scale"][1] == pytest.approx(10.0, abs=1e-9)
    assert res["rotation"][1] == pytest.approx(8.0, abs=1e-9)
    assert res["rejection"][0] == 0.0
    assert res["rejection"][1] == 0.0


def test_declined_present_set_d_earns_zero_bonus_credit():
    """Set D regression (PR #24/#25 review): the bonus path must read the
    SAME submission-masked credit as A/B/C.

    A declined PRESENT-D pair with perfect raw coordinates (err 0px -> raw
    geometric credit 1.0) must earn 0 D credit, because register.py submits
    found=0 with zeroed pose/location for it -- the credit was never
    submittable. The pre-fix scorer read unmasked df.loc_credit here and
    awarded the full 1.0 (the Set D hole in the #22 P0 mask).

    The mask is applied to the FULL df before the gray/D split, so this
    fixture exercises exactly the path the printed bonus block uses.
    """
    mod = _load()
    res, df = mod.score(_mk_df([
        _pair("D", 1, 0.10, 500.0, 300.0, 500.0, 300.0, 1.000, 0.00, 1.0, 0.0),
    ]), THRESHOLD, quiet=True)
    d = df[df["set"] == "D"]
    dp = d[d.gt_found == 1]
    dc = dp.loc_credit.mean()
    assert dp.pred_found.iloc[0] == 0          # declined at the shipped threshold
    assert dp.loc_credit.iloc[0] == 0.0        # masked, not the raw 1.0
    assert dc == 0.0                           # D bonus credit is zero
    # And the raw geometric credit really was perfect -- the zero comes from
    # the mask, not from a bad fixture.
    assert float(dp.err.iloc[0]) == 0.0


def test_accepted_present_set_d_keeps_its_credit():
    """Control for the D mask: a PRESENT-D pair SAID found with a perfect
    hit keeps its raw credit through the same masked path."""
    mod = _load()
    res, df = mod.score(_mk_df([
        _pair("D", 1, 0.90, 500.0, 300.0, 500.0, 300.0, 1.000, 0.00, 1.0, 0.0),
    ]), THRESHOLD, quiet=True)
    d = df[df["set"] == "D"]
    dp = d[d.gt_found == 1]
    assert dp.pred_found.iloc[0] == 1
    assert dp.loc_credit.iloc[0] == pytest.approx(1.0, abs=1e-9)


def test_declined_absent_is_credited_in_rejection():
    """An ABSENT pair correctly declined earns reject-F1 credit; the
    declined PRESENT pair in the same frame still earns zero loc/pose.

    Pairs: A present accepted (score .90, err 0 -> credit 1.0);
           B present declined (score .10, err 0 -> credit zeroed);
           C absent declined  (score .05).

    loc credit = 0.45*1.0 + 0.55*0.0 = 0.45          -> 18.0 pts
    pose ok    = A only: scale 1.0 -> 10.0; rotation 1.0 -> 10.0
    F1(reject): pf = [1, 0, 0] over gt_found = [1, 1, 0]
                -> tp=1, fp=1, fn=0 -> F1 = 2/3     -> 10.0 pts
    """
    mod = _load()
    res = _score(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.10,  50.0,  80.0,  50.0,  80.0, 1.000, 0.00, 1.0, 0.0),
        _pair("C", 0, 0.05,   0.0,   0.0,  99.0,  99.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert res["localisation"][1] == pytest.approx(18.0, abs=1e-9)
    assert res["scale"][1] == pytest.approx(10.0, abs=1e-9)
    assert res["rotation"][1] == pytest.approx(10.0, abs=1e-9)
    assert res["rejection"][0] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert res["rejection"][1] == pytest.approx(10.0, abs=1e-9)

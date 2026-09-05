"""The rubric has ONE implementation, and both evaluators call it.

`scripts/eval_ext.py` and `scripts/eval_phase2.py` carried two independent
scorers that disagreed on the two things that inflate a score:

  * submission masking -- eval_phase2 credited localisation and pose on
    present pairs the system had DECLINED, which register.py submits with
    zero-filled pose/location columns;
  * A/B strata -- eval_phase2 pooled the two present-pair sets instead of
    applying the published 0.45 A / 0.55 B weighting.

Because of that, a component delta measured by eval_phase2 could not be
multiplied by rubric weights and compared against an eval_ext subtotal --
which is exactly what the 80-pair G2 smoke test did. These tests pin the
shared scorer (`driftsense.rubric.score`) and the two behaviours.

Expected values are hand-computed from the published tier tables, not read
back from the implementation:

  LOC_TIERS:   err <= 1px -> 1.0, <= 2px -> 0.8, <= 3px -> 0.6, <= 5px -> 0.4
  SCALE_TIERS: s_err <= 0.01 -> 1.0, <= 0.02 -> 0.6, <= 0.05 -> 0.3
  ROT_TIERS:   r_err <= 0.25 -> 1.0, <= 0.50 -> 0.6, <= 1.00 -> 0.3
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense import rubric
from driftsense.config import SHIPPED_THRESHOLD as THRESHOLD


def _load(name):
    path = os.path.join(REPO_ROOT, "scripts", f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _pair(gt_found, score, x, y, gt_x, gt_y, scale, theta,
          gt_scale=1.0, gt_rot=0.0, set_=None):
    row = {"gt_found": gt_found, "score": score, "x": x, "y": y,
           "gt_x": gt_x, "gt_y": gt_y, "scale": scale, "theta": theta,
           "gt_scale": gt_scale, "gt_rot": gt_rot}
    if set_ is not None:
        row["set"] = set_
    return row


# --------------------------------------------------------------------------
# One definition
# --------------------------------------------------------------------------

def test_eval_ext_scorer_is_the_shared_scorer():
    """eval_ext must not carry a private copy of the rubric."""
    ev = _load("eval_ext")
    assert ev.score is rubric.score
    assert ev.tier is rubric.tier
    assert (ev.LOC_TIERS, ev.SCALE_TIERS, ev.ROT_TIERS) == (
        rubric.LOC_TIERS, rubric.SCALE_TIERS, rubric.ROT_TIERS)
    assert (ev.W_A, ev.W_B) == (rubric.W_A, rubric.W_B) == (0.45, 0.55)


def test_eval_phase2_imports_the_shared_scorer():
    """eval_phase2 must reach the rubric through the shared module, and must
    no longer define the private credit ladders it used to."""
    src = open(os.path.join(REPO_ROOT, "scripts", "eval_phase2.py")).read()
    assert "from driftsense.rubric import score" in src
    for gone in ("def loc_credit(", "def scale_credit(", "def rot_credit("):
        assert gone not in src, f"{gone} is a second rubric definition"


# --------------------------------------------------------------------------
# Submission masking, on the unlabelled (pooled) path eval_phase2 uses
# --------------------------------------------------------------------------

def test_pooled_declined_present_earns_zero_localisation():
    """Two geometrically perfect present pairs, one DECLINED.

    Accepted pair: err 0px -> credit 1.0. Declined pair: masked to 0.0.
    Pooled localisation = (1.0 + 0.0)/2 = 0.5 -> 20.0 pts.

    The pre-share eval_phase2 scored this frame 40.0: it took the mean over
    `found == 1` rows of the *manifest*, so a declined pair kept its credit.
    """
    res, _ = rubric.score(pd.DataFrame([
        _pair(1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.0, 0.0),
        _pair(1, 0.10, 300.0, 400.0, 300.0, 400.0, 1.0, 0.0),
    ]), THRESHOLD, quiet=True)
    assert res["localisation"][0] == pytest.approx(0.5)
    assert res["localisation"][1] == pytest.approx(20.0)


def test_pooled_pose_is_scored_only_where_localisation_earned_credit():
    """Pose follows the mask: the declined pair's perfect scale/rotation is
    not submittable, so only the accepted pair's pose counts.

    Accepted: s_err 0.008 -> 1.0, r_err 0.50 -> 0.6.
    Declined: masked out of `ok` entirely, despite exact pose.
    """
    res, _ = rubric.score(pd.DataFrame([
        _pair(1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.008, 0.50),
        _pair(1, 0.10, 300.0, 400.0, 300.0, 400.0, 1.000, 0.00),
    ]), THRESHOLD, quiet=True)
    assert res["scale"][1] == pytest.approx(10.0)
    assert res["rotation"][1] == pytest.approx(6.0)


# --------------------------------------------------------------------------
# Strata: weighted when labelled, pooled (not silently 0.45-weighted) when not
# --------------------------------------------------------------------------

def test_labelled_frame_uses_the_published_ab_weights():
    """A credit 1.0, B credit 0.8 -> 0.45*1.0 + 0.55*0.8 = 0.89 -> 35.6."""
    res, _ = rubric.score(pd.DataFrame([
        _pair(1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.0, 0.0, set_="A"),
        _pair(1, 0.90,  52.0,  80.0,  50.0,  80.0, 1.0, 0.0, set_="B"),
    ]), THRESHOLD, quiet=True)
    assert res["localisation"][1] == pytest.approx(35.6)


def test_unlabelled_frame_is_pooled_not_partially_weighted():
    """The regression this guards: an unlabelled frame must not be scored as
    if it were all Set A.

    Both present pairs earn credit 1.0. Pooled -> 1.0 -> 40.0 points.
    Feeding the same frame through the A/B path would give
    0.45*1.0 + 0.55*nan = nan.
    """
    df = pd.DataFrame([
        _pair(1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.0, 0.0),
        _pair(1, 0.90, 300.0, 400.0, 300.0, 400.0, 1.0, 0.0),
    ])
    assert "set" not in df.columns
    res, scored = rubric.score(df, THRESHOLD, quiet=True)
    assert res["localisation"][0] == pytest.approx(1.0)
    assert res["localisation"][1] == pytest.approx(40.0)
    assert not np.isnan(res["localisation"][1])


def test_pooled_absent_pairs_still_drive_rejection_f1():
    """Unlabelled absent pairs must land in the grayscale subset the F1 is
    taken over -- otherwise a split with no `set` column would report F1 on
    present pairs alone.

    Two present (accepted), two absent: one correctly declined, one accepted.
    reject-positive: tp=1 (correct decline), fp=0, fn=1 (missed absent)
    F1 = 2*1 / (2*1 + 0 + 1) = 0.6667 -> 10.0 pts.
    """
    res, _ = rubric.score(pd.DataFrame([
        _pair(1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.0, 0.0),
        _pair(1, 0.90, 300.0, 400.0, 300.0, 400.0, 1.0, 0.0),
        _pair(0, 0.05, 0.0, 0.0, np.nan, np.nan, 1.0, 0.0),
        _pair(0, 0.90, 0.0, 0.0, np.nan, np.nan, 1.0, 0.0),
    ]), THRESHOLD, quiet=True)
    assert res["rejection"][0] == pytest.approx(2 / 3)
    assert res["rejection"][1] == pytest.approx(10.0)


def test_pooled_and_labelled_agree_when_the_split_is_one_stratum():
    """Sanity: a frame whose present pairs are all Set A scores the same
    pooled as it does labelled-with-weight-1. This is what makes the
    pooled path a strata statement rather than a different rubric."""
    rows = [_pair(1, 0.90, 100.5, 200.0, 100.0, 200.0, 1.0, 0.0),
            _pair(1, 0.90, 302.0, 400.0, 300.0, 400.0, 1.0, 0.0)]
    pooled, _ = rubric.score(pd.DataFrame(rows), THRESHOLD, quiet=True)
    labelled, _ = rubric.score(
        pd.DataFrame([dict(r, set="A") for r in rows]), THRESHOLD, quiet=True)
    # 0.45-weighted A-only is nan; pooled is the plain mean of 1.0 and 0.8.
    assert pooled["localisation"][0] == pytest.approx(0.9)
    assert np.isnan(labelled["localisation"][0])

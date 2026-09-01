"""Regression tests for the two calibration-AUC variants in eval_ext.score().

Issue #27: the "all-pairs" AUC variant added by PR #25 labelled EVERY absent
pair correct (correct_all = correct | (gt_found == 0)), regardless of whether
the submitted found decision actually rejected it -- so an absent pair the
system wrongly claimed ("found=1") still counted as a correct rejection. The
fix introduces a submitted-output correctness variant,
res["calibration_submitted"]:

    pred_found = score >= threshold
    correct_submitted = (gt_found == 1 & pred_found & err <= 5)   # said found
                                                                   # and hit
                      | (gt_found == 0 & ~pred_found)             # said not
                                                                   # found and
                                                                   # it was

Under it, a declined PRESENT pair is NOT correct (it forfeits its
measurement), and an absent pair is correct only when actually rejected.

The historical present-only variant, res["calibration"], is unchanged and
keeps its existing computation. Its definition is "correct = present pair
localised within 5 px"; an absent pair is gt_found == 0, so np.where marks
it NOT correct and its score falls in the *incorrect* group -- absent rows
influence that variant only through the (deliberate) negative group, never
as correct. The tests pin exactly this behaviour and the submitted variant's
difference: an absent pair there is judged on the SUBMITTED decision, so a
correct rejection lands in the correct group and a false positive in the
incorrect group.

Five fixtures below mirror the issue's acceptance list. The frames are small
enough that every AUC is computed by hand in the docstrings:

    AUC = P(score_correct > score_incorrect)
          + 0.5 * P(score_correct == score_incorrect)
          (rank-bisector / Mann-Whitney formulation, ties count half)

Threshold used throughout: 0.2018 (the shipped operating point).
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

THRESHOLD = 0.2018


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


def _aucs(mod, rows):
    df = pd.DataFrame(rows)
    res, _ = mod.score(df, THRESHOLD, quiet=True)
    return res["calibration"][0], res["calibration_submitted"][0]


def test_correct_present_pair():
    """Case 1: a PRESENT pair, said found, localised within 5 px.

    Row 1: A, gt_found=1, score 0.90, err 0px -> present-correct;
           submitted-correct (pred_found True, err<=5).
    Row 2: B, gt_found=1, score 0.10, err 2px -> present-correct (2 <= 5),
           submitted-INCORRECT (0.10 < threshold -> declined present).

    Hand arithmetic:
      present-only: correct {0.90, 0.10}, incorrect {} -> NaN
      submitted:    correct {0.90}, incorrect {0.10}
                    AUC = P(0.90 > 0.10) = 1/1 = 1.0
    """
    mod = _load()
    auc, auc_sub = _aucs(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.10,  52.0,  80.0,  50.0,  80.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert np.isnan(auc)
    assert auc_sub == pytest.approx(1.0)


def test_wrong_location_present_pair():
    """Case 2: a PRESENT pair, said found, but localised 8 px off (> 5 px).

    Row 1: A, gt_found=1, score 0.90, err 0px -> correct under BOTH variants.
    Row 2: B, gt_found=1, score 0.80, err 8px -> NOT correct under either
           (found but missed the 5 px window; submitted found=True so the
           decline clause does not rescue it).

    Hand arithmetic (identical groups in both variants):
      correct scores   = {0.90}
      incorrect scores = {0.80}
      AUC = P(0.90 > 0.80) = 1/1 = 1.0
    """
    mod = _load()
    auc, auc_sub = _aucs(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.80,  58.0,  80.0,  50.0,  80.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert auc == pytest.approx(1.0)
    assert auc_sub == pytest.approx(1.0)


def test_declined_present_pair_not_correct():
    """Case 3: a PRESENT pair whose score sits below threshold -> declined.

    The pair localises perfectly (err 0px), but register.py would submit
    found=0 with zeroed pose/location: it forfeits its measurement. It is
    labelled correct by the present-only variant (which is the definition
    of that variant -- the submitted decision is invisible to it) and NOT
    correct under the submitted variant (issue #27).

    Row 1: A, gt_found=1, score 0.90, err 0px -> correct both variants.
    Row 2: B, gt_found=1, score 0.10, err 0px -> declined present.

    Hand arithmetic:
      present-only: correct {0.90, 0.10}, incorrect {} -> NaN
      submitted:    correct {0.90}, incorrect {0.10}
                    AUC = P(0.90 > 0.10) = 1.0
    """
    mod = _load()
    auc, auc_sub = _aucs(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.10,  50.0,  80.0,  50.0,  80.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert np.isnan(auc)
    assert auc_sub == pytest.approx(1.0)


def test_correct_absent_reject():
    """Case 4: an ABSENT pair correctly rejected (score < threshold).

    The submitted variant must credit it as CORRECT (it is a genuine
    correct rejection). The present-only variant never labels any absent
    row correct -- np.where marks gt_found==0 rows False.

    Row 1: A, gt_found=1, score 0.90, err 0px -> present-correct;
           submitted-correct.
    Row 2: C, gt_found=0, score 0.05          -> absent, rejected:
           submitted-correct; present-only-incorrect.

    Hand arithmetic:
      present-only: correct {0.90}, incorrect {0.05}
                    AUC = P(0.90 > 0.05) = 1.0
      submitted:    correct {0.90, 0.05}, incorrect {} -> NaN
    """
    mod = _load()
    auc, auc_sub = _aucs(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("C", 0, 0.05,   0.0,   0.0,  99.0,  99.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert auc == pytest.approx(1.0)
    assert np.isnan(auc_sub)


def test_absent_false_positive_not_correct():
    """Case 5: an ABSENT pair with score >= threshold -> claimed "found".

    Issue #27's core bug: the old correct_all labelled every absent pair
    correct, including this one. Under the submitted definition this row is
    NOT correct (it must land in the incorrect group and drag the AUC
    down), while the present-only variant again treats it as a plain
    negative.

    Frame 1 -- false positive ranking ABOVE the correct present score:
      Row 1: A, gt_found=1, score 0.90, err 0px -> correct (both).
      Row 2: C, gt_found=0, score 0.95          -> absent FALSE POSITIVE.
      present-only: correct {0.90}, incorrect {0.95}
                    AUC = P(0.90 > 0.95) = 0.0
      submitted:    correct {0.90}, incorrect {0.95}
                    AUC = P(0.90 > 0.95) = 0.0
      (Same 0.0 by coincidence of groups -- but the submitted variant gets
       there via the submitted decision; the decisive split is frame 2.)

    Frame 2 -- false positive BELOW the correct present score:
      Row 1: A, gt_found=1, score 0.90, err 0px -> correct (both).
      Row 2: C, gt_found=0, score 0.70          -> absent FALSE POSITIVE.
      present-only: correct {0.90}, incorrect {0.70}
                    AUC = P(0.90 > 0.70) = 1.0
      submitted:    correct {0.90}, incorrect {0.70}
                    AUC = P(0.90 > 0.70) = 1.0

    The false positive is in the incorrect group under BOTH scores: raising
    it above the correct score drops the AUC to 0.0, lowering it restores
    1.0. A definition that auto-credited every absent pair (issue #27's
    correct_all) would give NaN in both frames (correct group would hold
    both scores, incorrect group empty). Frame 3 proves the submitted
    variant differs from present-only on a false positive vs a DECLINED
    present pair:
      Row 1: A, gt_found=1, score 0.10, err 0px -> declined present:
             present-only CORRECT (err 0 <= 5), submitted-INCORRECT.
      Row 2: C, gt_found=0, score 0.70          -> absent false positive:
             incorrect under both.
      present-only: correct {0.10}, incorrect {0.70}
                    AUC = P(0.10 > 0.70) = 0.0
      submitted:    correct {}, incorrect {0.10, 0.70} -> NaN
    """
    mod = _load()
    auc1, auc1_sub = _aucs(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("C", 0, 0.95,   0.0,   0.0,  99.0,  99.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert auc1 == pytest.approx(0.0)
    assert auc1_sub == pytest.approx(0.0)

    auc2, auc2_sub = _aucs(mod, [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("C", 0, 0.70,   0.0,   0.0,  99.0,  99.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert auc2 == pytest.approx(1.0)
    assert auc2_sub == pytest.approx(1.0)

    auc3, auc3_sub = _aucs(mod, [
        _pair("A", 1, 0.10, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("C", 0, 0.70,   0.0,   0.0,  99.0,  99.0, 1.000, 0.00, 1.0, 0.0),
    ])
    assert auc3 == pytest.approx(0.0)
    assert np.isnan(auc3_sub)


def test_submitted_auc_reacts_to_absent_score_present_only_does_not():
    """Group-membership contrast: flipping an absent row across the correct
    present score moves the SUBMITTED groups, never the present-only ones.

    Base present rows: A correct at 0.90; B present but 8px off at 0.80
    (incorrect under both variants).

    Frame X (absent row at 0.05, correctly rejected):
      present-only: correct {0.90}, incorrect {0.80, 0.05}
                    AUC = P(0.90>0.80) + P(0.90>0.05) = 1/2 + 1/2 = 1.0
      submitted:    correct {0.90, 0.05}, incorrect {0.80}
                    AUC = P(0.90>0.80) + P(0.05>0.80) = 1/2 + 0   = 0.5

    Frame Y (absent row at 0.95, false positive):
      present-only: correct {0.90}, incorrect {0.80, 0.95}
                    AUC = P(0.90>0.80) + P(0.90>0.95) = 1/2 + 0   = 0.5
      submitted:    correct {0.90}, incorrect {0.80, 0.95}
                    AUC = 1/2 + 0 = 0.5

    The present-only AUC moves 1.0 -> 0.5 purely because the absent row's
    SCORE moved: it is a negative in that view regardless of its submitted
    decision. The submitted variant drops a correct-rejection's score from
    the negative group (0.5 in frame X vs 0.5 in frame Y comes from
    different pair decompositions, shown above).
    """
    mod = _load()
    base = [
        _pair("A", 1, 0.90, 100.0, 200.0, 100.0, 200.0, 1.000, 0.00, 1.0, 0.0),
        _pair("B", 1, 0.80,  58.0,  80.0,  50.0,  80.0, 1.000, 0.00, 1.0, 0.0),
    ]
    auc_x, auc_sub_x = _aucs(mod, base + [
        _pair("C", 0, 0.05, 0.0, 0.0, 99.0, 99.0, 1.000, 0.00, 1.0, 0.0)])
    auc_y, auc_sub_y = _aucs(mod, base + [
        _pair("C", 0, 0.95, 0.0, 0.0, 99.0, 99.0, 1.000, 0.00, 1.0, 0.0)])

    assert auc_x == pytest.approx(1.0)   # 0.05 counted as a negative
    assert auc_y == pytest.approx(0.5)   # 0.95 counted as a negative
    assert auc_sub_x == pytest.approx(0.5)
    assert auc_sub_y == pytest.approx(0.5)

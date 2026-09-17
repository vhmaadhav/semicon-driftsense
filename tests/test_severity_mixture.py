"""Severity-mixture draws in the blind-grade emulation (G2).

`grade_emulation.py --b-mix` constrains the Set B draw to a severity
distribution, so the A70/B70/C40 emulation can answer G2's question: what
happens to the +4 bonus gate when Set B is weighted toward severity 3-4
instead of being uniform, as our ext_p2 pool is.

The tests that matter here are the ones that check the mixture was
**realised**, not merely requested. Asking for a severity and getting
something else is this project's most expensive recurring defect -- it
retracted the 81.45/81.93 figures and it is what
tests/test_severity_pin_guard.py guards at the generator CLI. A mixture that
silently drew uniformly would make every number in the G2 campaign a
restatement of the baseline.
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense import rubric as shared


def _load():
    path = os.path.join(REPO_ROOT, "scripts", "grade_emulation.py")
    spec = importlib.util.spec_from_file_location("grade_emulation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GE = _load()


def _frame(n_per_level=100, seed=0):
    """A/B/C frame with severity levels 1-4 and severity-driven behaviour.

    Set B pairs at severity 4 are placed badly (20 px out) and scored below
    the shipped threshold; severity 1 pairs are exact and confident. That
    makes the drawn mixture visible in the score: if --b-mix is ignored, the
    level-4-only and level-1-only arms come out the same.
    """
    rng = np.random.RandomState(seed)
    rows = []
    for set_name in ("A", "B", "C"):
        for level in (1, 2, 3, 4):
            for i in range(n_per_level):
                present = set_name != "C"
                hard = (set_name == "B") and level == 4
                err = 20.0 if hard else 0.0
                rows.append({
                    "pair_id": f"{set_name}{level}_{i}", "set": set_name,
                    "severity": level, "gt_found": int(present),
                    "score": 0.05 if (hard or not present) else 0.9,
                    "x": 100.0 + err, "y": 200.0,
                    "gt_x": 100.0 if present else np.nan,
                    "gt_y": 200.0 if present else np.nan,
                    "scale": 1.0, "theta": 0.0,
                    "gt_scale": 1.0, "gt_rot": 0.0,
                })
    return pd.DataFrame(rows).sample(frac=1.0, random_state=rng).reset_index(drop=True)


# --------------------------------------------------------------------------
# Apportionment
# --------------------------------------------------------------------------

def test_counts_are_used_verbatim_when_they_sum_to_the_quota():
    assert GE.severity_quotas("7,14,24,25", 70) == {1: 7, 2: 14, 3: 24, 4: 25}


def test_weights_are_apportioned_to_an_exact_quota():
    """Largest remainder: 10,15,35,40 of 70 -> exact 7, 10.5, 24.5, 28 ->
    floors 7,10,24,28 = 69, one seat left. The remainders tie at 0.5 (L2 and
    L3) and the tie goes to the HIGHER severity level, so L3 takes it."""
    q = GE.severity_quotas("10,15,35,40", 70)
    assert sum(q.values()) == 70
    assert q == {1: 7, 2: 10, 3: 25, 4: 28}


def test_ties_do_not_systematically_favour_the_easier_levels():
    """The bias this guards: with four equal weights every remainder ties at
    0.5, and breaking those ties toward the low levels draws 48.6% at
    severity 3-4 when 50% was asked for -- easier data than requested, on
    every tie, silently."""
    q = GE.severity_quotas("1,1,1,1", 70)
    assert q[3] + q[4] >= q[1] + q[2]


def test_uniform_sweep_point_reproduces_the_pool_composition():
    """frac_34 = 0.50 is the ext_p2 pool as-is: exactly half the draw at 3-4,
    not 34 or 36 of 70."""
    q = GE.severity_quotas(GE.sweep_mix(0.50), 70)
    assert sum(q.values()) == 70
    assert q[3] + q[4] == 35


@pytest.mark.parametrize("frac", GE.SWEEP_FRACTIONS)
def test_every_sweep_point_hits_its_severity_fraction_exactly(frac):
    q = GE.severity_quotas(GE.sweep_mix(frac), 70)
    assert sum(q.values()) == 70
    assert (q[3] + q[4]) / 70 == pytest.approx(frac, abs=1 / 70)


def test_bad_mixes_are_refused():
    with pytest.raises(ValueError, match="needs 4 numbers"):
        GE.severity_quotas("10,20,70", 70)
    with pytest.raises(ValueError, match="negative"):
        GE.severity_quotas("10,-20,35,35", 70)
    with pytest.raises(ValueError, match="sums to zero"):
        GE.severity_quotas("0,0,0,0", 70)


# --------------------------------------------------------------------------
# The mixture is REALISED, not just requested
# --------------------------------------------------------------------------

def test_blocks_are_pure_and_large_enough():
    df = _frame()
    sets, sev = df["set"].to_numpy(), df["severity"].to_numpy()
    blocks = GE._blocks_for(sets, sev, "B", 70, "0,0,50,50")
    assert len(blocks) == 4
    got = {}
    for block, want in blocks:
        levels = set(sev[block].tolist())
        assert len(levels) == 1, "a block must hold exactly one severity level"
        assert set(sets[block].tolist()) == {"B"}
        got[levels.pop()] = want
        assert len(block) >= want
    assert got == {1: 0, 2: 0, 3: 35, 4: 35}


def test_a_mixture_that_the_pool_cannot_supply_is_refused_before_drawing():
    df = _frame(n_per_level=10)
    sets, sev = df["set"].to_numpy(), df["severity"].to_numpy()
    with pytest.raises(ValueError, match="cannot draw this mixture"):
        GE._blocks_for(sets, sev, "B", 70, "0,0,50,50")


def test_mixture_without_a_severity_column_is_refused_not_ignored():
    df = _frame().drop(columns=["severity"])
    with pytest.raises(ValueError, match="no `severity` column"):
        GE.bootstrap(df, thresholds=[0.18], draws=2, mixes={"B": "0,0,50,50"})


def test_the_mixture_changes_the_grade_in_the_direction_the_data_implies():
    """End-to-end: on a frame where only severity-4 B pairs fail, drawing
    all-level-4 must score strictly worse than all-level-1, and the uniform
    draw must sit between them. A mixture silently ignored would make all
    three equal."""
    df = _frame()
    kw = dict(thresholds=[0.18], draws=200, seed=3)
    easy = GE.bootstrap(df, mixes={"B": "100,0,0,0"}, **kw)[0]
    unif = GE.bootstrap(df, **kw)[0]
    hard = GE.bootstrap(df, mixes={"B": "0,0,0,100"}, **kw)[0]
    assert easy["e_total"] > unif["e_total"] > hard["e_total"]
    # Every drawn B pair is a declined present pair in the hard arm, so Set B
    # earns zero localisation: loc = 0.45*1.0 + 0.55*0.0 = 0.45 -> 18 pts.
    assert hard["e_total"] < easy["e_total"] - 15


def test_default_path_is_unchanged_by_the_mixture_option():
    """The omitted, empty and None mixture options must remain equivalent.
    Independent per-set streams deliberately supersede the old sequence."""
    df = _frame()
    a = GE.bootstrap(df, thresholds=[0.18], draws=100, seed=11)[0]
    b = GE.bootstrap(df, thresholds=[0.18], draws=100, seed=11, mixes={})[0]
    c = GE.bootstrap(df, thresholds=[0.18], draws=100, seed=11, mixes=None)[0]
    assert a["e_total"] == b["e_total"] == c["e_total"]
    assert a["p_f1_ge_bonus"] == b["p_f1_ge_bonus"] == c["p_f1_ge_bonus"]


# --------------------------------------------------------------------------
# The fast rubric and the shared rubric must agree
# --------------------------------------------------------------------------

def test_grade_emulation_rubric_agrees_with_the_shared_scorer():
    """grade_emulation keeps its own numpy rubric because it runs 20,000
    times per sweep point; that makes it a third implementation of the
    tiers. It has to agree with driftsense.rubric.score on the same frame,
    or the sweep and the scorecard describe different systems."""
    df = _frame()
    gray = df[df["set"].isin(("A", "B", "C"))]
    fast = GE.rubric(gray, 0.18)
    res, _ = shared.score(gray, 0.18, quiet=True)
    assert fast["loc"] == pytest.approx(res["localisation"][0])
    assert fast["scale"] == pytest.approx(res["scale"][0])
    assert fast["rot"] == pytest.approx(res["rotation"][0])
    assert fast["f1_reject"] == pytest.approx(res["rejection"][0])
    assert fast["auc"] == pytest.approx(res["calibration"][0])
    assert fast["total"] == pytest.approx(sum(
        v[1] for k, v in res.items() if k != "calibration_submitted"))


def test_changing_b_mixture_keeps_actual_a_c_draws_fixed(monkeypatch):
    """A single shared RNG would let B's blocks perturb the subsequent C draw."""
    df = _frame()
    # Unique scores identify the real rows delivered to the F1 consumer.
    df['score'] = (np.arange(len(df)) + .5) / len(df)
    lookup = df.set_index('score')
    seen = []
    original = GE._f1

    def trace(score, gt, t, positive):
        selected = lookup.loc[score]
        assert selected['set'].value_counts().to_dict() == {'A': 70, 'B': 70, 'C': 40}
        seen.append({s: tuple(selected.loc[selected['set'] == s, 'pair_id'])
                     for s in ('A', 'C')})
        return original(score, gt, t, positive)

    monkeypatch.setattr(GE, '_f1', trace)
    arms = []
    for mix in (None, {'B': '70,0,0,0'}, {'B': '0,0,35,35'}):
        seen.clear()
        GE.bootstrap(df, thresholds=[.18], draws=3, seed=7, mixes=mix)
        arms.append(list(seen))
    assert arms[0] == arms[1] == arms[2]


# --------------------------------------------------------------------------
# the realised-severity audit has to fail closed across the whole ladder
# --------------------------------------------------------------------------

def _severity_module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "scripts", "severity_breakdown.py")
    spec = importlib.util.spec_from_file_location("severity_breakdown", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _manifest(tmp_path, per_level, name="m.csv"):
    """One synthetic manifest; per_level maps level -> (jitter, speckle, det, band)."""
    rows = []
    for lv, (jit, spk, det, (blo, bhi)) in per_level.items():
        for i in range(10):
            frac = i / 9.0
            rows.append(dict(phase2_set="B", severity_level=lv,
                             severity_continuous=blo + (bhi - blo) * frac,
                             drift_jitter_px=jit, speckle_sigma=spk,
                             detector_noise_sigma_search=det))
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_a_genuine_four_rung_ladder_passes(tmp_path):
    mod = _severity_module()
    man = _manifest(tmp_path, {
        1: (0.8, 0.010, 0.010, (0.00, 0.20)),
        2: (1.3, 0.016, 0.016, (0.26, 0.45)),
        3: (1.9, 0.023, 0.023, (0.51, 0.70)),
        4: (2.3, 0.030, 0.030, (0.76, 1.00)),
    })
    assert mod.realised_severity_audit(man) == {"B": True}


def test_a_flat_middle_rung_is_caught(tmp_path):
    """L1 -> L4 moves, but L2 and L3 sit on top of each other.

    The old check compared only L4 against L1 and passed this.
    """
    mod = _severity_module()
    man = _manifest(tmp_path, {
        1: (0.8, 0.010, 0.010, (0.00, 0.20)),
        2: (1.5, 0.020, 0.020, (0.26, 0.45)),
        3: (1.5, 0.020, 0.020, (0.51, 0.70)),
        4: (2.3, 0.030, 0.030, (0.76, 1.00)),
    })
    assert mod.realised_severity_audit(man) == {"B": False}


def test_a_missing_level_fails_closed(tmp_path):
    """Three rungs is not a four-level ladder, however well they progress."""
    mod = _severity_module()
    man = _manifest(tmp_path, {
        1: (0.8, 0.010, 0.010, (0.00, 0.20)),
        2: (1.3, 0.016, 0.016, (0.26, 0.45)),
        4: (2.3, 0.030, 0.030, (0.76, 1.00)),
    })
    assert mod.realised_severity_audit(man) == {"B": False}


def test_materially_overlapping_bands_fail(tmp_path):
    """Means progress, but the levels are not distinct rungs."""
    mod = _severity_module()
    man = _manifest(tmp_path, {
        1: (0.8, 0.010, 0.010, (0.00, 0.90)),
        2: (1.3, 0.016, 0.016, (0.05, 0.95)),
        3: (1.9, 0.023, 0.023, (0.10, 0.97)),
        4: (2.3, 0.030, 0.030, (0.15, 1.00)),
    })
    assert mod.realised_severity_audit(man) == {"B": False}

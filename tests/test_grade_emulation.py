"""tests/test_grade_emulation.py -- stratified blind-grade emulation.

Covers (PR #24 review blocker 4):
  * the exact stratified draw (70/70/40 from a synthetic 875/875/500 frame;
    deterministic for a fixed seed; sets never mix; clear error on a short set);
  * the credit-tier function edges (1/2/3/5 px; 0.25/0.5/1.0 deg; 1/2/5%);
  * one hand-computed present-positive F1 case.

NOT covered here: the full bootstrap statistics (slow) -- only a tiny
draws=200 smoke test. Torch-free by design (stdlib + numpy + pandas only).
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
    path = os.path.join(REPO_ROOT, "scripts", "grade_emulation.py")
    spec = importlib.util.spec_from_file_location("grade_emulation", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _synthetic_frame():
    """Full 2250-pair frame with the real composition: 875 A, 875 B, 500 C
    (no D -- the grade draw is grayscale-only). Scores/coords are simple
    deterministic functions of the row index."""
    counts = {"A": 875, "B": 875, "C": 500}
    rows = []
    i = 0
    for s, n in counts.items():
        for k in range(n):
            rows.append({
                "pair_id": f"test_{s}_{k:08d}",
                "set": s,
                "gt_found": 1 if s in ("A", "B") else 0,
                "score": 0.20 + 0.60 * ((k * 37) % 100) / 100.0,
                "x": 100.0 + 0.05 * (k % 7),
                "y": 200.0 - 0.03 * (k % 5),
                "gt_x": 100.0, "gt_y": 200.0,
                "scale": 1.0, "theta": 0.0,
                "gt_scale": 1.0, "gt_rot": 0.0,
            })
            i += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def frame():
    return _synthetic_frame()


@pytest.fixture(scope="module")
def mod():
    return _load()


# ---------------------------------------------------------------------------
# Stratified draw
# ---------------------------------------------------------------------------

def test_draw_is_exact_composition(mod, frame):
    draw = mod.stratified_draw(frame, seed=0)
    assert len(draw) == 180
    assert dict(draw["set"].value_counts()) == {"A": 70, "B": 70, "C": 40}


def test_draw_is_deterministic_per_seed(mod, frame):
    d0a = mod.stratified_draw(frame, seed=0)
    d0b = mod.stratified_draw(frame, seed=0)
    d1 = mod.stratified_draw(frame, seed=1)
    pd.testing.assert_frame_equal(d0a, d0b)
    # A different seed draws different pairs from the same pools.
    assert set(d0a["pair_id"]) != set(d1["pair_id"]) or not d0a["pair_id"].equals(
        d1.reset_index(drop=True)["pair_id"]
    )


def test_draw_never_mixes_sets(mod, frame):
    draw = mod.stratified_draw(frame, seed=7)
    # Every drawn pair keeps the identity of its source pool -- i.e. each
    # drawn row must exist among the frame rows of ITS OWN set.
    for s in ("A", "B", "C"):
        pool = set(frame.loc[frame["set"] == s, "pair_id"])
        drawn = set(draw.loc[draw["set"] == s, "pair_id"])
        assert len(drawn) == {"A": 70, "B": 70, "C": 40}[s]
        assert drawn <= pool


def test_draw_is_without_replacement_and_order_stable(mod, frame):
    draw = mod.stratified_draw(frame, seed=3)
    assert draw["pair_id"].is_unique
    # Order-stable: rows appear in the original frame order.
    pos = {pid: i for i, pid in enumerate(frame["pair_id"])}
    order = [pos[p] for p in draw["pair_id"]]
    assert order == sorted(order)


def test_draw_raises_on_short_set(mod, frame):
    # Drop C rows until set C is below its 40-pair quota (A/B stay full).
    short = frame[frame["set"] != "C"].copy()
    c_rows = frame[frame["set"] == "C"].head(39)
    short = pd.concat([short, c_rows])
    assert (short["set"] == "C").sum() == 39
    with pytest.raises(ValueError, match="quota"):
        mod.stratified_draw(short, seed=0)


# ---------------------------------------------------------------------------
# Credit-tier edges
# ---------------------------------------------------------------------------

def test_loc_tier_edges(mod):
    tiers = mod.LOC_TIERS
    assert mod.tier(0.0, tiers) == 1.00
    assert mod.tier(1.0, tiers) == 1.00      # <=1 px: full credit
    assert mod.tier(1.0001, tiers) == 0.80
    assert mod.tier(2.0, tiers) == 0.80
    assert mod.tier(2.5, tiers) == 0.60
    assert mod.tier(3.0, tiers) == 0.60
    assert mod.tier(3.1, tiers) == 0.40
    assert mod.tier(5.0, tiers) == 0.40
    assert mod.tier(5.0001, tiers) == 0.0    # past the last bound


def test_rot_tier_edges(mod):
    tiers = mod.ROT_TIERS
    assert mod.tier(0.25, tiers) == 1.00
    assert mod.tier(0.26, tiers) == 0.60
    assert mod.tier(0.50, tiers) == 0.60
    assert mod.tier(0.51, tiers) == 0.30
    assert mod.tier(1.00, tiers) == 0.30
    assert mod.tier(1.01, tiers) == 0.0


def test_scale_tier_edges(mod):
    tiers = mod.SCALE_TIERS
    assert mod.tier(0.01, tiers) == 1.00
    assert mod.tier(0.011, tiers) == 0.60
    assert mod.tier(0.02, tiers) == 0.60
    assert mod.tier(0.021, tiers) == 0.30
    assert mod.tier(0.05, tiers) == 0.30
    assert mod.tier(0.06, tiers) == 0.0


# ---------------------------------------------------------------------------
# Hand-computed F1 case (present-as-positive)
# ---------------------------------------------------------------------------

def _mini_frame():
    # 6 grayscale pairs. Scores chosen so that at t=0.5 the found flag is
    # [1, 1, 1, 0, 0, 0] against gt_found [1, 1, 0, 1, 0, 0]:
    #   tp=2 (p0,p1), fp=1 (p2), fn=1 (p3)
    #   F1(present) = 2*2 / (2*2 + 1 + 1) = 0.6667
    #   F1(reject)  = tp=2 (p4,p5), fp=1 (p3), fn=1 (p2) = 0.6667
    # p0 is 10 px away (x=11 vs gt 1) so its loc credit is 0 despite being
    # found; every other pair is exact.
    return pd.DataFrame({
        "pair_id": [f"p{i}" for i in range(6)],
        "set": ["A", "A", "A", "B", "B", "B"],
        "gt_found": [1, 1, 0, 1, 0, 0],
        "score": [0.9, 0.6, 0.7, 0.4, 0.2, 0.3],
        "x": [11.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "y": [1.0] * 6,
        "gt_x": [1.0] * 6, "gt_y": [1.0] * 6,
        "scale": [1.0] * 6, "theta": [0.0] * 6,
        "gt_scale": [1.0] * 6, "gt_rot": [0.0] * 6,
    })


def test_f1_hand_computed(mod):
    g = _mini_frame()
    assert mod.f1_found(g, 0.5) == pytest.approx(2 * 2 / (4 + 1 + 1))
    assert mod.f1_reject(g, 0.5) == pytest.approx(2 * 2 / (4 + 1 + 1))


def test_f1_never_found_is_zero_present_positive(mod):
    g = _mini_frame()
    # t=2.0: nothing is called found -> F1(present) = 0.
    assert mod.f1_found(g, 2.0) == 0.0
    # t=0.0: everything is called found -> F1(reject) = 0.
    assert mod.f1_reject(g, 0.0) == 0.0


def test_rubric_declined_present_pair_earns_zero_loc(mod):
    """Corrected semantics: a present pair below the threshold gets zero
    localisation (and therefore zero pose), because register.py writes no
    pose/location fields for a declined answer."""
    g = _mini_frame()
    # p1: present, 0.6 px away, score 0.6 >= 0.5 -> full loc credit.
    # p3: present, 0.0 px away, score 0.4 < 0.5 -> DECLINED -> 0 credit.
    r = mod.rubric(g, 0.5)
    # loc A: p0,p1,p2 -> present: p0 (10.0 px away -> 0 credit), p1 (1.0),
    # p2 absent. mean = (0 + 1) / 2 = 0.5
    # loc B: p3 present but declined -> 0. mean = 0.
    assert r["loc_A"] == pytest.approx(0.5)
    assert r["loc_B"] == pytest.approx(0.0)
    assert r["loc"] == pytest.approx(0.45 * 0.5 + 0.55 * 0.0)
    # Pose is scored only where loc credit > 0: only p1 qualifies there,
    # with 0% scale error and 0 deg rotation -> full credit.
    assert r["scale"] == pytest.approx(1.0)
    assert r["rot"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Bootstrap smoke test (tiny; the statistics themselves are NOT asserted)
# ---------------------------------------------------------------------------

def test_bootstrap_smoke(mod, frame):
    out = mod.bootstrap(frame, thresholds=[0.5], draws=200, seed=0)
    assert len(out) == 1
    r = out[0]
    assert r["draws"] == 200
    assert 0.0 <= r["p_f1_ge_bonus"] <= 1.0
    assert np.isfinite(r["e_total"])
    assert np.isfinite(r["e_total_plus_bonus"])
    assert r["e_total_plus_bonus"] == pytest.approx(
        r["e_total"] + 4.0 * r["p_f1_ge_bonus"])
    # Deterministic for a fixed seed.
    out2 = mod.bootstrap(frame, thresholds=[0.5], draws=200, seed=0)
    assert out2[0]["p_f1_ge_bonus"] == r["p_f1_ge_bonus"]
    assert out2[0]["e_total"] == pytest.approx(r["e_total"])

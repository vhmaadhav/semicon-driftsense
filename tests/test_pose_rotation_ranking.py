"""Task 2 (inference-efficiency plan): rotation-aware scale ranking in
pose_candidates, plus the E3 grid-pruning equality guard.

The failure mode (competitor pair p008, same class): the first-stage scale
scan ranks at rot=0.0 only, so on a ROTATED pair the true basin can rank
below the hypothesis cut while same-family unrotated decoy lattices fill the
top-k -- at rot=0 the rotated template correlates poorly at every scale, so
a wrong-scale alias wins. The fix re-ranks the top surviving rot=0 peaks by
their own best-rotation score (the same per-peak scan the refinement stage
already paid for) BEFORE choosing which k advance to the golden-section
refine. The full joint 17x11 scale-rotation grid is deliberately NOT built:
the coarse sweep is 66.8% of pair time (issue #7) and a joint grid would
multiply the dominant cost by the rotation count (plan ruling).

The synthetic frame here is constructed so the p008 structure is exact and
verifiable:

* the reference is a dram_1x canvas crop (the vendored generator, as in
  tests/test_scale_semantics.py), planted at z=9.0 rotated 3.0 deg in the
  frame centre;
* four corner tiles are resampled canvas regions at z in {8.0, 10.25,
  11.25, 11.75} -- same-family lattice decoys whose aliases beat the true
  basin at rot=0 (each ~0.64-0.76 vs the true basin's 0.39) but LOSE to it
  once each peak is judged at its best rotation (each alias is an unrotated
  lattice, so its best-rotation score equals its rot=0 score, while the true
  basin rises to ~0.93 at the planted angle);
* edge tiles are flat noise so they cannot create spurious peaks.

Under the OLD ranking (replicated below from the pre-change code) the true
basin is a rot=0 peak ranked 4th -- the old top-3 excludes it entirely. The
re-rank must put it first.

The E3 pruning gate is pinned two ways here: test_pruned_scan_returns_exhaustive_topk
keeps the plan's equality requirement on this fixture (identical top-k
ordering against the exhaustive scan), and the gate's own margin semantics
are pinned directly as unit asserts on _odd_point_pruned (deep valley pruned,
shallow neighbour kept, strict boundary at margin * kth) plus a smoke that
the wired margin still returns k hypotheses. The gate's firing behaviour on
real data is measured by the full 2,250-pair audit, not by this fixture,
where the shipped 0.5 margin correctly never fires.
"""

import os
import sys
from functools import lru_cache

import numpy as np
import cv2
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# The vendored generator is a namespace package rooted at generator/ (see
# tests/conftest.py).
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

from driftsense.matching import (  # noqa: E402
    COARSE_SCALES,
    _odd_point_pruned,
    _peak_score,
    _probe,
    _refine_pose_local,
    make_template,
    pose_candidates,
)
from generator.src.patterns.dram import generate_dram_canvas  # noqa: E402
from generator.src.presets import get_preset  # noqa: E402

Z_TRUE = 9.0
THETA_TRUE = 3.0          # rotated pair: the p008 mode needs rotation != 0
ALIAS_SCALES = [8.0, 10.25, 11.25, 11.75]
ALIAS_BLUR = 2.6          # places the aliases between the true basin's
# rot-0 score and its best-rotation score
CANVAS_PX = 9000

# Candidate prune margins for the equality audit below. Deliberately literal
# constants in the test, NOT the production E3_PRUNE_MARGIN: the shipped
# default is None (exhaustive) until the full-2,250 audit, so auditing that
# constant would compare None against itself. 0.5 is the conservative margin
# earmarked for the audit; 0.65 is aggressive enough to exercise the gate.
AUDIT_PRUNE_MARGINS = (0.5, 0.65)


@lru_cache(maxsize=1)
def _fixture():
    """Return (reference, search) for the p008-shaped synthetic frame."""
    rng = np.random.default_rng(26)
    canvas = generate_dram_canvas(CANVAS_PX, get_preset("dram_1x"), 10.0, rng)

    # p_search = (1/z) R(theta) (p_canvas - c_canvas) + c_search, solved for
    # the canvas centre mapping to the frame centre (the same construction
    # tests/test_scale_semantics.py pins).
    t = np.deg2rad(THETA_TRUE)
    A = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]]) / Z_TRUE
    M = np.zeros((2, 3))
    M[:, :2] = A
    M[:, 2] = np.array([500.0, 500.0]) - A @ np.array([4500.0, 4500.0])
    warped = cv2.warpAffine(canvas, M, (1000, 1000), flags=cv2.INTER_LINEAR)
    reference = canvas[4000:5000, 4000:5000]
    # Instance degradation so the alias band can sit above its rot-0 score
    # (a clean instance would beat every decoy at rot=0 and there would be
    # no failure mode to fix).
    warped = np.clip(warped.astype(np.float64)
                     + np.random.default_rng(5).normal(0, 12.0, warped.shape),
                     0, 255).astype(np.uint8)

    half = 250

    def alias_tile(z_alias, seed):
        size = int(half * z_alias)
        r = np.random.default_rng(seed)
        y = int(r.integers(0, CANVAS_PX - size))
        x = int(r.integers(0, CANVAS_PX - size))
        region = canvas[y:y + size, x:x + size]
        tile = cv2.resize(region, (half, half),
                          interpolation=cv2.INTER_AREA).astype(np.float32)
        # The blur emulates a different-process lattice: it weakens the tile's
        # correlation with the (crisp) reference, which is what keeps the
        # alias scores in the band between the true basin's rot-0 and
        # best-rotation values instead of dominating them outright.
        return np.clip(cv2.GaussianBlur(tile, (0, 0), ALIAS_BLUR),
                       0, 255).astype(np.uint8)

    def weak_tile(seed):
        r = np.random.default_rng(seed)
        return np.clip(110.0 + r.normal(0, 20.0, (half, half)),
                       0, 255).astype(np.uint8)

    frame = np.zeros((1000, 1000), np.uint8)
    corners = [(0, 0), (0, 500), (500, 0), (500, 500)]
    edges = [(0, half), (half, 0), (500, half), (half, 500)]
    for i, (y0, x0) in enumerate(corners):
        frame[y0:y0 + half, x0:x0 + half] = alias_tile(ALIAS_SCALES[i], 100 + i)
    for i, (y0, x0) in enumerate(edges):
        frame[y0:y0 + half, x0:x0 + half] = weak_tile(200 + i)
    frame[half:1000 - half, half:1000 - half] = \
        warped[half:1000 - half, half:1000 - half]
    return reference, frame


def _old_ranking_topk(reference, search, k=3):
    """The pre-change rot=0-only ranking, replicated from HEAD~.

    Uses only helpers this task did not touch (make_template, _probe,
    _peak_score, _refine_pose_local), so the replication cannot silently
    drift with the new code: if the helpers change semantics, the fixture
    assertions below fail loudly rather than passing vacuously.
    """
    lo_s, hi_s = 8.0, 12.0
    lo_r, hi_r = -5.0, 5.0
    probe_search = _probe(search)

    def coarse(f, r=0.0):
        t = _probe(make_template(reference, f, r))
        return _peak_score(probe_search, t)

    grid = np.linspace(lo_s, hi_s, COARSE_SCALES)
    vals = [coarse(f) for f in grid]
    peaks = [j for j in range(len(grid))
             if (j == 0 or vals[j] >= vals[j - 1])
             and (j == len(grid) - 1 or vals[j] >= vals[j + 1])]
    peaks.sort(key=lambda j: -vals[j])
    top = peaks[:k]
    out = []
    for j in top:
        f0 = float(grid[j])
        r0 = float(max(np.linspace(lo_r, hi_r, 11), key=lambda r: coarse(f0, r)))
        out.append(_refine_pose_local(reference, search, f0, r0,
                                      (hi_s - lo_s) / 16.0,
                                      (hi_r - lo_r) / 10.0,
                                      (lo_s, hi_s), (lo_r, hi_r)))
    return out


def test_fixture_exhibits_p008_structure():
    """The fixture must genuinely reproduce the failure mode: under the OLD
    rot=0-only ranking the true basin misses the top-k cut."""
    pytest.importorskip("cv2")
    reference, search = _fixture()
    old = _old_ranking_topk(reference, search, k=3)
    old_scales = [f for f, _, _ in old]
    assert not any(abs(f - Z_TRUE) <= 0.25 for f in old_scales), (
        "fixture regressed: the old rot=0-only ranking no longer misses the "
        "true basin (top-3 scales %s include z=9.0)" % old_scales)


def test_rerank_rescues_rotated_basin():
    """With rotation-aware re-ranking the true basin must be in the top-k,
    and ranked first: its best-rotation score separates it from the aliases."""
    pytest.importorskip("cv2")
    reference, search = _fixture()
    out = pose_candidates(reference, search, k=3, band=False,
                          prune_margin=None)
    scales = [f for f, _, _ in out]
    assert any(abs(f - Z_TRUE) <= 0.35 for f in scales), (
        "re-ranked top-3 %s does not contain the true basin z=9.0" % scales)
    # The true basin's best-rotation score (~0.93) beats every alias's
    # best-rotation score (~their rot-0 values, <=0.76), so it must lead.
    f_best, r_best, _ = out[0]
    assert abs(f_best - Z_TRUE) <= 0.35, (
        "winner scale %.3f is not the true basin" % f_best)
    assert abs(r_best - THETA_TRUE) <= 0.4, (
        "winner rotation %.3f is not the planted %.1f deg" % (r_best, THETA_TRUE))


def test_pruned_scan_returns_exhaustive_topk():
    """E3 pruning (equality-audit requirement, plan task 2): the neighbour
    gate must not change the returned hypotheses on this fixture.

    Two explicit margins: the conservative candidate margin 0.5 (on this
    fixture the gate may not fire at all, so the equality is cheap insurance)
    and an aggressive 0.65 whose gate semantics are pinned directly by
    test_odd_point_pruned_margin_semantics. Both must return the exhaustive
    top-k. The margins are AUDIT_PRUNE_MARGINS, literal constants in this
    module, NOT read from the production E3_PRUNE_MARGIN: the shipped default
    is deliberately None until the full-2,250 audit, and looping over that
    constant would make this an equality check of None vs exhaustive --
    vacuously true. The `__defaults__` assert below pins that exhaustive
    default; the binding equality check remains the full 2,250-pair audit;
    this pins the mechanism, not the ship margin."""
    pytest.importorskip("cv2")
    # The production default must stay exhaustive until the promised audit;
    # if that default ever changes, this audit's margins must be revisited.
    assert pose_candidates.__defaults__[-1] is None, (
        "pose_candidates no longer defaults to prune_margin=None; the "
        "candidate margins audited below were chosen for the exhaustive-"
        "default regime and the full 2,250-pair equality audit must be "
        "re-run before shipping any enabled default")
    reference, search = _fixture()
    exhaustive = pose_candidates(reference, search, k=3, band=False,
                                 prune_margin=None)
    for margin in AUDIT_PRUNE_MARGINS:
        pruned = pose_candidates(reference, search, k=3, band=False,
                                 prune_margin=margin)
        assert len(exhaustive) == len(pruned) == 3
        for (fe, re_, pe), (fp, rp, pp) in zip(exhaustive, pruned):
            assert fp == pytest.approx(fe, abs=1e-9), (
                "pruning (margin %s) changed the ranked scale order: %s vs %s"
                % (margin, [f for f, _, _ in pruned],
                   [f for f, _, _ in exhaustive]))
            assert rp == pytest.approx(re_, abs=1e-9)
            assert pp == pytest.approx(pe, abs=1e-9)

    # The k=1 path uses the most conservative gate (k-th best = running max)
    # and must also agree.
    one_ex = pose_candidates(reference, search, k=1, band=False,
                             prune_margin=None)
    one_pr = pose_candidates(reference, search, k=1, band=False,
                             prune_margin=0.65)
    assert one_pr[0][0] == pytest.approx(one_ex[0][0], abs=1e-9)


def test_odd_point_pruned_margin_semantics():
    """Unit asserts on the extracted E3 gate (_odd_point_pruned): the gate
    prunes an odd grid point only when BOTH of its evaluated even neighbours
    sit strictly below margin * kth.

    Deep valley -> True (nothing near it can reach the top-k); a shallow
    neighbour -> False (the point stays in the scan); and the boundary is
    STRICT -- a neighbour sitting exactly at margin * kth is NOT pruned,
    because the comparison is `<`, so equality errs on the side of evaluating.
    Pinning the gate directly makes the margin semantics deterministic; the
    fixture's correlation landscape cannot (the shipped 0.5 margin never
    fires on this frame, which the old evaluation-count test measured)."""
    pytest.importorskip("cv2")
    margin, kth = 0.5, 0.9

    # Deep valley: both evaluated even neighbours far below margin * kth.
    assert _odd_point_pruned(0.10, 0.20, kth, margin) is True
    # The real call site's right-edge sentinel: no right neighbour (-inf).
    assert _odd_point_pruned(0.10, -np.inf, kth, margin) is True

    # Shallow: either shoulder at or above the threshold keeps the point.
    assert _odd_point_pruned(0.50, 0.20, kth, margin) is False
    assert _odd_point_pruned(0.10, 0.50, kth, margin) is False

    # Strict boundary: exactly margin * kth is NOT pruned (comparison is <),
    # on either side.
    assert _odd_point_pruned(margin * kth, -np.inf, kth, margin) is False
    assert _odd_point_pruned(-np.inf, margin * kth, kth, margin) is False


def test_pose_candidates_prune_margin_smoke():
    """Smoke on the full path: with the gate wired to the 0.5 margin,
    pose_candidates still returns k=3 hypotheses on this fixture."""
    pytest.importorskip("cv2")
    reference, search = _fixture()
    out = pose_candidates(reference, search, k=3, band=False,
                          prune_margin=0.5)
    assert len(out) == 3
    for f, r, peak in out:
        assert np.isfinite(f) and np.isfinite(r) and np.isfinite(peak)

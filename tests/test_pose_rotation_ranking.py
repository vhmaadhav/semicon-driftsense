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

# Issue #37 acceptance criterion: "Verify endpoint rotations near +/-5 deg and
# off-grid rotations between coarse angle samples."
#
# The coarse rotation grid is np.linspace(-5, 5, 11) -- the eleven integer
# degrees. Two distinct risks live at its edges:
#
#   * ENDPOINTS. At +/-5.0 the golden-section refine has no room on one side
#     (its window is clamped to rotation_bounds), so a basin promoted at an
#     endpoint must already be seeded at the endpoint -- the refine cannot
#     walk it there. 4.9 is the "near +/-5" case the issue names: the true
#     angle sits just inside the bound but its best GRID sample is the bound
#     itself, so the re-rank must score the basin at the clamped sample and
#     still promote it.
#   * OFF-GRID. 2.5 is exactly midway between two samples -- the worst case
#     for the re-rank, because NEITHER neighbouring sample sees the basin at
#     its true angle, so the promotion has to survive the largest possible
#     grid-quantisation penalty. -3.7 is an arbitrary off-grid angle in the
#     other direction, so the coverage is not accidentally symmetric.
#
# Every angle here is checked twice: test_fixture_exhibits_p008_structure_at
# asserts the OLD rot=0-only ranking really does drop the true basin at that
# angle (otherwise the case proves nothing), and
# test_rerank_rescues_rotated_basin_at asserts the new ranking keeps it.
ENDPOINT_THETAS = (5.0, -5.0, 4.9, -4.9)
OFF_GRID_THETAS = (2.5, -3.7)
ROTATION_CASES = ENDPOINT_THETAS + OFF_GRID_THETAS


@lru_cache(maxsize=None)
def _fixture(theta: float = THETA_TRUE):
    """Return (reference, search) for the p008-shaped synthetic frame, with
    the true instance planted at `theta` degrees.

    Cached per angle: the 9000x9000 canvas generation dominates the fixture
    cost and the parameterised cases below reuse it across tests.
    """
    rng = np.random.default_rng(26)
    canvas = generate_dram_canvas(CANVAS_PX, get_preset("dram_1x"), 10.0, rng)

    # p_search = (1/z) R(theta) (p_canvas - c_canvas) + c_search, solved for
    # the canvas centre mapping to the frame centre (the same construction
    # tests/test_scale_semantics.py pins).
    t = np.deg2rad(theta)
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
    module, NOT read from the production E3_PRUNE_MARGIN, so looping here can
    never become a vacuous None-vs-exhaustive check. The `__defaults__` assert
    below pins that exhaustive default. Audit status (2026-09-02): the
    end-to-end equality audit PASSED (bit-identical x/y/scale/theta/score on a
    200-pair seeded draw) but the clock showed no speedup (p50 0.98x, mean
    1.00x), so the default stays exhaustive for keeps-the-semantics reasons,
    not equality ones -- see the AUDITED note at matching.E3_PRUNE_MARGIN."""
    pytest.importorskip("cv2")
    # The production default stays exhaustive: equality held, the clock did
    # not pay. If that default ever changes, re-run both audit legs.
    assert pose_candidates.__defaults__[-1] is None, (
        "pose_candidates no longer defaults to prune_margin=None; the "
        "candidate margins audited below were chosen for the exhaustive-"
        "default regime -- re-run the equality AND clock audits before "
        "shipping any enabled default")
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


def _shortlist_seeds(reference, search, monkeypatch, k=3):
    """The (scale, rotation) seeds pose_candidates actually hands to the
    refine -- i.e. the candidate shortlist itself, read before any polish.

    Issue #37 requires the regression to assert that the true pose basin
    SURVIVES CANDIDATE GENERATION, not merely that a refined x/y happened to
    come out right. Spying on _refine_pose_local is the direct reading: it is
    the single call site the shortlist flows into, so whatever is recorded
    here is exactly what candidate generation offered downstream, with the
    golden-section polish taken out of the picture entirely.
    """
    import driftsense.matching as M

    seeds = []

    def _spy(reference, search, f0, r0, span_s, span_r,
             scale_bounds, rotation_bounds, rounds=2):
        seeds.append((float(f0), float(r0)))
        return float(f0), float(r0), 0.0

    monkeypatch.setattr(M, "_refine_pose_local", _spy)
    M.pose_candidates(reference, search, k=k, band=False, prune_margin=None)
    return seeds


@pytest.mark.parametrize("theta", ROTATION_CASES)
def test_fixture_exhibits_p008_structure_at(theta):
    """Case validity: at every audited angle the OLD rot=0-only ranking must
    still drop the true basin.

    Without this the rescue assertions below would be vacuous -- a case where
    the old code already succeeded proves nothing about the fix.
    """
    pytest.importorskip("cv2")
    reference, search = _fixture(theta)
    old = _old_ranking_topk(reference, search, k=3)
    old_scales = [f for f, _, _ in old]
    assert not any(abs(f - Z_TRUE) <= 0.25 for f in old_scales), (
        "case theta=%.2f is not a regression case: the old rot=0-only "
        "ranking already keeps the true basin (top-3 scales %s)"
        % (theta, old_scales))


@pytest.mark.parametrize("theta", ROTATION_CASES)
def test_rerank_rescues_rotated_basin_at(theta, monkeypatch):
    """Issue #37 acceptance: at endpoint (+/-5) and off-grid rotations the
    true basin must survive CANDIDATE GENERATION under the new ranking.

    The assertion is on the shortlist handed to the refine, not on a refined
    pose: a hypothesis that was never offered cannot be recovered by any
    later stage, which is the whole point of the defect.
    """
    pytest.importorskip("cv2")
    reference, search = _fixture(theta)
    seeds = _shortlist_seeds(reference, search, monkeypatch, k=3)
    assert len(seeds) == 3, "expected k=3 shortlisted hypotheses, got %d" % len(seeds)
    scales = [f for f, _ in seeds]
    assert any(abs(f - Z_TRUE) <= 0.35 for f in scales), (
        "theta=%.2f: candidate generation dropped the true basin z=9.0; "
        "shortlisted scales %s" % (theta, scales))
    # The seed rotation for the true basin must be a coarse GRID sample
    # within one grid step of the planted angle, so the refine (whose window
    # is exactly one step wide, and is clamped at the +/-5 bounds) can reach
    # the truth from it. Not the *nearest* sample: at theta=2.5 the two
    # neighbouring samples are equidistant and the correlation landscape,
    # not the arithmetic, decides which wins -- either is a valid seed.
    grid = [-5.0 + i for i in range(11)]
    step = 1.0
    r_seed = next(r for f, r in seeds if abs(f - Z_TRUE) <= 0.35)
    assert any(r_seed == pytest.approx(g, abs=1e-9) for g in grid), (
        "theta=%.2f: true basin seeded at rotation %.3f, which is not a "
        "coarse grid sample" % (theta, r_seed))
    assert abs(r_seed - theta) <= step + 1e-9, (
        "theta=%.2f: true basin seeded at rotation %.2f, more than one grid "
        "step (%.1f deg) from the planted angle -- the refine window cannot "
        "reach the truth from there" % (theta, r_seed, step))


@pytest.mark.parametrize("theta", ROTATION_CASES)
def test_rerank_recovers_pose_at(theta):
    """End-to-end on the same cases: the promoted basin refines to the
    planted scale and angle, including at the clamped +/-5 endpoints (where
    the golden-section window has no room on one side) and at the off-grid
    angles (where no coarse sample sits on the truth)."""
    pytest.importorskip("cv2")
    reference, search = _fixture(theta)
    out = pose_candidates(reference, search, k=3, band=False, prune_margin=None)
    f_best, r_best, _ = out[0]
    assert abs(f_best - Z_TRUE) <= 0.35, (
        "theta=%.2f: winner scale %.3f is not the true basin" % (theta, f_best))
    assert abs(r_best - theta) <= 0.4, (
        "theta=%.2f: winner rotation %.3f missed the planted angle"
        % (theta, r_best))


def test_shortlist_survival_is_what_changed(monkeypatch):
    """The nominal (3.0 deg) case, read at the same shortlist level: the true
    basin is absent from the OLD rot=0-only top-k and present in the new one.

    This is the paired statement issue #37 asks for -- same frame, same k,
    the only difference being whether the scale shortlist was ranked with
    rotation evidence.
    """
    pytest.importorskip("cv2")
    reference, search = _fixture()
    old_scales = [f for f, _, _ in _old_ranking_topk(reference, search, k=3)]
    assert not any(abs(f - Z_TRUE) <= 0.35 for f in old_scales)
    new_scales = [f for f, _ in _shortlist_seeds(reference, search, monkeypatch, k=3)]
    assert any(abs(f - Z_TRUE) <= 0.35 for f in new_scales), (
        "rotation-aware shortlist %s still drops the true basin" % new_scales)

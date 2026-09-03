"""Decoding a response map into a single (x, y), and refining it.

Two stages, deliberately split:

* the network decides *which* of the many identical-looking candidates is the
  right one -- that is the hard, learned part;
* a classical ZNCC snap at full resolution decides *exactly where* -- that is
  the easy, precise part, and a 4 px-stride heatmap cannot do it alone.

Handing the sub-pixel job to correlation rather than to the offset head is
what keeps the final error well under one pixel.

Sizes are derived from the inputs rather than assumed, so a reference or
search frame that is not exactly 1000 px still localises correctly.
"""

from __future__ import annotations

import cv2
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from driftsense.model import SCALE, STRIDE, TEMPLATE_SIZE
from driftsense.config import SHIPPED_CONFIDENCE
from driftsense.verification import (
    common_band,
    dog_feature,
    local_match_score,
    rank_transform,
)

# Two candidates whose scores differ by less than this are treated as tied,
# and the problem statement's rule applies: prefer the one nearer the centre
# of the search image.
TIE_REL_TOL = 0.04

# Half-width of the ZNCC snap window, in search pixels. Tuned on validation
# (scripts/ablate.py): a wider window improves nothing and costs accuracy,
# because at +/-8px or more the snap can reach an adjacent repeat and drag a
# correct coarse prediction one period off. +/-4px is enough to absorb the
# 4px response-grid stride while staying inside the correct cell.
REFINE_RADIUS = 4

# TTA cluster arbitration: candidate regions are scored as
#   ZNCC(region) + VERIFY_ALPHA * (normalised network confidence)
# Tuned on validation (scripts/sweep_aggregation.py). The result is flat across
# alpha 0.35-0.5 and any top_k >= 3, and worth exactly +1 sample in 300 on both
# validation and test -- real but small. Set VERIFY_ALPHA = 0 to disable.
VERIFY_ALPHA = 0.5
VERIFY_TOP_K = 4


def make_template(reference: np.ndarray, factor: float = SCALE,
                  rotation_deg: float = 0.0,
                  canvas: tuple[int, int] | None = None) -> np.ndarray:
    """Reference -> template at the search frame's pixel size and pose.

    The realised scale must be *continuous* in `factor`. Resizing straight to
    `round(H/factor)` does not give that: the template can only ever be an
    integer number of pixels, so the scale it actually realises is
    `W / round(W / factor)` -- a 43-step staircase across [8, 12] whose steps
    are 0.81-1.22% wide. The Phase 2 scale tier pays full credit below 1%, so
    the quantisation step is as wide as the whole full-credit band, and any
    search over `factor` is optimising a piecewise-constant function. That is
    the real reason correlation-vs-scale looked "nearly flat": it *was* flat,
    in 43 plateaus.

    So the reduction is done in two stages. INTER_AREA takes it to the nearest
    enclosing integer size -- the same area-average the physical search image
    underwent, and bilinear here measurably softens the match. A single affine
    then applies the residual sub-pixel scale *and* any rotation together, so
    continuity costs no extra resampling pass over the previous code, which
    already paid for a warp whenever rotation was non-zero.

    The canvas is floor()ed rather than round()ed so that every canvas pixel
    holds real reference content. Rounding up would leave a replicated border
    whose width jumps as the canvas crosses an integer, reintroducing a
    discontinuity in the correlation value at exactly the scales we are trying
    to resolve. At `factor` values that divide the reference exactly (10.0 for
    a 1000 px reference) this returns bit-identical output to the old path.

    `canvas` pins the output size (height, width), centre-cropping the content
    instead of letting the canvas follow `factor`. Comparing TM_CCOEFF_NORMED
    across templates of *different* sizes is biased -- fewer pixels correlate
    better by chance, which pushes any scale search towards larger `factor` --
    so a search that sweeps scale must hold the canvas fixed to be fair.
    """
    h, w = reference.shape[:2]
    fh, fw = h / float(factor), w / float(factor)          # exact footprint
    th, tw = max(int(np.floor(fh)), 1), max(int(np.floor(fw)), 1)
    if canvas is not None:
        th, tw = max(int(canvas[0]), 1), max(int(canvas[1]), 1)
    ah, aw = max(int(np.ceil(fh)), 1), max(int(np.ceil(fw)), 1)

    base = cv2.resize(reference, (aw, ah), interpolation=cv2.INTER_AREA)
    resid = fw / aw                                        # (0.5, 1.0]
    if rotation_deg == 0.0 and abs(resid - 1.0) < 1e-9 and (aw, ah) == (tw, th):
        return base

    # Rotate/scale about the reference centre, then move that centre to the
    # canvas centre. Both are expressed in pixel-centre coordinates, which
    # keeps the "centre = top-left + tw/2" convention the rest of the file
    # relies on.
    M = cv2.getRotationMatrix2D(((aw - 1) / 2.0, (ah - 1) / 2.0), rotation_deg, resid)
    M[0, 2] += (tw - 1) / 2.0 - (aw - 1) / 2.0
    M[1, 2] += (th - 1) / 2.0 - (ah - 1) / 2.0
    return cv2.warpAffine(base, M, (tw, th), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def template_hypotheses(reference: np.ndarray) -> list[float]:
    """Plausible reference-to-search downsample factors, best guess first.

    The reference covers a fixed 1 um field of view and the search frame is
    10 nm/px, so the pattern's footprint there is ~100 px *regardless of what
    pixel dimensions the reference itself arrives at*. A reference delivered at
    1 nm/px (1000 px) needs dividing by 10; one already delivered at the search
    resolution (100 px) needs dividing by 1. Dividing unconditionally by 10
    turns the latter into a 10x10 template and the match becomes noise.

    We cannot read nm/px out of a PNG, so both readings are offered and the
    caller picks by correlation score. For the usual 1000 px reference the two
    hypotheses coincide and this costs nothing.
    """
    h, w = reference.shape[:2]
    longest = max(h, w)
    factors = [SCALE, longest / float(TEMPLATE_SIZE)]

    out: list[float] = []
    for f in factors:
        if f < 1.0:                       # reference smaller than its footprint
            f = 1.0
        if not any(abs(f - g) < 1e-6 for g in out):
            out.append(f)
    return out


def choose_factor(reference: np.ndarray, search: np.ndarray) -> float:
    """Pick the reference->search downsample factor by correlation evidence.

    Costs one full-frame ZNCC per hypothesis (milliseconds) and is only ever
    ambiguous when the reference does not arrive at the expected 1000 px, so
    the normal path pays nothing. ZNCC is a poor *localiser* on periodic
    layouts -- that is what the network is for -- but telling a correctly
    scaled template from one scaled 10x wrong is exactly the coarse judgement
    it is reliable at.
    """
    hypotheses = template_hypotheses(reference)
    if len(hypotheses) == 1:
        return hypotheses[0]

    best, best_score = hypotheses[0], -np.inf
    for f in hypotheses:
        tmpl = make_template(reference, f)
        if tmpl.shape[0] >= search.shape[0] or tmpl.shape[1] >= search.shape[1]:
            continue
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        score = float(cv2.minMaxLoc(res)[1])
        if score > best_score:
            best, best_score = f, score
    return best


# Departing from the nominal pose must be *earned*. The network was trained at
# exactly 10x and 0 degrees, so feeding it an off-nominal template is a real
# cost, and a periodic layout will always hand some wrong pose a lucky
# correlation peak. A searched pose is adopted only if it beats nominal by this
# margin, which keeps the whole thing a no-op on nominal data.
POSE_MARGIN = 0.05
POSE_SCALES = (0.90, 0.95, 1.05, 1.10)        # the 9-11x range, relative
POSE_ROTATIONS = (-2.0, -1.0, 1.0, 2.0)       # degrees


def _peak_score(search: np.ndarray, tmpl: np.ndarray) -> float:
    if tmpl.shape[0] >= search.shape[0] or tmpl.shape[1] >= search.shape[1]:
        return -np.inf
    return float(cv2.minMaxLoc(cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED))[1])


# The pose search is a coarse decision -- which of nine candidates, not where
# -- so it runs on reduced-resolution copies. Halving is the useful limit: a
# quarter cannot separate a 5% scale difference, because 5% of a 25 px probe
# template is sub-pixel. Combined with the skip below, nominal scenes cost a
# few milliseconds and only genuinely off-nominal ones pay for the search.
POSE_PROBE_DOWNSCALE = 2
# Above this correlation at the nominal pose there is nothing to look for, and
# the search is skipped outright. Nominal scenes are the overwhelming majority.
POSE_SKIP_ABOVE = 0.70


# Band-pass the coarse probe before correlating. Measured on Set B: of the
# >5 px failures, only 24% had a correct pose hypothesis generated at all --
# the other 76% never had a right answer to select, so the *search* was
# failing, not the ranking. The coarse score is a half-resolution correlation
# on raw pixels, and Set B's dominant degradations sit at both ends of the
# spectrum: charging streaks are low-frequency, shot and impulse noise are
# high-frequency, and the layout structure is in between. A difference of
# Gaussians keeps the band that carries the pattern and discards both.
#
# The same filter beat raw ZNCC at *verifying* hypotheses in the same
# experiment (net +10 pairs against +7; scripts/verify_scores.py), which is
# what suggested trying it one stage earlier. A rank transform recovered as
# many failures but broke three times as many successes, so it is not used.
def _band(img: np.ndarray, s1: float = 1.0, s2: float = 4.0) -> np.ndarray:
    f = img.astype(np.float32)
    return cv2.GaussianBlur(f, (0, 0), s1) - cv2.GaussianBlur(f, (0, 0), s2)


def _probe(img: np.ndarray, k: int = POSE_PROBE_DOWNSCALE) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(w // k, 1), max(h // k, 1)),
                      interpolation=cv2.INTER_AREA)


def canonicalize_search(search: np.ndarray, magnification: float, rotation_deg: float,
                       target: float = float(SCALE)) -> tuple[np.ndarray, np.ndarray]:
    """Resample a search frame to the nominal pose. Returns (frame, affine).

    The network was trained on matched-scale, unrotated pairs and that is where
    its sub-pixel behaviour lives, so rather than teach it to tolerate pose we
    undo the pose first and hand it the input it already understands. The
    returned 2x3 affine maps native search coordinates to canonical ones;
    invert it to bring a canonical answer back.

    Only the *coarse* decision runs here. Sub-pixel refinement deliberately
    goes back to the native frame, so the answer never inherits this
    resampling's interpolation blur.
    """
    k = magnification / target
    h, w = search.shape[:2]
    n_h, n_w = int(round(h * k)), int(round(w * k))
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), -rotation_deg, k)
    M[0, 2] += (n_w - 1) / 2.0 - (w - 1) / 2.0
    M[1, 2] += (n_h - 1) / 2.0 - (h - 1) / 2.0
    out = cv2.warpAffine(search, M, (n_w, n_h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE)
    return out, M


def uncanonicalize_point(M: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Map a canonical-frame point back to native search coordinates."""
    inv = cv2.invertAffineTransform(M)
    return (float(inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]),
            float(inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]))


# Phase 2 discloses the pose bounds outright and the rules explicitly permit
# hard-coding them, so the wide search below is bounded by fact rather than by
# guesswork. Phase 1's narrow POSE_SCALES/POSE_ROTATIONS stay as they are: the
# shipped weights and every Phase 1 number were produced with them.
PHASE2_SCALE_BOUNDS = (8.0, 12.0)
PHASE2_ROTATION_BOUNDS = (-5.0, 5.0)

# Sub-pixel drift recovery (see `drift_row_refine`). The lag covers 3 sigma of
# the severity-4 drift jitter (sd up to ~2.1 px); a narrower window clips the
# peak exactly where the points are. Both other values are measured optima on
# the full 1,750 present pairs -- see the A/B table in `.agents/SUBPIXEL_DRIFT.md`.
DRIFT_ROW_LAG = 12
DRIFT_ROW_MIN_CORR = 0.30
DRIFT_MAX_SHIFT = 5.0      # upper bound; the effective clamp is drift-scaled
DRIFT_CLAMP_K = 2.0        # clamp = clip(K * measured drift sd, 2.0, DRIFT_MAX_SHIFT)

# Samples in the coarse scale sweep. This is a *sampling* parameter, not a
# ranking one, and it was undersampling its own objective: 17 points across
# [8, 12] is a 2.5% step at m=10, while the correlation-vs-scale peak is only
# ~1-2% wide -- which is why the full-credit pose tier is 1% wide in the first
# place. The sweep could therefore step straight over the true peak, so the
# true basin never became a local maximum at all.
#
# That is also why raising the hypothesis count did nothing when it was tried:
# extra candidate slots cannot hold a peak that was never sampled. A true-pose
# oracle confirmed the diagnosis -- 52% of the >5 px failures were pose-search
# failures, with median scale error 2.99% among failures against 0.40% among
# successes.
#
# The obvious fix does not pay, and the measurement is kept here so nobody
# re-runs it. 41 points (a 1.0% step) moved set B credit 0.7566 -> 0.7669 but
# gave it all back on set A (0.9509 -> 0.9440) and on rotation, for a net
# **-0.006 points**. Decoupling the refinement window from the sample count so
# the finer grid kept the wider refine reach was worse still (73.76 against
# 74.32). So the sampling really is not the binding constraint: the coarse
# score itself is noise-dominated on degraded frames, and ranking, not
# resolution, is what fails. See .agents/PHASE2_STATE.md for the next lever
# (a rank-transform coarse score, which is robust to the impulse noise that
# dominates these failures).
COARSE_SCALES = 17

# E3 pruning gate (inference-efficiency plan, task 2 / issue #7): skip a grid
# point's make_template+_peak_score when its already-evaluated left neighbour
# sits below this fraction of the running k-th best valley value. A margin of
# 0.5 keeps the gate conservative (only deep two-sample valleys are skipped).
# Set to None (or 0.0) to restore the exhaustive scan.
#
# AUDITED (2026-09-02, 200-pair seeded-draw audit -- NOT the full-2,250
# equality audit this default was pinned on; that stays pending and is
# required before any enabled default): on a
# seeded 200-pair draw of the full shards, margin 0.5 produces bit-identical
# output -- x, y, scale, theta and score exactly 0.0e+00 delta on 200/200
# pairs; only n_hyp instrumentation differs (15/200 pairs report fewer offered
# grid points, as expected). But the single-process clock (20 pairs x 3 reps,
# interleaved configs, 4 torch threads) shows NO speedup: p50 0.98x, mean
# 1.00x -- the skipped coarse evaluations are noise against the network
# forward + refine + polish. A change that alters instrumentation semantics
# for a 1.00x clock is not shipped: the default stays the exhaustive scan.
E3_PRUNE_MARGIN = None

# How many rot=0 peaks enter the rotation re-rank, as a multiple of k. The
# plan allows k or 2k: 2k covers the case where the true basin ranks between
# k+1 and 2k at rot=0, which the k-only window cannot rescue. The extra
# rotation scans are bounded (2k x coarse_rotations) and cached for the
# refine, and E3 pruning offsets them.
RERANK_MULTIPLIER = 2

# Rotation-aware re-ranking of the scale shortlist (issue #37). OFF by default.
#
# The mechanism is sound -- ranking every scale at rot=0 can discard a true
# basin that only separates at its own rotation -- but enabling it changes the
# *shipped decoder*, and the full-2,250 A/B for it has not been run. Turning it
# on without that evidence would swap the final decode on argument alone right
# before submission. With this False, `pose_candidates` reproduces the previous
# `peaks[:k]` ranking exactly: the shortlist is sliced straight to k and no
# rotation scan is cached, so the refine loop takes its original direct-scan
# path (pinned by tests/test_pose_rotation_ranking.py).
#
# To enable: set this True, run the full 2,250-pair A/B, and record the paired
# delta per component before changing the default.
RERANK_ROTATION = False


def _golden_max(f, lo: float, hi: float, iters: int = 8) -> tuple[float, float]:
    """Maximise a unimodal f on [lo, hi]. Returns (argmax, max).

    Correlation-vs-pose is single-peaked near the true pose, which is what
    makes golden section legitimate here -- and it reaches the 1% scale /
    0.25 deg rotation tolerances in ~8 evaluations, where an equally fine grid
    would need ~40.
    """
    invphi = (5.0 ** 0.5 - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = f(d)
    return ((c, fc) if fc > fd else (d, fd))


def _refine_pose_local(reference, search, f0: float, r0: float,
                       span_s: float, span_r: float,
                       scale_bounds, rotation_bounds, rounds: int = 1, iters: int = 4):
    """Fast polish of one (scale, rotation) hypothesis for candidate generation,
    on a crop around its peak. Deep polish is done downstream by polish_pose."""
    lo_s, hi_s = scale_bounds
    lo_r, hi_r = rotation_bounds
    tpl = make_template(reference, f0, r0)
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    pad = int(max(th, tw) * 1.5)
    y0, x0 = max(int(loc[1]) - pad, 0), max(int(loc[0]) - pad, 0)
    crop = search[y0:int(loc[1]) + th + pad, x0:int(loc[0]) + tw + pad]

    def fine(f, r):
        t = make_template(reference, f, r)
        if t.shape[0] >= crop.shape[0] or t.shape[1] >= crop.shape[1]:
            return -np.inf
        return float(cv2.minMaxLoc(cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED))[1])

    f, r, peak = f0, r0, fine(f0, r0)
    for _ in range(rounds):
        f, peak = _golden_max(lambda v: fine(v, r), max(f - span_s, lo_s), min(f + span_s, hi_s), iters=iters)
        r, peak = _golden_max(lambda v: fine(f, v), max(r - span_r, lo_r), min(r + span_r, hi_r), iters=iters)
        span_s, span_r = span_s / 3.0, span_r / 3.0
    return float(f), float(r), float(peak)


def winner_margin(candidates: list, winner: dict) -> float:
    """The SELECTED winner's min(score, zncc) margin over the best runner-up.

    A present/absent uncertainty feature (issue #6): an absent pair can only
    produce low, closely-spaced responses, so a small margin is evidence the
    'match' is not a real instance. `winner` must be the candidate the decode
    actually returned (choose() selects by max zncc on the shipped path, which
    need not be the min-metric leader -- the margin can legitimately be
    negative, and that disagreement is itself signal). A missing score is
    skipped (the remaining values are the evidence); a winner or runner-up
    with no finite evidence, or fewer than two candidates, yields NaN rather
    than a fake or infinite margin.
    """
    def strength(c: dict) -> float:
        vals = [float(c[k]) for k in ("score", "zncc")
                if c.get(k) is not None and np.isfinite(c[k])]
        return min(vals) if vals else -np.inf

    if winner is None or not candidates or len(candidates) < 2:
        return float("nan")
    w = strength(winner)
    others = [strength(c) for c in candidates if c is not winner]
    best_other = max(others) if others else -np.inf
    if w == -np.inf or best_other == -np.inf:
        return float("nan")
    return float(w - best_other)


def _odd_point_pruned(prev: float, nxt: float, kth: float,
                      margin: float) -> bool:
    """E3 gate for one odd grid point (see pose_candidates). True = skip it:
    both evaluated even neighbours sit below margin * kth, so a hill here
    could not reach the top-k without lifting its own shoulders first."""
    return max(prev, nxt) < margin * kth


def pose_candidates(reference: np.ndarray, search: np.ndarray, k: int = 3,
                    scale_bounds: tuple[float, float] = PHASE2_SCALE_BOUNDS,
                    rotation_bounds: tuple[float, float] = PHASE2_ROTATION_BOUNDS,
                    coarse_scales: int = COARSE_SCALES, coarse_rotations: int = 11,
                    refine_span_scales: int = 17, band: bool = True,
                    prune_margin: float | None = E3_PRUNE_MARGIN,
                    rerank_rotation: bool = RERANK_ROTATION) -> list:
    """Up to `k` distinct (scale, rotation, peak) hypotheses, best first.

    Correlation against a periodic layout is multi-peaked in *scale*: a wrong
    magnification can align the template with the wrong repeat and score well.
    Measured on validation, every localisation failure was one of these -- the
    failures sat 15.8% off in scale while the successes sat at 0.89%, so they
    were not near-misses but confident lock-ons to the wrong basin.

    Committing to the single best coarse peak therefore throws the answer away
    whenever the true basin ranks second. Returning the top few local maxima
    instead lets the caller settle it with a full-resolution ZNCC check, which
    separates the basins cleanly where the low-resolution probe cannot.

    Ranking is rotation-aware (inference-efficiency plan, task 2): the cheap
    first pass ranks scale at rot=0 only, so a rotated pair can promote a
    wrong-scale basin whose rot=0 correlation happens to beat the true basin's
    (the competitor's p008 failure mode -- a rotated pair ranked a wrong scale
    first because at rot=0 the template correlated poorly at *every* scale).
    The fix re-ranks the top surviving peaks by their own best-rotation score
    -- the same per-peak scan the refinement stage already paid for -- before
    choosing which k advance to the golden-section refine. The full joint
    17x11 scale-rotation grid is deliberately NOT built: the coarse sweep is
    66.8% of pair time (issue #7) and a joint grid multiplies that dominant
    cost by the rotation count, buying coverage the per-peak rotation scan
    and the refine already supply inside each surviving basin.
    """
    lo_s, hi_s = scale_bounds
    lo_r, hi_r = rotation_bounds
    probe_search = _probe(search)
    if band:
        probe_search = _band(probe_search)

    def coarse(f, r=0.0):
        t = _probe(make_template(reference, f, r))
        return _peak_score(probe_search, _band(t) if band else t)

    # One rotation grid, shared by the re-rank and the per-peak scan.
    rots = np.linspace(lo_r, hi_r, coarse_rotations)

    grid = np.linspace(lo_s, hi_s, coarse_scales)
    # E3 grid pruning (issue #7; audited on a seeded 200-pair draw -- the
    # full-2,250 equality audit stays pending and is required before any
    # enabled default). Two
    # passes: even grid points are always evaluated, then odd points are
    # evaluated only when at least one of their two (now evaluated) even
    # neighbours is within prune_margin of the running k-th best value -- a
    # point buried between two deep two-sample valleys is not worth a
    # template evaluation, because a hill that could ever enter the top-k
    # forces the k-th best value to at least its own neighbour samples. The
    # gate is a documented HEURISTIC, not a certified bound: correlation-
    # vs-scale is multi-peaked, a skipped point that would have been a
    # hill's first sample leaves that hill represented by its shoulder, and
    # a skipped point can itself change the k-th best reference for later
    # points. Enabling the pruning (setting E3_PRUNE_MARGIN) requires the
    # full 2,250-pair equality audit (identical found/x/y, or paired delta
    # CI within [-0.1, +0.1]) -- it has not been run; the 200-pair draw is
    # the only equality evidence so far. `prune_margin=None` (or 0.0)
    # restores the exhaustive scan.
    vals: list[float] = []
    if prune_margin:
        vals = [0.0] * len(grid)
        for i in range(0, len(grid), 2):
            vals[i] = coarse(float(grid[i]))
        evaluated = sorted(vals[i] for i in range(0, len(grid), 2))
        kth = evaluated[-min(int(k), len(evaluated))]
        for i in range(1, len(grid), 2):
            nxt = vals[i + 1] if i + 1 < len(grid) else -np.inf
            if _odd_point_pruned(vals[i - 1], nxt, kth, prune_margin):
                vals[i] = -np.inf
            else:
                vals[i] = coarse(float(grid[i]))
    else:
        vals = [coarse(float(f)) for f in grid]

    # Interior local maxima, so two samples on one hill do not both survive.
    # Points pruned above were never evaluated (-inf) and are not maxima.
    peaks = [i for i in range(len(grid))
             if vals[i] > -np.inf
             and (i == 0 or vals[i] >= vals[i - 1])
             and (i == len(grid) - 1 or vals[i] >= vals[i + 1])]
    peaks.sort(key=lambda i: -vals[i])

    # Re-rank the top surviving peaks by their best-rotation score BEFORE
    # choosing which k advance to the refine (the plan's "(k or 2k)" window).
    # At rot=0 a rotated pair's true basin can rank below k while a
    # wrong-scale basin wins -- the p008 failure mode -- but at the true
    # rotation the true basin's peak separates, so the window is widened to
    # RERANK_MULTIPLIER * k before the cut. The scan is the same per-peak
    # rotation sweep the refinement stage already paid for, so its (value,
    # argmax) is cached here and the refine loop below reuses it: the marginal
    # cost is only the scans for peaks that fail to make the cut, and the E3
    # pruning above repays that.
    # With `rerank_rotation` False this is `peaks[:k]` after the slice below,
    # and `rot_best` stays empty -- i.e. byte-for-byte the pre-#37 behaviour.
    ranked = peaks[:(RERANK_MULTIPLIER if rerank_rotation else 1) * max(int(k), 1)]
    rot_best: dict[int, tuple[float, float]] = {}
    if rerank_rotation and len(ranked) > 1:
        for i in ranked:
            scores = [float(coarse(float(grid[i]), r)) for r in rots]
            j = int(np.argmax(scores))   # first max -- same tie rule as max()
            rot_best[i] = (scores[j], float(rots[j]))
        ranked.sort(key=lambda i: (-rot_best[i][0], -vals[i]))
    ranked = ranked[:max(int(k), 1)]

    # The refinement window is deliberately NOT derived from the sample count.
    # Sampling density and refinement reach are independent concerns: a finer
    # grid resolves *which* basin is real, but the local refine still needs a
    # window wide enough to walk to the true optimum from a grid point that may
    # sit up to a full step away. Tying them together meant that going from 17
    # to 41 samples silently shrank the refine window from 0.25 to 0.10 and
    # gave back on set A and on pose what it won on set B (measured: a net
    # -0.006 points, i.e. exactly nothing).
    span_s = (hi_s - lo_s) / (refine_span_scales - 1)
    span_r = (hi_r - lo_r) / (coarse_rotations - 1)
    out = []
    for i in ranked:
        f0 = float(grid[i])
        # The rotation scan was already paid for in the re-rank above; the
        # single-peak path falls back to a direct scan.
        r0 = (rot_best[i][1] if i in rot_best
              else float(max(rots, key=lambda r: coarse(f0, r))))
        out.append(_refine_pose_local(reference, search, f0, r0, span_s, span_r,
                                      scale_bounds, rotation_bounds))

    # Basin deduplication: if two candidates fall in the same scale/rotation basin,
    # downstream polish_pose already searches across +/-3% scale and +/-0.8 deg,
    # so evaluating duplicate hypotheses in the neural network is redundant.
    deduped = []
    for c in out:
        if not any(abs(c[0] - d[0]) < 0.35 and abs(c[1] - d[1]) < 1.0 for d in deduped):
            deduped.append(c)
    return deduped or out or [(float(np.mean(scale_bounds)), 0.0, -np.inf)]


def choose_pose_wide(reference: np.ndarray, search: np.ndarray,
                     scale_bounds: tuple[float, float] = PHASE2_SCALE_BOUNDS,
                     rotation_bounds: tuple[float, float] = PHASE2_ROTATION_BOUNDS,
                     coarse_scales: int = 11, coarse_rotations: int = 11,
                     rounds: int = 2) -> tuple[float, float, float]:
    """Recover (factor, rotation_deg, peak_correlation) over the Phase 2 bounds.

    Three stages, cheapest first, because the CPU budget is 5 s median:

    1. A coarse grid over the full disclosed box, on half-resolution probes.
       Only the ranking matters here, so resolution is wasted effort.
    2. Localise the coarse winner's peak and crop the search frame around it.
       Refinement only ever needs the neighbourhood of the match, and a crop
       turns every later correlation from a 1000x1000 problem into a small one.
    3. Alternating golden-section refinement of scale and rotation on that
       crop, at full resolution -- which is where the 1% / 0.25 deg pose
       tolerances are actually won.
    """
    lo_s, hi_s = scale_bounds
    lo_r, hi_r = rotation_bounds
    probe_search = _probe(search)

    def coarse(factor, rot=0.0):
        return _peak_score(probe_search, _probe(make_template(reference, factor, rot)))

    best_f = max((f for f in np.linspace(lo_s, hi_s, coarse_scales)), key=lambda f: coarse(f))
    best_r = max((r for r in np.linspace(lo_r, hi_r, coarse_rotations)),
                 key=lambda r: coarse(best_f, r))

    # Crop around the coarse peak so refinement is cheap.
    tpl = make_template(reference, best_f, best_r)
    res = cv2.matchTemplate(search, tpl, cv2.TM_CCOEFF_NORMED)
    _, _, _, loc = cv2.minMaxLoc(res)
    th, tw = tpl.shape[:2]
    pad = int(max(th, tw) * 1.5)
    y0 = max(int(loc[1]) - pad, 0)
    x0 = max(int(loc[0]) - pad, 0)
    crop = search[y0:int(loc[1]) + th + pad, x0:int(loc[0]) + tw + pad]

    def fine(factor, rot):
        return _peak_score(crop, make_template(reference, factor, rot))

    span_s = (hi_s - lo_s) / (coarse_scales - 1)
    span_r = (hi_r - lo_r) / (coarse_rotations - 1)
    peak = fine(best_f, best_r)
    for _ in range(rounds):
        f_lo, f_hi = max(best_f - span_s, lo_s), min(best_f + span_s, hi_s)
        best_f, peak = _golden_max(lambda f: fine(f, best_r), f_lo, f_hi)
        r_lo, r_hi = max(best_r - span_r, lo_r), min(best_r + span_r, hi_r)
        best_r, peak = _golden_max(lambda r: fine(best_f, r), r_lo, r_hi)
        span_s, span_r = span_s / 3.0, span_r / 3.0
    return float(best_f), float(best_r), float(peak)


def choose_pose(reference: np.ndarray, search: np.ndarray,
                margin: float = POSE_MARGIN) -> tuple[float, float]:
    """Estimate (downsample factor, rotation degrees) by correlation evidence.

    Coordinate descent rather than a full grid: scale first at zero rotation,
    then rotation at the winning scale -- nine correlations instead of
    twenty-five, and the two axes are near enough to independent here that the
    joint optimum is not missed in practice. All of it on quarter-resolution
    probes, since only the ranking matters.

    Returns the nominal pose unless an off-nominal one wins by `margin`.
    """
    base = choose_factor(reference, search)
    probe_search = _probe(search)

    def score(factor, rot=0.0):
        tpl = make_template(reference, factor, rot)
        return _peak_score(probe_search, _probe(tpl))

    nominal = score(base)
    if nominal >= POSE_SKIP_ABOVE:
        return base, 0.0

    best_f, best_s = base, nominal
    for m in POSE_SCALES:
        sc = score(base * m)
        if sc > best_s:
            best_f, best_s = base * m, sc

    best_r = 0.0
    for r in POSE_ROTATIONS:
        sc = score(best_f, r)
        if sc > best_s:
            best_r, best_s = r, sc

    if best_s < nominal + margin:
        return base, 0.0
    return best_f, best_r


def standardize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.mean()) / max(float(x.std()), 1e-5)


def pad_to_stride(img: np.ndarray, stride: int = STRIDE) -> np.ndarray:
    """Pad bottom/right to a multiple of the encoder stride. Padding only on
    those edges leaves every pixel's coordinate unchanged."""
    h, w = img.shape
    ph, pw = (-h) % stride, (-w) % stride
    if ph or pw:
        img = np.pad(img, ((0, ph), (0, pw)), mode="edge")
    return img


def response_to_center(i: float, j: float, th: int, tw: int,
                       dy: float = 0.0, dx: float = 0.0) -> tuple[float, float]:
    """Response cell (+ sub-cell offset) -> search-image centre (x, y)."""
    return (j + dx) * STRIDE + tw / 2.0, (i + dy) * STRIDE + th / 2.0


def peak_stats(prob: np.ndarray, i: int, j: int, exclude: int = 5) -> tuple[float, float]:
    """Peak-to-sidelobe ratio and APCE for a response map.

    Both measure how far the winner stands out from the *background
    distribution* of the surface, which is different information from the
    winner's height (`score`) or from its margin over the runner-up
    (`peak_ratio`). A confident lock-on to the wrong repeat produces a tall
    peak on a busy surface; a true match produces a tall peak on a quiet one.
    Height alone cannot tell those apart, and on a periodic layout that is
    exactly the confusion we are trying to resolve.

    PSR  = (peak - mean(sidelobe)) / std(sidelobe), sidelobe excluding an
           (2*exclude+1)^2 window around the peak. Bolme, D. S., Beveridge,
           J. R., Draper, B. A. and Lui, Y. M., "Visual Object Tracking using
           Adaptive Correlation Filters", CVPR 2010 -- where it is used to
           detect occlusion and tracking failure, structurally the same
           present/absent question we face.
    APCE = (peak - min)^2 / mean((surface - min)^2), the average
           peak-to-correlation energy, reported in the correlation-filter
           literature as the more stable of the two.

    Free: the response map is already computed by `locate`.
    """
    p = np.asarray(prob, dtype=np.float64)
    h, w = p.shape
    peak = float(p[i, j])
    mask = np.ones_like(p, dtype=bool)
    mask[max(i - exclude, 0):i + exclude + 1, max(j - exclude, 0):j + exclude + 1] = False
    side = p[mask]
    if side.size < 16:
        return 0.0, 0.0
    sd = float(side.std())
    psr = float((peak - side.mean()) / sd) if sd > 1e-9 else 0.0
    mn = float(p.min())
    denom = float(((p - mn) ** 2).mean())
    apce = float((peak - mn) ** 2 / denom) if denom > 1e-12 else 0.0
    return psr, apce


def find_peaks(prob: np.ndarray, max_peaks: int = 32) -> list[tuple[int, int, float]]:
    """Local maxima of the response map, strongest first."""
    t = torch.from_numpy(np.ascontiguousarray(prob))[None, None]
    pooled = F.max_pool2d(t, kernel_size=5, stride=1, padding=2)[0, 0].numpy()
    ii, jj = np.nonzero(prob >= pooled - 1e-9)
    vals = prob[ii, jj]
    order = np.argsort(-vals)[:max_peaks]
    return [(int(ii[o]), int(jj[o]), float(vals[o])) for o in order]


def select_peak(prob: np.ndarray, search_hw: tuple[int, int],
                template_hw: tuple[int, int],
                tie_tol: float = TIE_REL_TOL) -> tuple[int, int, float]:
    """Pick the reported match.

    Strongest peak wins, except that near-ties are resolved toward the centre
    of the search image, as the problem statement requires.
    """
    peaks = find_peaks(prob)
    if not peaks:
        i, j = np.unravel_index(int(np.argmax(prob)), prob.shape)
        return int(i), int(j), float(prob[i, j])

    best = peaks[0][2]
    tied = [p for p in peaks if p[2] >= best * (1.0 - tie_tol)]
    if len(tied) == 1:
        return tied[0]

    h, w = search_hw
    th, tw = template_hw
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    def dist(p):
        x, y = response_to_center(p[0], p[1], th, tw)
        return float(np.hypot(x - cx, y - cy))

    return min(tied, key=dist)


def parabolic(a: float, b: float, c: float) -> float:
    """Sub-sample peak offset in [-1, 1] from three samples around a max."""
    denom = a - 2.0 * b + c
    if abs(denom) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (a - c) / denom, -1.0, 1.0))


def refine_zncc(search: np.ndarray, template: np.ndarray,
                cx: float, cy: float, radius: int = REFINE_RADIUS) -> tuple[float, float, float]:
    """Snap a coarse centre to the local ZNCC optimum at full resolution.

    Searches +/- `radius` px around the coarse box position. The placement
    rule is the shipped config (driftsense.config.SHIPPED_SUBPIXEL, ONE
    definition): "bicubic" upsamples the correlation surface around the peak
    (driftsense.subpixel.refine_bicubic; rescues the 1px-tier boundary pairs,
    Debella-Gilo & Kaab 2011), "parabola" is the historical 1-D parabolic
    fit through the peak. Both share this function's contract and window.
    """
    from driftsense.config import SHIPPED_SUBPIXEL
    if SHIPPED_SUBPIXEL == "bicubic":
        from driftsense.subpixel import refine_bicubic
        return refine_bicubic(search, template, cx, cy, radius=radius)
    h, w = search.shape
    th, tw = template.shape
    bx, by = cx - tw / 2.0, cy - th / 2.0

    x0, y0 = int(round(bx)) - radius, int(round(by)) - radius
    x1, y1 = x0 + tw + 2 * radius, y0 + th + 2 * radius

    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, w), min(y1, h)
    if x1c - x0c < tw + 1 or y1c - y0c < th + 1:
        return cx, cy, 0.0

    window = search[y0c:y1c, x0c:x1c]
    res = cv2.matchTemplate(window.astype(np.float32),
                            template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    pj, pi = loc

    dx = parabolic(res[pi, pj - 1], res[pi, pj], res[pi, pj + 1]) if 0 < pj < res.shape[1] - 1 else 0.0
    dy = parabolic(res[pi - 1, pj], res[pi, pj], res[pi + 1, pj]) if 0 < pi < res.shape[0] - 1 else 0.0

    return (x0c + pj + dx) + tw / 2.0, (y0c + pi + dy) + th / 2.0, float(score)


# --- Sub-pixel: the centre row's raster-drift sample ------------------------
#
# Raster drift shifts each scan row of the search frame horizontally by its own
# amount, and only horizontally: `generator/src/sem_imaging.py` perturbs `map_x`
# with `row_shift[:, None]` and leaves `map_y` alone. The jitter inside
# `row_shift` is drawn i.i.d. per row. The label then takes the shift of the
# single row the target centre falls on -- `generate.correct_gt` reads
# `row_shift[round(py)]` -- while `refine_zncc` correlates the whole ~100-row
# template and so recovers the row *average*. The residual localisation error
# is therefore exactly `row_shift[centre] - mean(row_shift)`, which no rigid fit
# can remove because the jitter is white.
#
# Measured on the 1,750 present pairs of `data/ext_p2` (shipped weights):
# the error is one-dimensional -- median |dx| 0.793 px against median |dy|
# 0.081 px on set B -- and `std(dx) / drift_jitter_px` is 1.08 (set B) and 1.11
# (set A) across a 14x range of drift. A row-lag scan of the correction peaks
# sharply at the labelled row and sits at ~0.05 on every neighbouring row: the
# signature of a per-row white process.
#
# Correcting x post-hoc from the measured row offset is only worth ~0.5
# correlation, because the same wobble also degrades the rigid match that
# produced x. Dewarping the rows first and re-matching on the flattened patch
# fixes both, and is what this pair of functions does.


def row_pitch(template: np.ndarray) -> float | None:
    """Dominant horizontal period of the layout, from the template's own
    mean row autocorrelation.

    The search frame's lattice pitch is ~10 px, while raster drift has sd <=
    2.1 px. A row whose correlation peak sits a whole pitch off the field is
    therefore a repeat error, not a 5-sigma drift sample -- which is what
    `drift_row_refine` uses this for. Measured on 900 pairs: resolvable on 94%,
    median 9.9 px.
    """
    t = template.astype(np.float32)
    t = t - t.mean(axis=1, keepdims=True)
    f = np.fft.rfft(t, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1, n=t.shape[1]).mean(axis=0)
    ac = ac[:min(40, len(ac))]
    if ac.size < 6 or ac[0] <= 0:
        return None
    ac = ac / ac[0]
    for k in range(3, len(ac) - 1):
        if ac[k] > ac[k - 1] and ac[k] >= ac[k + 1] and ac[k] > 0.25:
            return float(k + parabolic(ac[k - 1], ac[k], ac[k + 1]))
    return None


def row_offsets(search: np.ndarray, template: np.ndarray, cx: float, cy: float,
                lag: int = DRIFT_ROW_LAG, return_corr: bool = False):
    """Per-row horizontal offset between the search frame and the posed template.

    `make_template` returns the reference already rotated and scaled into the
    search frame, so template row i lines up with exactly one search row --
    which is the granularity raster drift acts on.

    Returns `(offset, peak)`, each of length `template.shape[0]`, with NaN where
    the 1-D correlation peak landed on the window edge and cannot be
    interpolated.
    """
    th, tw = template.shape
    y0 = int(round(cy - th / 2.0))
    x0 = int(round(cx - tw / 2.0))
    h, w = search.shape
    if y0 < 0 or y0 + th > h or x0 - lag < 0 or x0 + tw + lag > w:
        return (None, None, None) if return_corr else (None, None)

    win = search[y0:y0 + th, x0 - lag:x0 + tw + lag].astype(np.float32)
    tpl = template.astype(np.float32)
    tpl = tpl - tpl.mean(axis=1, keepdims=True)
    tn = np.sqrt((tpl ** 2).sum(axis=1))
    tn[tn < 1e-6] = 1e-6

    nlag = 2 * lag + 1
    corr = np.empty((th, nlag), np.float32)
    for i in range(nlag):
        seg = win[:, i:i + tw]
        seg = seg - seg.mean(axis=1, keepdims=True)
        sn = np.sqrt((seg ** 2).sum(axis=1))
        sn[sn < 1e-6] = 1e-6
        corr[:, i] = (seg * tpl).sum(axis=1) / (sn * tn)

    k = np.argmax(corr, axis=1)
    off = np.full(th, np.nan)
    peak = np.full(th, np.nan)
    for i, ki in enumerate(k):
        if ki == 0 or ki == nlag - 1:
            continue                      # peak on the edge: the true one is outside
        off[i] = (ki - lag) + parabolic(corr[i, ki - 1], corr[i, ki], corr[i, ki + 1])
        peak[i] = corr[i, ki]
    if return_corr:
        return off, peak, corr
    return off, peak


def drift_row_refine(search: np.ndarray, template: np.ndarray, cx: float, cy: float,
                     radius: int = 3, lag: int = DRIFT_ROW_LAG,
                     min_corr: float = DRIFT_ROW_MIN_CORR,
                     max_shift: float = DRIFT_MAX_SHIFT) -> tuple[float, float] | None:
    """Re-place a match at the drift row the label is defined on.

    Returns the corrected `(x, y)`, or None to decline -- the caller then keeps
    the rigid estimate. Declining is deliberate and common (~19% of pairs):
    loosening `min_corr` to correct more pairs was measured *worse* on the full
    set, because a badly measured row is worse than no correction at all.
    """
    off, peak, corr = row_offsets(search, template, cx, cy, lag=lag, return_corr=True)
    if off is None:
        return None
    ok = np.isfinite(off) & (peak > min_corr)
    if ok.sum() < 12:
        return None

    # Unwrap rows that locked onto the neighbouring lattice repeat. Accept the
    # shifted candidate only where the correlation there is still competitive,
    # so a genuinely large drift sample is never rewritten. Measured on 900
    # pairs: coverage 69% -> 77%, set B <=1px 67.8% -> 70.0%, set A 94.1% ->
    # 94.8% -- every component moves the right way at once.
    pitch = row_pitch(template)
    if pitch is not None and 4.0 <= pitch <= 30.0:
        centre = float(np.median(off[ok]))
        nlag = 2 * lag + 1
        for i in range(len(off)):
            if not np.isfinite(off[i]):
                continue
            best = off[i]
            for n in (-2, -1, 1, 2):
                cand = off[i] + n * pitch
                if abs(cand - centre) < abs(best - centre) - 1e-9 and abs(cand) <= lag:
                    j = int(round(cand + lag))
                    if 0 < j < nlag - 1 and corr[i, j] > 0.6 * peak[i]:
                        best = cand
            off[i] = best
        ok = np.isfinite(off) & (peak > min_corr)

    th, tw = template.shape
    y0 = int(round(cy - th / 2.0))
    ci = int(round(cy)) - y0              # the search row `correct_gt` labels against
    if not (0 <= ci < th) or not ok[ci]:
        return None                       # the row that decides the answer is unusable

    # Gaps are interpolated only to flatten the patch; the centre row itself is
    # never interpolated (see the `ok[ci]` guard) because drift is white and an
    # interpolated value carries none of its neighbours' information.
    dense = np.interp(np.arange(th), np.arange(th)[ok], off[ok])

    # Scale the runaway guard to the drift this pair actually shows: at severity
    # 4 (sd ~2 px) a 3 px correction is an ordinary sample, while on a quiet
    # frame it is a mis-read. Fixed 3.0 px scored 36.33 against 36.50 here.
    idx = np.arange(th)[ok]
    resid_sd = float(np.std(off[ok] - np.polyval(np.polyfit(idx, off[ok], 3), idx)))
    max_shift = float(np.clip(DRIFT_CLAMP_K * resid_sd, 2.0, max_shift))

    h, w = search.shape
    pad = 2 * radius + tw // 8
    ya, yb = max(y0, 0), min(y0 + th, h)
    xa = max(int(round(cx - tw / 2.0)) - pad, 0)
    xb = min(int(round(cx + tw / 2.0)) + pad, w)
    if yb - ya < th or xb - xa < tw + 2:
        return None

    patch = search[ya:yb, xa:xb].astype(np.float32)
    shift = np.zeros(patch.shape[0], np.float32)
    n = min(th, patch.shape[0])
    shift[:n] = dense[:n]
    map_x = np.arange(patch.shape[1], dtype=np.float32)[None, :] + shift[:, None]
    map_y = np.tile(np.arange(patch.shape[0], dtype=np.float32)[:, None],
                    (1, patch.shape[1]))
    flat = cv2.remap(patch, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)

    rx, ry, _ = refine_zncc(standardize(flat / 255.0), standardize(template / 255.0),
                            cx - xa, cy - ya, radius=radius)
    nx, ny = rx + xa + float(dense[ci]), ry + ya
    if not np.isfinite(nx) or abs(nx - cx) > max_shift:
        return None                       # runaway re-match; keep the rigid answer
    return float(nx), float(ny)


def zncc_only(reference: np.ndarray, search: np.ndarray) -> dict:
    """Classical multi-scale ZNCC. Used as the fallback path when no trained
    weights are available, so the inference script always returns a result."""
    best = None
    # +/-10% around each plausible factor, matching the upstream baseline's
    # 9-11x sweep but anchored on whatever scale the reference actually is.
    scales = [f * m for f in template_hypotheses(reference)
              for m in (0.9, 0.95, 1.0, 1.05, 1.1)]
    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        tmpl = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if best is None or score > best["score"]:
            best = {"x": loc[0] + tw / 2.0, "y": loc[1] + th / 2.0, "score": float(score)}
    if best is None:
        return {"x": search.shape[1] / 2.0, "y": search.shape[0] / 2.0, "score": 0.0}
    return best


def polish_pose(reference: np.ndarray, search: np.ndarray, x: float, y: float,
                magnification: float, rotation_deg: float,
                scale_band: float = 0.03, rot_band: float = 0.8,
                rounds: int = 2, iters: int = 7) -> tuple[float, float, float]:
    """Re-fit (scale, rotation) against a known match location.

    The first pose estimate is made before the match is located, from a coarse
    peak on half-resolution probes -- good enough to canonicalise, but the
    scale credit tier is 1% wide and that estimate sits right on it. Once the
    ZNCC stage has placed the match to sub-pixel accuracy, the pose can be
    re-fit against that location instead: the correlation is evaluated in a
    small window around the known centre, so it is both sharper (no competing
    repeats in frame) and cheap (a ~250 px window, not 1000 px).

    Returns (scale, rotation, peak). Bands are deliberately narrow -- this
    polishes an estimate, it does not search.

    Both axes are polished. An earlier version deliberately threw the scale
    result away because polishing it *lowered* scale credit (0.860 -> 0.808),
    which was read at the time as correlation-vs-scale being flat inside the
    window. It was flat, but not for that reason: `make_template` quantised the
    realised scale to `W / round(W / m)`, so the objective really was constant
    across plateaus ~1% wide and the search returned an arbitrary point inside
    one. With the template continuous in `m`, and the canvas pinned so that
    every candidate is scored over the same pixel count, the objective is a
    genuine unimodal peak and the scale result is worth keeping.
    """
    m, r = float(magnification), float(rotation_deg)
    ds, dr = m * scale_band, rot_band

    # Pin the canvas to the *smallest* template in the band, so content always
    # covers it and every candidate is scored over an identical pixel count.
    h, w = reference.shape[:2]
    hi = m + ds
    canvas = (max(int(np.floor(h / hi)), 1), max(int(np.floor(w / hi)), 1))

    th, tw = canvas
    pad = int(max(th, tw) * 0.6)
    y0 = max(int(round(y - th / 2.0)) - pad, 0)
    x0 = max(int(round(x - tw / 2.0)) - pad, 0)
    win = search[y0:int(round(y + th / 2.0)) + pad, x0:int(round(x + tw / 2.0)) + pad]

    def fit(mm, rr):
        t = make_template(reference, mm, rr, canvas=canvas)
        if t.shape[0] >= win.shape[0] or t.shape[1] >= win.shape[1]:
            return -np.inf
        return float(cv2.minMaxLoc(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED))[1])

    peak = fit(m, r)
    for _ in range(rounds):
        m, peak = _golden_max(lambda v: fit(v, r), m - ds, m + ds, iters)
        r, peak = _golden_max(lambda v: fit(m, v), r - dr, r + dr, iters)
        ds, dr = ds / 3.0, dr / 3.0
    return m, r, peak


@torch.no_grad()
def locate_phase2(model, reference: np.ndarray, search: np.ndarray, device,
                  refine: bool = True, pose: tuple[float, float] | None = None,
                  refine_radius: int = REFINE_RADIUS, polish: bool = True,
                  polish_scale: bool = True, refit_xy: bool = False,
                  hypotheses: int = 3, coarse_scales: int = COARSE_SCALES,
                  band: bool = False, return_hypotheses: bool = False,
                  early_exit_zncc: float | None = None,
                  rescue_margin: float | None = None, rescue_delta: float = 0.0,
                  verification: str = "zncc", denoise: int = 0,
                  subpixel_rows: bool = True, **kw) -> dict:
    """Phase 2 inference: unknown scale and rotation, with a rejection score.

    band=False is the measured default (full 2,250-pair A/B, 2026-08-31):
    band-passing the coarse probe cost 0.45 rubric points (paired loc delta
    +0.167, CI [+0.014, +0.327]; rescued 18 / broken 9) and buys no measurable
    time. The old True default shipped without an A/B -- see issue #9.

    For each pose hypothesis: canonicalise the search frame, let the network
    make the coarse decision on the matched-scale input it was trained for,
    map the answer back, and verify by ZNCC at native resolution. The
    hypothesis with the best native ZNCC wins.

    Two deliberate choices:

    * **Refinement happens in the native frame, never the canonical one.**
      Otherwise the answer inherits the resampling blur, and the credit tiers
      (1.00 at 1 px against 0.40 at 5 px) make that expensive.
    * **The winner is chosen by native ZNCC, not by the network score or the
      low-resolution pose peak.** A wrong scale basin can out-score the right
      one on a half-resolution probe -- that was every measured localisation
      failure -- but at full resolution the wrong basin correlates near zero
      while the right one is around 0.9, so the decision becomes easy.

    band defaults to False: it mirrors the shipped decode (register.py passes
    band=False; the DoG pre-filter measured negative in #18/#24). Pass
    band=True only for A/B measurement.
    """
    verification = str(verification).lower()
    valid_verification = {"zncc", "majority", "consensus"}
    if verification not in valid_verification:
        raise ValueError(f"verification must be one of {sorted(valid_verification)}")

    # The baseline path never enters this block. Research instrumentation and
    # optional selectors share one set of full-search feature maps per pair.
    need_verification_scores = return_hypotheses or verification != "zncc"
    search_features = None
    verification_secs = 0.0
    if need_verification_scores:
        import time
        t_verify = time.perf_counter()
        search_features = {
            "rank": rank_transform(search),
            "band": common_band(search),
        }
        if return_hypotheses:
            search_features["dog"] = dog_feature(search)
        verification_secs += time.perf_counter() - t_verify

    # Suppress impulse noise before anything looks at the frame. Every
    # similarity tried so far changed the *metric*; this changes the *input*,
    # which is a different axis and the one the registration literature points
    # at -- noise corrupts the similarity surface itself, so a more robust
    # statistic on a corrupted surface is fighting the wrong battle. Set B's
    # failures are led by salt-and-pepper (Cohen's d = 1.21), speckle (1.18)
    # and detector noise (1.15), and a median filter is the textbook answer to
    # the first. Costs ~2 ms against a 3.35 s pair.
    # ...but only on the *correlation* path. Measured on one Set B pair, a 3x3
    # median lifts native ZNCC 0.812 -> 0.855 while dropping the network score
    # 0.714 -> 0.361: the network was trained on noisy frames, so denoising its
    # input is a distribution shift and it pays for it. A 5x5 median moves the
    # answer 600 px, i.e. destroys the match outright.
    #
    # So the network keeps raw pixels and the classical stages -- the coarse
    # pose sweep and the full-resolution ZNCC -- get the cleaned frame. They
    # are the parts that noise actually corrupts, and they have no learned
    # distribution to violate.
    search_corr = search
    if denoise and denoise >= 3:
        search_corr = cv2.medianBlur(search, int(denoise) | 1)

    search_corr_std = standardize(search_corr / 255.0) if refine else None

    def attempt(m: float, rot: float) -> dict:
        nonlocal verification_secs
        canon, M = canonicalize_search(search, m, rot)
        coarse = locate(model, reference, canon, device, refine=False,
                        factor=float(SCALE), rotation_deg=0.0, **kw)
        cx, cy = uncanonicalize_point(M, coarse["x"], coarse["y"])
        out = dict(coarse)
        out.update({"x": cx, "y": cy, "scale": float(m), "theta": float(rot),
                    "canonical": (coarse["x"], coarse["y"])})
        if return_hypotheses:
            out.update({"coarse_x_native": float(cx), "coarse_y_native": float(cy)})
        template = None
        if refine:
            template = make_template(reference, m, rot)
            rx, ry, zn = refine_zncc(search_corr_std,
                                     standardize(template / 255.0),
                                     cx, cy, radius=refine_radius)
            if np.hypot(rx - cx, ry - cy) <= 10.0:
                out.update({"x": rx, "y": ry})
            out["zncc"] = float(zn)
        if search_features is not None:
            import time
            t_verify = time.perf_counter()
            if template is None:
                template = make_template(reference, m, rot)
            out.update({
                "rank": local_match_score(search_features["rank"],
                                          rank_transform(template), out["x"], out["y"]),
                "band": local_match_score(search_features["band"],
                                          common_band(template), out["x"], out["y"]),
            })
            if "dog" in search_features:
                out["dog"] = local_match_score(search_features["dog"],
                                                dog_feature(template), out["x"], out["y"])
            verification_secs += time.perf_counter() - t_verify
        return out

    def choose(candidates: list[dict]) -> dict:
        zncc_i = max(range(len(candidates)),
                     key=lambda i: candidates[i].get("zncc", candidates[i].get("score", -np.inf)))
        if verification == "zncc" or len(candidates) == 1:
            return candidates[zncc_i]
        rank_i = max(range(len(candidates)), key=lambda i: candidates[i]["rank"])
        band_i = max(range(len(candidates)), key=lambda i: candidates[i]["band"])
        if verification == "consensus":
            return candidates[rank_i] if rank_i == band_i else candidates[zncc_i]
        votes = [zncc_i, rank_i, band_i]
        winner = max(range(len(candidates)), key=votes.count)
        return candidates[winner] if votes.count(winner) >= 2 else candidates[zncc_i]

    if pose is not None:
        candidates = [attempt(float(pose[0]), float(pose[1]))]
        candidates[0]["pose_peak"] = float("nan")
        best = choose(candidates)
    else:
        cands = pose_candidates(reference, search_corr, k=max(int(hypotheses), 1),
                                coarse_scales=int(coarse_scales), band=band)
        candidates = []
        for m, rot, coarse_peak in cands:
            r = attempt(m, rot)
            r["pose_peak"] = float(coarse_peak)
            candidates.append(r)
            # Early exit. `pose_candidates` returns hypotheses already ranked by
            # coarse peak and `choose` takes the highest native ZNCC, so once a
            # candidate verifies strongly enough there is nothing for the rest to
            # win. The network is ~86% of a pair and is paid once per hypothesis,
            # so stopping here is close to a 3x saving on the pairs that take it.
            if (early_exit_zncc is not None
                    and len(candidates) < len(cands)
                    and r.get("zncc", -np.inf) >= early_exit_zncc):
                break
            # Uncontested top candidate exit: when hypothesis 1 exhibits high
            # network score, strong ZNCC verification, no rival peak, and a clear
            # coarse lead over runner-up, remaining candidates cannot overturn it.
            if (len(candidates) == 1
                    and len(cands) > 1
                    and r.get("score", 0.0) >= 0.75
                    and r.get("zncc", -np.inf) >= 0.75
                    and r.get("peak_ratio", 1.0) <= 0.35
                    and (cands[0][2] - cands[1][2] >= 0.05)):
                break
        best = choose(candidates)

        # --- rescue pass (issue #5), off by default -----------------------
        # The coarse sweep is discrete, so when two hypotheses finish close
        # together the true pose is often *between* them rather than at either.
        # The oracle puts 58% of the remaining set B gap on the pose search and
        # the failures are wrong-scale lock-ons (median 6.65% scale error
        # against 0.60% on successes), while widening the sweep globally is
        # monotonically worse -- more candidates means more decoys on the ~87%
        # already correct. Gating on a contested decision keeps the extra
        # candidates away from the pairs that do not need them.
        #
        # Measured negative across five settings on the full 2250 (paired delta
        # -0.022 to -0.069, none near the +0.35 gate). Kept because it is cheap
        # to re-test on new weights, not because it currently pays.
        if rescue_margin is not None and len(candidates) > 1:
            gate = winner_margin(candidates, best)
            if np.isfinite(gate) and gate < rescue_margin:
                def _v(c):
                    return float(c.get("zncc", c.get("score", -np.inf)))
                top = sorted(candidates, key=_v, reverse=True)[:2]
                ds = abs(top[0]["scale"] - top[1]["scale"]) or 0.05
                dr = abs(top[0]["theta"] - top[1]["theta"]) or 0.5
                extra = [(0.5 * (top[0]["scale"] + top[1]["scale"]),
                          0.5 * (top[0]["theta"] + top[1]["theta"])),
                         (top[0]["scale"] + 0.5 * ds, top[0]["theta"]),
                         (top[0]["scale"] - 0.5 * ds, top[0]["theta"]),
                         (top[0]["scale"], top[0]["theta"] + 0.5 * dr),
                         (top[0]["scale"], top[0]["theta"] - 0.5 * dr)]
                lo_s, hi_s = PHASE2_SCALE_BOUNDS
                lo_r, hi_r = PHASE2_ROTATION_BOUNDS
                rescued = []
                for mm, rr in extra:
                    if not (lo_s <= mm <= hi_s and lo_r <= rr <= hi_r):
                        continue
                    c = attempt(float(mm), float(rr))
                    c["pose_peak"] = float("nan")
                    rescued.append(c)
                if rescued:
                    challenger = max(rescued, key=_v)
                    if _v(challenger) > _v(best) + rescue_delta:
                        best = challenger
                        best["rescued"] = True
                    candidates.extend(rescued)
        best["n_hypotheses"] = len(cands)
    # The SELECTED winner's min(score, zncc) margin over the best runner-up:
    # an inference-time uncertainty feature for the present/absent rejector
    # (issue #6). Needs only the candidate list the decode already built, so
    # it is recorded on every path -- including the shipped default-zncc one.
    best["winner_margin"] = winner_margin(candidates, best)

    if refine and polish:
        pm, pr, _ = polish_pose(reference, search, best["x"], best["y"],
                                best["scale"], best["theta"])
        best["theta"] = float(pr)
        # Scale is adopted too now that `make_template` is continuous in it and
        # `polish_pose` pins the canvas; see the note in `polish_pose`. Kept
        # switchable because this was the one change that previously made the
        # metric worse, and a regression here is worth being able to isolate.
        if polish_scale:
            best["scale"] = float(pm)

        # The ZNCC snap that placed x, y ran against a template built from the
        # *pre-polish* pose. A 0.9% scale error displaces the template's own
        # edges by ~0.4 px relative to its centre, which broadens the
        # correlation peak and biases the parabolic sub-pixel fit. Re-snapping
        # against the polished template costs one more matchTemplate over a
        # +/-2 px window and is the only route by which better pose feeds back
        # into the 40-point metric. Off by default: it is the change most able
        # to trade localisation for pose, so it ships only if measured to help.
        if refit_xy:
            tpl = make_template(reference, best["scale"], best["theta"])
            rx, ry, zn = refine_zncc(standardize(search / 255.0),
                                     standardize(tpl / 255.0),
                                     best["x"], best["y"], radius=2)
            if np.hypot(rx - best["x"], ry - best["y"]) <= 3.0:
                best.update({"x": rx, "y": ry, "zncc": float(zn)})

    # Re-place the match on the scan row the label is actually defined against.
    # Runs after every pose decision is final, so it can only move x -- it never
    # feeds back into scale, rotation or the confidence, and a decline leaves the
    # rigid answer untouched. Measured on all 1,750 present pairs of
    # `data/ext_p2` (shipped weights, full-set A/B):
    #
    #   localisation 35.71 -> 36.29 / 40   (+0.58)
    #   set A credit 0.9758 -> 0.9806, <=1px 92.7% -> 95.1%
    #   set B credit 0.8247 -> 0.8471, <=1px 57.6% -> 67.0%
    #
    # y is deliberately left alone: it is already at 0.081 px median error on
    # set B because raster drift has no vertical component.
    if subpixel_rows:
        # Never let the refinement cost a pair. register.py zero-fills the whole
        # row on any exception, so a throw here would turn a correctly located
        # pair into found=0 with score 0.0 -- indistinguishable, in the output
        # contract, from a confident rejection. `np.polyfit` can raise
        # LinAlgError on a degenerate fit, and cv2.remap can reject an
        # unexpected dtype/shape; both are recoverable by simply keeping the
        # rigid answer, which is what the pipeline produced before this stage.
        try:
            tpl = make_template(reference, best["scale"], best["theta"])
            moved = drift_row_refine(search, tpl, best["x"], best["y"])
            if moved is not None:
                best["x"] = moved[0]
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"sub-pixel row refinement skipped: "
                          f"{type(e).__name__}: {e}", RuntimeWarning, stacklevel=2)

    # The statement guarantees the true pose lies in these boxes and the rules
    # explicitly permit hard-coding them, so clipping a reported value into the
    # feasible set can only reduce the error -- it is arithmetic, not a
    # heuristic. It matters because `polish_pose` searches a band around its
    # starting estimate without regard to the bound, so a pair whose true
    # magnification sits near 8 or 12 can be polished just outside it. Measured
    # on 400 external present pairs: 9 predictions fell outside [8, 12] and 4
    # outside +/-5 deg, and clipping them lifted scale credit 0.9000 -> 0.9057.
    best["scale"] = float(np.clip(best["scale"], *PHASE2_SCALE_BOUNDS))
    best["theta"] = float(np.clip(best["theta"], *PHASE2_ROTATION_BOUNDS))

    # Reported confidence. TWO definitions exist, selected by
    # driftsense.config.SHIPPED_CONFIDENCE (the ONE definition; the parity
    # test pins register.py and eval_ext.py to it):
    #
    # "fused6" (implemented, NOT shipped -- measured out, see config): a
    # 6-feature logistic over statistics this decode
    # already computes -- score, zncc, peak_ratio, pose_peak, psr, apce
    # (driftsense.calibration.calibrate(), frozen constants, zero inference
    # cost). Held-out 4-fold CV on the 2,250-pair holdout: AUC 0.9877 ->
    # 0.9915 vs the legacy scalar (.agents/B_CALIBRATION_REPORT.md, protocol
    # identical to REJECTOR_FINDINGS.md). A monotone map of the legacy scalar
    # provably cannot move AUC (Guo et al., arXiv:1706.04599); the gain comes
    # from recombining the six signals, not from rescaling one.
    #
    # "legacy_min": the historical min(). They fail differently, which is why
    # the minimum was chosen originally: the network can be confident on a
    # plausible wrong repeat -- it is a relative judgement and something
    # always wins -- while ZNCC can be respectable on a degraded frame with no
    # true instance. The fused statistic subsumes both heights plus four
    # peak-quality/contest signals, and measured better on held-out AUC.
    if SHIPPED_CONFIDENCE == "fused6":
        from driftsense.calibration import calibrate_shipped
        best["confidence"] = float(calibrate_shipped({
            "score": float(best.get("score", 0.0)),
            "zncc": float(best.get("zncc", best.get("score", 0.0))),
            "peak_ratio": float(best.get("peak_ratio", np.nan)),
            "pose_peak": float(best.get("pose_peak", np.nan)),
            "psr": float(best.get("psr", np.nan)),
            "apce": float(best.get("apce", np.nan)),
        }))
    else:
        best["confidence"] = float(min(float(best.get("score", 0.0)),
                                       float(best.get("zncc", best.get("score", 0.0)))))

    if return_hypotheses:
        best["hypotheses"] = [{
            "scale": float(r["scale"]),
            "theta": float(r["theta"]),
            "pose_peak": float(r.get("pose_peak", np.nan)),
            "network_score": float(r.get("score", np.nan)),
            "peak_ratio": float(r.get("peak_ratio", np.nan)),
            "coarse_x_native": float(r["coarse_x_native"]),
            "coarse_y_native": float(r["coarse_y_native"]),
            "x": float(r["x"]),
            "y": float(r["y"]),
            "zncc": float(r.get("zncc", np.nan)),
            "rank": float(r.get("rank", np.nan)),
            "band": float(r.get("band", np.nan)),
            "dog": float(r.get("dog", np.nan)),
        } for r in candidates]
        best["secs_verification"] = float(verification_secs)
    else:
        # Optional selectors change only the chosen hypothesis. Keep the
        # WINNER's rank/band (drop dog — only computed under return_hypotheses):
        # eval_ext records them, which is what lets rejector_cv.py fit the
        # present/absent rejector on features that exist at inference time
        # (issue #6). The default zncc path never computes them, so the result
        # contract register.py consumes is unchanged there.
        best.pop("dog", None)
    return best


@torch.no_grad()
def locate(model, reference: np.ndarray, search: np.ndarray, device,
           refine: bool = True, return_heatmap: bool = False,
           tie_tol: float = TIE_REL_TOL, refine_radius: int = REFINE_RADIUS,
           refine_accept_px: float = 10.0, factor: float = SCALE,
           rotation_deg: float = 0.0,
           ref_feat: "torch.Tensor | None" = None) -> dict:
    """Full inference: reference + search (uint8 grayscale) -> centre (x, y).

    ref_feat: optional precomputed template-branch embedding. The template is
    make_template(reference, SCALE, 0) for every pose hypothesis (the pose is
    applied by canonicalizing the search), so callers that attempt several
    hypotheses of one pair can encode the template once and pass it here --
    output-identical, minus the redundant encoder passes (issue #7, E1)."""
    model.eval()

    template = make_template(reference, factor, rotation_deg)
    th, tw = template.shape

    tpl_n = standardize(template / 255.0)
    sea_n = standardize(search / 255.0)

    t = torch.from_numpy(tpl_n)[None, None].to(device)
    s = torch.from_numpy(pad_to_stride(sea_n))[None, None].to(device)

    out = model(t, s, ref_feat=ref_feat)
    prob = torch.sigmoid(out["logit"])[0, 0].float().cpu().numpy()
    offs = out["offset"][0].float().cpu().numpy()

    i, j, score = select_peak(prob, search.shape, (th, tw), tie_tol)
    cx, cy = response_to_center(i, j, th, tw,
                                float(offs[1, i, j]), float(offs[0, i, j]))

    # How contested is this decision? On a periodic layout the runner-up is a
    # decoy one lattice period away, so the ratio between the two strongest
    # *well-separated* peaks says how much the network is actually committing.
    # A ratio near 1 means it is guessing between repeats -- the exact case
    # dihedral voting is there to arbitrate.
    peaks = find_peaks(prob)
    rival = next((p_ for p_ in peaks
                  if np.hypot(p_[0] - i, p_[1] - j) > 2.0), None)
    peak_ratio = (float(rival[2]) / max(float(peaks[0][2]), 1e-9)) if rival else 0.0

    psr, apce = peak_stats(prob, i, j)
    result = {"x": cx, "y": cy, "score": float(score), "coarse": (cx, cy),
              "peak_ratio": peak_ratio, "psr": psr, "apce": apce}

    if refine:
        rx, ry, zn = refine_zncc(sea_n, tpl_n, cx, cy, radius=refine_radius)
        # Accept the snap only if it stayed in the same neighbourhood; a large
        # jump means correlation latched onto an adjacent repeat, which is
        # exactly the failure the network is there to avoid.
        if np.hypot(rx - cx, ry - cy) <= refine_accept_px:
            result.update({"x": rx, "y": ry, "zncc": zn})

    h, w = search.shape
    result["x"] = float(np.clip(result["x"], 0, w - 1))
    result["y"] = float(np.clip(result["y"], 0, h - 1))
    if return_heatmap:
        result["heatmap"] = prob
    return result


# ---------------------------------------------------------------------------
# Test-time augmentation
#
# 20 of the 24 test failures are wrong-repeat lock-ons, and 21 of 24 carry a
# confidence below 0.5 -- the model is genuinely uncertain on exactly the ones
# it gets wrong. Running the 8 square symmetries (all seen during training, so
# all in-distribution) and letting them vote turns that uncertainty into a
# usable signal: a decoy that wins under one view rarely wins under all eight,
# whereas the true site is stable.
# ---------------------------------------------------------------------------

def _dihedral_img(img: np.ndarray, t: int) -> np.ndarray:
    k, flip = t % 4, t // 4
    if k:
        img = np.rot90(img, k)
    if flip:
        img = np.fliplr(img)
    return np.ascontiguousarray(img)


def _dihedral_point_inv(x: float, y: float, shape: tuple[int, int],
                       t: int) -> tuple[float, float]:
    """Map a point in the transformed frame back to original coordinates.

    `shape` is the ORIGINAL (h, w) of the search image. Forward is `k`
    counter-clockwise rot90s then an optional fliplr, so the inverse undoes the
    flip first and then applies the inverse rotation k times.

    Each rot90 swaps the axes, so the width used to mirror a coordinate changes
    at every step -- tracking it is what makes this correct on non-square
    frames as well as on the 1000x1000 ones the spec defines. Sharing a single
    `size` here silently returns garbage the moment the search image is not
    square.
    """
    k, flip = t % 4, t // 4
    h, w = shape

    # Dimensions of the frame the point currently lives in: k rot90s swap h/w
    # for odd k, and fliplr leaves them alone.
    ch, cw = (w, h) if k % 2 else (h, w)

    if flip:
        x = cw - 1 - x
    for _ in range(k):
        # Undo one rot90: (x, y) in the rotated frame came from (cw' - 1 - y, x)
        # in the frame one step back, whose width is the rotated frame's height.
        x, y = ch - 1 - y, x
        ch, cw = cw, ch
    return x, y


@torch.no_grad()
def locate_tta(model, reference: np.ndarray, search: np.ndarray, device,
               transforms=range(8), cluster_px: float = 6.0,
               refine: bool = True, verify_alpha: float = VERIFY_ALPHA,
               verify_top_k: int = VERIFY_TOP_K, factor: float = SCALE,
               rotation_deg: float = 0.0) -> dict:
    """Dihedral test-time augmentation with cluster voting.

    Each view proposes a centre; proposals are mapped back to the original
    frame and grouped. The group with the greatest total confidence wins, and
    its members are averaged (confidence-weighted). Voting on *agreement*
    rather than averaging heatmaps is what makes this robust: a wrong-repeat
    proposal is an outlier one period away, and averaging it in would drag the
    answer off, while clustering discards it.
    """
    props = []
    for t in transforms:
        r_t = _dihedral_img(reference, t)
        s_t = _dihedral_img(search, t)
        res = locate(model, r_t, s_t, device, refine=False, factor=factor,
                     rotation_deg=rotation_deg)
        x, y = _dihedral_point_inv(res["x"], res["y"], search.shape, t)
        props.append((x, y, res["score"]))

    # Greedy clustering by proximity, strongest proposal first.
    props.sort(key=lambda p: -p[2])
    clusters: list[list[tuple]] = []
    for p in props:
        for c in clusters:
            if np.hypot(p[0] - c[0][0], p[1] - c[0][1]) <= cluster_px:
                c.append(p)
                break
        else:
            clusters.append([p])

    tpl_n = standardize(make_template(reference, factor, rotation_deg) / 255.0)
    sea_n = standardize(search / 255.0)

    def _centroid(c):
        w = np.array([m[2] for m in c], dtype=np.float64)
        return (float(np.average([m[0] for m in c], weights=w)),
                float(np.average([m[1] for m in c], weights=w)))

    ranked = sorted(clusters, key=lambda c: -sum(m[2] for m in c))

    if verify_alpha > 0 and len(ranked) > 1:
        # Two-stage arbitration: the network shortlists candidate regions and a
        # full-resolution ZNCC check at each one breaks the tie. ZNCC is an
        # *independent* signal from the network's own confidence, but it cannot
        # be trusted alone -- searching the whole frame it picks the wrong
        # repeat ~50% of the time, and even choosing among these shortlisted
        # candidates it scored 0.920 vs 0.943 on validation because it prefers
        # crisper-looking decoys. Adding the network score back as a prior is
        # what makes it a small net win instead of a regression.
        cands = ranked[:verify_top_k]
        total = sum(sum(m[2] for m in c) for c in cands) or 1.0
        best, best_s = ranked[0], -np.inf
        for c in cands:
            ccx, ccy = _centroid(c)
            _, _, zn = refine_zncc(sea_n, tpl_n, ccx, ccy)
            s = zn + verify_alpha * (sum(m[2] for m in c) / total)
            if s > best_s:
                best, best_s = c, s
    else:
        best = ranked[0]

    w = np.array([m[2] for m in best], dtype=np.float64)
    cx, cy = _centroid(best)

    result = {"x": cx, "y": cy,
              "score": float(w.mean()),
              "votes": len(best), "n_views": len(props),
              "agreement": len(best) / max(len(props), 1)}

    if refine:
        rx, ry, zn = refine_zncc(sea_n, tpl_n, cx, cy)
        if np.hypot(rx - cx, ry - cy) <= 10.0:
            result.update({"x": rx, "y": ry, "zncc": zn})

    h, w_ = search.shape
    result["x"] = float(np.clip(result["x"], 0, w_ - 1))
    result["y"] = float(np.clip(result["y"], 0, h - 1))
    return result

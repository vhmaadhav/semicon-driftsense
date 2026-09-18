"""CAD-anchored Phase 3 registration: reference CAD -> search CAD -> SEM image.

Phase 3 ships the search site's own CAD (`search_gds_path`, kept on the blind
split) next to the reference CAD. That changes what has to be solved:

  1. **Where is the reference in the design?** A CAD-to-CAD question with an
     exact answer. Both files are the same undistorted design database (the
     organizer's export writes `search.gds` from the same `design_cell`s the
     reference is clipped from), so the reference's polygons reappear in the
     search CAD at one integer-nm offset, and nowhere else. No noise, no
     brightness to infer. A reference cut from an unrelated canvas (the ~1 in
     12 no-match sites) has no such offset, which is the rejection signal.

  2. **How does the design sit in the image?** One rigid transform for the
     whole frame. The generator images the full canvas and rotates it about
     its centre (`src/cad_pipeline.py`, render_cad_sample), and its label is
     that rotation applied to the design-window centre. Estimated here over
     the whole frame -- 1000x1000 px of evidence instead of the 100x100 px the
     reference covers -- so the lattice ambiguity that dominates image-only
     matching never arises.

The label then follows by arithmetic: the generator's own mapping from the
design window to search pixels, with the estimated angle. Raster drift, jitter
and shear displace image content but are *not* in the label (the generator
computes it before imaging), so the estimate deliberately ignores them:
rotation is read from vertical displacements only, the axis drift cannot
reach (the same argument as issue #88's strip rotation).

Per-layer brightness ("yours to infer, per layer") is fitted by linear least
squares once the frame is aligned: every pixel is a blend of the layers
visible there, so the grey levels are the regression coefficients. The fit's
R^2 is reported -- the brief scores "whether you knew your yield fit was
wrong" under calibration, and this is that number.

Everything here is classical, CPU-only and deterministic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from driftsense.gds import (
    GdsError, REF_SIZE, background_intensity, layer_yield, read_gds_layers, yield_to_intensity,
)

# Search pixels are 10 nm, reference design units are 1 nm (Phase 3 brief).
SEARCH_NM_PER_PX = 10.0
# Top-K coarse CAD-to-CAD peaks that get the exact refinement. At 10 nm/px
# the sampling phase lets a lattice alias outscore the true site: on the dev
# split the truth ranked as low as 10th (0-based 9th) in 6 of 723 present pairs,
# every one lost at K=5. The loop stops at the first exact match, so present
# pairs rarely pay for the extra candidates.
COARSE_CANDIDATES = 20
# Both rasters are blurred before the coarse match. Without it, thin periodic
# features (13 nm fins at 10 nm/px) sample differently at every nm of phase,
# so aliases on the 10 nm grid outscore a true origin that sits off it: in a
# fin-and-gate lattice the truth fell out of the top 40; at sigma 1.5 px it
# ranks first (1.0 -> 2nd, 2.0 smears the peak past the refine window).
COARSE_BLUR_PX = 1.5
# Exact refinement: search window around a coarse peak, and the bounding-box
# size tolerance for "the same polygon" (design coordinates are exact, so this
# only absorbs float formatting).
REFINE_WINDOW_NM = 15.0
REFINE_SIZE_TOL_NM = 0.3
# Fewer interior reference polygons than this and the CAD-to-CAD support is
# not a meaningful fraction.
MIN_INTERIOR_POLYGONS = 4
# Found needs this share of interior polygons at one exact offset -- more when
# there are few of them. Dev split: present pairs 0.81-1.00 (1.00 at 6-7
# polygons), absent at most 0.12 (0.29 at 7 polygons).
SUPPORT_FOUND = 0.5
SUPPORT_FOUND_FEW = 0.8
FEW_POLYGONS = 20
# Tile grid for the fine rotation fit.
TILE_PX = 100
TILE_SEARCH_PX = 4
TILE_MIN_NCC = 0.3
# Tile agreement floors, tried in order. 0.3 is right for a clean capture and
# keeps weak tiles out of the fit; on a starved-dose, high-noise frame it
# admits NOTHING (measured: 0 tiles on 13 of 18 present pairs of the harsh
# 20-pair set), the fit raises, and an exact CAD anchor is thrown away. The
# lower rungs are only reached when the rung above did not answer, so a clean
# pair's fit is bit-for-bit what it was.
TILE_NCC_FLOORS = (0.3, 0.15, 0.05)
FINE_ITERATIONS = 2
# A fitted magnification this far from nominal is believed; smaller values are
# not applied. The CAD generator never scales, and radial distortion (barrel,
# which the organizer rules out of the evaluation sets) reads as up to ~1%
# magnification in the vertical fit -- believing that moved answers by 4 px.
SCALE_BELIEVE = 0.03
# A whole-frame translation this large is believed. Below it, what phase
# correlation sees is raster drift's mean, which the label does not contain:
# on true frames tx ran -1.94..-0.35 px, tracking -shear/2 (r = 0.92). A
# 1.5 px bar applied it on 27% of pairs and cost them ~1.7 px each.
TRANSLATION_BELIEVE_PX = 5.0
# A search CAD must span at least this share of the search frame on each axis.
# Datasets written without a real search CAD point search_gds_path at the
# reference file (1000 nm against a 10000 nm frame); that must fall back,
# not answer.
MIN_FRAME_COVERAGE = 0.5
# And the frame fit must agree on at least this share of all tile positions.
MIN_TILE_SHARE = 0.25
# Confidence bands. A pair leaves this module in one of three states, and they
# are not comparable on one continuous scale: the reference is absent from the
# design; it is present at an exact offset but the design-to-image pose never
# verified; or both are established. The bands are disjoint so the ordering is
# structural rather than something that happens to hold on a given draw --
# within each band the quality terms rank the pairs. Measured: overlapping
# bands cost 0.49 calibration points on the harsh 20-pair set, because a
# correct pair with a weak tile fit scored under an unverified one.
ABSENT_BAND = 0.05              # absent          -> [0.00, 0.05]
UNVERIFIED_BAND = (0.05, 0.05)  # pose unverified -> [0.05, 0.10]
VERIFIED_BAND = (0.10, 0.90)    # verified        -> [0.10, 1.00]


class CadAnchorUnavailable(RuntimeError):
    """The CAD-anchored path cannot answer this pair; decode it another way."""


@dataclass
class SearchCad:
    """The search-side CAD, loaded once per pair."""
    polys: dict
    num_layers: int
    bboxes: dict                    # layer -> (N, 4) [x0, y0, x1, y1] nm
    masks: np.ndarray               # (num_layers, H, W) coverage at SEARCH_NM_PER_PX


@dataclass
class CadAnchorResult:
    found: bool
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    scale: float = 10.0
    score: float = 0.0
    support: float = 0.0            # share of interior reference polygons matched exactly
    origin_nm: tuple | None = None
    coarse_peak: float = 0.0
    n_interior: int = 0
    theta_coarse: float = float("nan")
    tiles_used: int = 0
    tiles_total: int = 0
    tile_ncc: float = 0.0
    tile_resid_px: float = float("nan")
    magnification: float = 1.0
    translation_px: tuple = (0.0, 0.0)
    yield_r2: float = float("nan")
    greys: dict = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------

def _flat(ps: list):
    """All vertices of a polygon list in one array, plus each polygon's start
    index -- so per-polygon work is one vectorised call, not a Python loop
    over tens of thousands of polygons."""
    lens = np.fromiter((len(p) for p in ps), np.int64, len(ps))
    starts = np.zeros(len(ps), np.int64)
    np.cumsum(lens[:-1], out=starts[1:])
    return np.concatenate(ps), starts


def layer_masks(polys: dict, num_layers: int, size: tuple, nm_per_px: float,
                origin=(0.0, 0.0)) -> np.ndarray:
    """Anti-aliased per-layer coverage in [0, 1], shape (num_layers, h, w)."""
    w, h = size
    out = np.zeros((num_layers, h, w), np.float32)
    buf = np.zeros((h, w), np.uint8)
    for L in range(num_layers):
        ps = polys.get(L)
        if not ps:
            continue
        pts, starts = _flat(ps)
        q = np.round((pts - origin) * (16.0 / nm_per_px)).astype(np.int32)
        buf[:] = 0
        cv2.fillPoly(buf, np.split(q, starts[1:]), 255, lineType=cv2.LINE_AA, shift=4)
        out[L] = buf * (1.0 / 255.0)
    return out


def visible_fractions(masks: np.ndarray) -> np.ndarray:
    """Painter's algorithm as fractions: channel 0 is background, channel
    L+1 the share of each pixel where layer L is the top-most surface."""
    n = masks.shape[0]
    out = np.empty((n + 1,) + masks.shape[1:], np.float32)
    cover = np.ones(masks.shape[1:], np.float32)       # share not yet hidden
    for L in range(n - 1, -1, -1):
        out[L + 1] = masks[L] * cover
        cover = cover * (1.0 - masks[L])
    out[0] = cover
    return out


def default_greys(num_layers: int) -> np.ndarray:
    g = [background_intensity()] + [yield_to_intensity(layer_yield(L, num_layers))
                                    for L in range(num_layers)]
    return np.asarray(g, np.float32)


def compose(vis: np.ndarray, greys: np.ndarray) -> np.ndarray:
    return np.tensordot(greys.astype(np.float32), vis, axes=1)


def _bboxes(polys: dict) -> dict:
    out = {}
    for L, ps in polys.items():
        if ps:
            pts, starts = _flat(ps)
            lo = np.minimum.reduceat(pts, starts, axis=0)
            hi = np.maximum.reduceat(pts, starts, axis=0)
            out[L] = np.c_[lo, hi]
    return out


def load_search_cad(path: str, image_shape: tuple) -> SearchCad:
    polys, nl = read_gds_layers(path)
    h, w = image_shape
    pts = np.concatenate([np.concatenate(ps) for ps in polys.values() if ps])
    span = pts.max(axis=0) - pts.min(axis=0)
    need = np.array([w, h], float) * SEARCH_NM_PER_PX * MIN_FRAME_COVERAGE
    if np.any(span < need):
        raise CadAnchorUnavailable(
            f"search CAD spans {span[0]:.0f} x {span[1]:.0f} nm, not the {w * SEARCH_NM_PER_PX:.0f} nm "
            "search frame -- it is not a search-side CAD")
    masks = layer_masks(polys, nl, (w, h), SEARCH_NM_PER_PX)
    if masks.max() <= 0:
        raise CadAnchorUnavailable("search CAD has no geometry inside the search frame")
    return SearchCad(polys=polys, num_layers=nl, bboxes=_bboxes(polys), masks=masks)


# ---------------------------------------------------------------------------
# Step 1: the reference inside the search CAD
# ---------------------------------------------------------------------------

def coarse_cad_peaks(ref_masks: np.ndarray, search_masks: np.ndarray, k: int = COARSE_CANDIDATES):
    """Top-k design offsets (nm) of the reference in the search CAD, by
    variance-weighted per-layer ZNCC at search resolution."""
    acc, wsum = None, 0.0
    for L in range(min(ref_masks.shape[0], search_masks.shape[0])):
        t = ref_masks[L]
        wt = float(t.std())
        if wt < 1e-3:
            continue
        r = cv2.matchTemplate(search_masks[L], t, cv2.TM_CCOEFF_NORMED)
        r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
        acc = wt * r if acc is None else acc + wt * r
        wsum += wt
    if acc is None:
        raise CadAnchorUnavailable("reference CAD has no rasterizable geometry")
    acc /= wsum
    peaks, a = [], acc.copy()
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(a)), a.shape)
        peaks.append((x * SEARCH_NM_PER_PX, y * SEARCH_NM_PER_PX, float(acc[y, x])))
        a[max(0, y - 3):y + 4, max(0, x - 3):x + 4] = -np.inf
    return peaks


def exact_offset(ref_bboxes: dict, search: SearchCad, guess: tuple, ref_size: float = REF_SIZE):
    """Exact reference origin near `guess`, by matching polygon bounding boxes.

    Only reference polygons strictly inside the design window count: a polygon
    the window clipped has a different bounding box from its search-side
    original. Returns (origin, support, n_interior) where support is the share
    of interior polygons that reappear at exactly that origin.
    """
    gx, gy = guess
    offsets, n_int = [], 0
    for L, a in ref_bboxes.items():
        inside = ((a[:, 0] > 0.5) & (a[:, 1] > 0.5)
                  & (a[:, 2] < ref_size - 0.5) & (a[:, 3] < ref_size - 0.5))
        a = a[inside]
        n_int += len(a)
        b = search.bboxes.get(L)
        if b is None or not len(a):
            continue
        lo_x, hi_x = gx - 2 * REFINE_WINDOW_NM, gx + ref_size + 2 * REFINE_WINDOW_NM
        lo_y, hi_y = gy - 2 * REFINE_WINDOW_NM, gy + ref_size + 2 * REFINE_WINDOW_NM
        near = (b[:, 2] > lo_x) & (b[:, 0] < hi_x) & (b[:, 3] > lo_y) & (b[:, 1] < hi_y)
        b = b[near]
        if not len(b):
            continue
        rc = np.c_[(a[:, 0] + a[:, 2]) / 2 + gx, (a[:, 1] + a[:, 3]) / 2 + gy]
        rw = np.c_[a[:, 2] - a[:, 0], a[:, 3] - a[:, 1]]
        sc = np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]
        sw = np.c_[b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]]
        d = sc[None, :, :] - rc[:, None, :]                          # (nr, ns, 2)
        same = ((np.abs(d).max(axis=2) < REFINE_WINDOW_NM)
                & (np.abs(sw[None, :, :] - rw[:, None, :]).max(axis=2) < REFINE_SIZE_TOL_NM))
        i, j = np.nonzero(same)
        if len(i):
            offsets.append(d[i, j] + (gx, gy))
    if n_int == 0 or not offsets:
        return None, 0.0, n_int
    off = np.round(np.concatenate(offsets), 2)
    keys, counts = np.unique(off, axis=0, return_counts=True)
    best = int(np.argmax(counts))
    return (float(keys[best, 0]), float(keys[best, 1])), float(counts[best] / n_int), n_int


# Exact matching for a reference with (almost) no polygon fully inside its
# window -- e.g. a FinFET crop where every fin and gate runs off the edge: try
# every integer-nm offset this close to a coarse peak, clip the search
# polygons to that window, and count reference boxes reproduced exactly.
CLIPPED_RADIUS_NM = 12
# Clipped matching counts every reference polygon, including the rounded ones
# a clip reshapes, so it gets its own, stricter bar.
SUPPORT_FOUND_CLIPPED = 0.8


def _clipped_support(ref_keys: set, near: dict, ox: float, oy: float, ref_size: float) -> float:
    keys = set()
    for L, b in near.items():
        c = np.clip(b - (ox, oy, ox, oy), 0.0, ref_size)
        c = c[(c[:, 2] > c[:, 0]) & (c[:, 3] > c[:, 1])]
        keys.update((L, *row) for row in np.round(c, 1))
    return len(ref_keys & keys) / len(ref_keys)


def clipped_offset(ref_bboxes: dict, search: SearchCad, guess: tuple, ref_size: float = REF_SIZE,
                   radius: int = CLIPPED_RADIUS_NM, top: int = 3):
    """(origin, support, n) comparing every reference box -- clipped ones
    included -- against search boxes clipped to the candidate window.

    Each box edge that lies inside the window fixes one offset component
    exactly (a horizontal fin fixes y, a vertical gate fixes x); the edges
    vote, and the best-voted combinations are verified in full."""
    ref_keys = {(L, *np.round(b, 1)) for L, a in ref_bboxes.items() for b in a}
    if not ref_keys:
        return None, 0.0, 0
    gx, gy = float(guess[0]), float(guess[1])
    near, votes = {}, ([], [])
    for L, a in ref_bboxes.items():
        b = search.bboxes.get(L)
        if b is None:
            continue
        m = ((b[:, 2] > gx - radius - 1) & (b[:, 0] < gx + ref_size + radius + 1)
             & (b[:, 3] > gy - radius - 1) & (b[:, 1] < gy + ref_size + radius + 1))
        b = near[L] = b[m]
        if not len(b):
            continue
        for axis, g in ((0, gx), (1, gy)):
            for col in (axis, axis + 2):                  # low edge, high edge
                inner = (a[:, col] > 0.5) & (a[:, col] < ref_size - 0.5)
                if not inner.any():
                    continue
                d = b[None, :, col] - a[inner][:, None, col]        # origin candidates
                d = d[np.abs(d - g) <= radius]
                votes[axis].append(np.round(d, 2))
    cand = []
    for axis, g in ((0, gx), (1, gy)):
        v = np.concatenate(votes[axis]) if votes[axis] else np.zeros(0)
        if len(v):
            vals, cnt = np.unique(v, return_counts=True)
            cand.append([float(x) for x in vals[np.argsort(-cnt)[:top]]])
        else:
            cand.append([float(round(g))])
    best = (None, 0.0)
    for ox in cand[0]:
        for oy in cand[1]:
            hit = _clipped_support(ref_keys, near, ox, oy, ref_size)
            if hit > best[1]:
                best = ((ox, oy), hit)
    return best[0], best[1], len(ref_keys)


def locate_reference(ref_polys: dict, ref_nl: int, search: SearchCad, ref_size: float = REF_SIZE):
    """(origin_nm or None, support, n_interior, coarse_peak)."""
    nl = max(ref_nl, search.num_layers)
    side = int(round(ref_size / SEARCH_NM_PER_PX))
    ref_m = layer_masks(ref_polys, nl, (side, side), SEARCH_NM_PER_PX)
    sm = search.masks
    if sm.shape[0] < nl:
        sm = np.concatenate([sm, np.zeros((nl - sm.shape[0],) + sm.shape[1:], np.float32)])
    blur = lambda a: np.stack([cv2.GaussianBlur(c, (0, 0), COARSE_BLUR_PX) for c in a])  # noqa: E731
    peaks = coarse_cad_peaks(blur(ref_m), blur(sm))
    ref_bb = _bboxes(ref_polys)
    n_interior = sum(int(((a[:, 0] > 0.5) & (a[:, 1] > 0.5) & (a[:, 2] < ref_size - 0.5)
                          & (a[:, 3] < ref_size - 0.5)).sum()) for a in ref_bb.values())
    best = (None, 0.0, n_interior, peaks[0][2])
    if n_interior < MIN_INTERIOR_POLYGONS:
        # Clipped matching. No early stop: with long periodic lines a lattice
        # alias can reproduce all but the few non-periodic boxes, so every
        # candidate is scored and the best one wins.
        for gx, gy, pk in peaks:
            origin, support, n = clipped_offset(ref_bb, search, (gx, gy), ref_size)
            if support > best[1]:
                best = (origin, support, -n, pk)       # negative n marks clipped matching
        if best[2] >= 0:
            best = (best[0], best[1], -sum(len(a) for a in ref_bb.values()), best[3])
        return best
    for gx, gy, pk in peaks:
        origin, support, n_int = exact_offset(ref_bb, search, (gx, gy), ref_size)
        if support > best[1]:
            best = (origin, support, n_int, pk)
        if support > 0.9:
            break
    return best


# ---------------------------------------------------------------------------
# Step 2: the design -> image rotation
# ---------------------------------------------------------------------------

ANGLE_BINS = 3600              # 0.1 deg per polar row


def _polar_spectrum(img: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = img.astype(np.float32) - float(img.mean())
    x *= cv2.createHanningWindow((w, h), cv2.CV_32F)
    F = cv2.dft(x, flags=cv2.DFT_COMPLEX_OUTPUT)
    mag = np.log1p(cv2.magnitude(F[..., 0], F[..., 1]))
    mag = np.fft.fftshift(mag)
    P = cv2.warpPolar(mag, (400, ANGLE_BINS), (w / 2.0, h / 2.0), min(h, w) / 2.0,
                      cv2.WARP_POLAR_LINEAR)
    P = P[:, 20:300]            # drop DC and the noise-dominated rim
    return P - P.mean(axis=0, keepdims=True)


def coarse_rotations(img: np.ndarray, model: np.ndarray, max_deg: float, k: int = 3) -> list:
    """Up to k candidate angles (deg, cv2 sign), best spectral match first,
    at least 0.5 deg apart. The spectral peak is usually right; when the
    layout's spectrum is nearly symmetric a second peak can win, which is why
    the caller settles between candidates on the tile evidence."""
    a, b = _polar_spectrum(img), _polar_spectrum(model)
    fa = cv2.dft(np.ascontiguousarray(a.T), flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
    fb = cv2.dft(np.ascontiguousarray(b.T), flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
    prod = cv2.mulSpectrums(fa, fb, cv2.DFT_ROWS, conjB=True)
    xc = cv2.idft(prod, flags=cv2.DFT_ROWS | cv2.DFT_REAL_OUTPUT).sum(axis=0)
    step = 360.0 / ANGLE_BINS
    lags = np.fft.fftfreq(ANGLE_BINS, d=1.0 / ANGLE_BINS) * step
    xc = np.where(np.abs(lags) <= max_deg + 0.5, xc, -np.inf)
    out, work = [], xc.copy()
    guard = int(round(0.5 / step))
    for _ in range(k):
        j = int(np.argmax(work))
        if not np.isfinite(work[j]):
            break
        y0, y1, y2 = xc[j - 1], xc[j], xc[(j + 1) % ANGLE_BINS]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if np.isfinite(den) and den != 0 else 0.0
        out.append(-float(lags[j] + d * step))
        for g in range(-guard, guard + 1):
            work[(j + g) % ANGLE_BINS] = -np.inf
    return out


def coarse_rotation(img: np.ndarray, model: np.ndarray, max_deg: float) -> float:
    """Angle (deg, cv2 sign) that rotates `model` onto `img`, from the angular
    cross-correlation of their magnitude spectra (translation-invariant)."""
    a, b = _polar_spectrum(img), _polar_spectrum(model)
    # Circular cross-correlation along the angle axis, summed over radius.
    fa = cv2.dft(np.ascontiguousarray(a.T), flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
    fb = cv2.dft(np.ascontiguousarray(b.T), flags=cv2.DFT_ROWS | cv2.DFT_COMPLEX_OUTPUT)
    prod = cv2.mulSpectrums(fa, fb, cv2.DFT_ROWS, conjB=True)
    xc = cv2.idft(prod, flags=cv2.DFT_ROWS | cv2.DFT_REAL_OUTPUT).sum(axis=0)
    lags = np.fft.fftfreq(ANGLE_BINS, d=1.0 / ANGLE_BINS) * (360.0 / ANGLE_BINS)
    xc = np.where(np.abs(lags) <= max_deg + 0.5, xc, -np.inf)
    k = int(np.argmax(xc))
    y0, y1, y2 = xc[k - 1], xc[k], xc[(k + 1) % ANGLE_BINS]
    den = y0 - 2 * y1 + y2
    d = 0.5 * (y0 - y2) / den if np.isfinite(den) and den != 0 else 0.0
    return -float(lags[k] + d * (360.0 / ANGLE_BINS))


def _rotate(img: np.ndarray, deg: float, shift=(0.0, 0.0)) -> np.ndarray:
    """Rotate about the frame centre (cv2 sign), then translate by `shift` px."""
    h, w = img.shape
    M = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), deg, 1.0)
    M[:, 2] += shift
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def frame_translation(img: np.ndarray, model_rot: np.ndarray) -> tuple:
    """Whole-frame shift (px) of the image relative to the rotated model."""
    win = cv2.createHanningWindow(img.shape[::-1], cv2.CV_32F)
    (tx, ty), resp = cv2.phaseCorrelate(model_rot.astype(np.float32), img.astype(np.float32), win)
    return float(tx), float(ty), float(resp)


def _parabolic(a, b, c):
    den = a - 2 * b + c
    return 0.5 * (a - c) / den if den != 0 else 0.0


def tile_displacements(img: np.ndarray, model: np.ndarray, tile: int = TILE_PX,
                       rad: int = TILE_SEARCH_PX, min_ncc: float = TILE_MIN_NCC) -> np.ndarray:
    """(cx, cy, dx, dy, ncc) per tile: where each model tile sits in the image."""
    h, w = img.shape
    rows = []
    margin = rad + 20
    for ty in range(margin, h - tile - margin + 1, tile):
        for tx in range(margin, w - tile - margin + 1, tile):
            t = model[ty:ty + tile, tx:tx + tile]
            if float(t.std()) < 2.0:
                continue
            s = img[ty - rad:ty + tile + rad, tx - rad:tx + tile + rad]
            r = cv2.matchTemplate(s, t, cv2.TM_CCOEFF_NORMED)
            y, x = np.unravel_index(int(np.argmax(r)), r.shape)
            if not (0 < y < r.shape[0] - 1 and 0 < x < r.shape[1] - 1):
                continue
            ncc = float(r[y, x])
            if ncc < min_ncc:
                continue
            dy = y + _parabolic(r[y - 1, x], r[y, x], r[y + 1, x]) - rad
            dx = x + _parabolic(r[y, x - 1], r[y, x], r[y, x + 1]) - rad
            rows.append((tx + tile / 2.0, ty + tile / 2.0, dx, dy, ncc))
    return np.asarray(rows, np.float64).reshape(-1, 5)


def fine_rotation(img: np.ndarray, model: np.ndarray, theta0: float, iterations: int = FINE_ITERATIONS,
                  shift=(0.0, 0.0), min_ncc: float = TILE_MIN_NCC):
    """Refine the angle from the vertical displacement field.

    Raster drift moves each scan row horizontally and nothing vertically, so
    `dy` sees only the rigid part: dy = -dtheta * x + ds * y + b (small angle,
    cv2 sign, x and y centred). `ds` is a magnification residual and `b` a
    vertical translation, both expected to be ~0 for the CAD generator.
    """
    h, w = img.shape
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    theta, ds, b, keep, pts, resid = theta0, 0.0, 0.0, np.zeros(0, bool), np.zeros((0, 5)), float("nan")
    for _ in range(iterations):
        pts = tile_displacements(img, _rotate(model, theta, shift), min_ncc=min_ncc)
        if len(pts) < 6:
            raise CadAnchorUnavailable(f"only {len(pts)} tiles aligned")
        x, y, dy = pts[:, 0] - cx, pts[:, 1] - cy, pts[:, 3]
        A = np.c_[x, y, np.ones_like(x)]
        keep = np.ones(len(pts), bool)
        for _ in range(3):
            coef, *_ = np.linalg.lstsq(A[keep], dy[keep], rcond=None)
            res = np.abs(dy - A @ coef)
            keep = res < max(3.0 * float(np.median(res[keep])), 0.15)
        if keep.sum() < 6:
            raise CadAnchorUnavailable("tile displacements disagree")
        coef, *_ = np.linalg.lstsq(A[keep], dy[keep], rcond=None)
        theta -= math.degrees(coef[0])
        ds, b = float(coef[1]), float(coef[2])
        resid = float(np.median(np.abs(dy - A @ coef)[keep]))
        if abs(math.degrees(coef[0])) < 1e-4:
            break
    return theta, ds, b, pts, keep, resid


# ---------------------------------------------------------------------------
# The yield fit
# ---------------------------------------------------------------------------

def fit_greys(img: np.ndarray, vis_rot: np.ndarray, step: int = 3, border: int = 20):
    """Least-squares grey level per visible surface (background + layers),
    Huber-reweighted once; returns (greys, R^2)."""
    sl = (slice(border, -border, step), slice(border, -border, step))
    y = img[sl].ravel().astype(np.float64)
    X = vis_rot[(slice(None),) + sl].reshape(vis_rot.shape[0], -1).T.astype(np.float64)
    used = X.sum(axis=0) > 1e-3 * len(y)
    Xu = X[:, used]
    wts = np.ones_like(y)
    g = np.zeros(X.shape[1])
    for _ in range(2):
        sw = np.sqrt(wts)
        gu, *_ = np.linalg.lstsq(Xu * sw[:, None], y * sw, rcond=None)
        r = y - Xu @ gu
        s = 1.4826 * float(np.median(np.abs(r))) + 1e-6
        wts = np.minimum(1.0, 1.5 * s / (np.abs(r) + 1e-9))
    g[used] = gu
    pred = X @ g
    ss_tot = float(((y - y.mean()) ** 2).sum()) + 1e-9
    r2 = 1.0 - float(((y - pred) ** 2).sum()) / ss_tot
    return g.astype(np.float32), r2


# ---------------------------------------------------------------------------
# Label arithmetic
# ---------------------------------------------------------------------------

def design_to_search(px_nm: float, py_nm: float, theta_deg: float, image_shape: tuple,
                     magnification: float = 1.0, translation_px=(0.0, 0.0)) -> tuple:
    """The generator's own design-point -> search-pixel mapping.

    render_cad_sample rotates the full canvas about its centre with
    cv2.getRotationMatrix2D(centre, angle, 1) and labels the design point
    through that same matrix, divided by the 10x scale factor. Written in
    offsets from the centre, the padding margin cancels exactly.
    """
    h, w = image_shape
    cxn, cyn = w * SEARCH_NM_PER_PX / 2.0, h * SEARCH_NM_PER_PX / 2.0
    a, b = math.cos(math.radians(theta_deg)), math.sin(math.radians(theta_deg))
    ux, uy = (px_nm - cxn) * magnification, (py_nm - cyn) * magnification
    x = (cxn + a * ux + b * uy) / SEARCH_NM_PER_PX + translation_px[0]
    y = (cyn - b * ux + a * uy) / SEARCH_NM_PER_PX + translation_px[1]
    return x, y


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def register(ref_gds: str, search_gds: str, search_img: np.ndarray,
             rotation_bounds=(-10.0, 10.0), ref_size: float = REF_SIZE) -> CadAnchorResult:
    """Register the reference CAD against a search image through the search CAD.

    Raises CadAnchorUnavailable when this path cannot answer (no usable search
    CAD, or the frame does not align); the caller then falls back to
    image-only matching. A reference that is simply not in the search CAD is
    an answer -- found=False -- not an exception.
    """
    img = search_img.astype(np.float32)
    try:
        ref_polys, ref_nl = read_gds_layers(ref_gds)
        search = load_search_cad(search_gds, img.shape)
    except GdsError as exc:
        raise CadAnchorUnavailable(str(exc)) from exc

    origin, support, n_int, peak = locate_reference(ref_polys, ref_nl, search, ref_size)
    clipped = n_int < 0
    if clipped:
        n_int = -n_int
        if n_int < MIN_INTERIOR_POLYGONS:
            raise CadAnchorUnavailable(f"reference has only {n_int} polygons; support is not meaningful")
        bar = SUPPORT_FOUND_CLIPPED
    else:
        bar = SUPPORT_FOUND_FEW if n_int < FEW_POLYGONS else SUPPORT_FOUND
    out = CadAnchorResult(found=False, support=support, origin_nm=origin, coarse_peak=peak, n_interior=n_int)
    if origin is None or support < bar:
        out.score = float(ABSENT_BAND * min(support, 1.0))
        out.reason = "reference not in search CAD"
        return out

    # Frame alignment, against the design rendered with the default yield model.
    vis = visible_fractions(search.masks)
    model = cv2.GaussianBlur(compose(vis, default_greys(search.num_layers)), (0, 0), 0.6)
    max_deg = max(abs(rotation_bounds[0]), abs(rotation_bounds[1]))
    # Candidate angles from the spectrum; each is refined and the one whose
    # tiles agree best wins (a wrong coarse angle leaves only the central
    # tiles inside their search window, so it cannot fake a consensus).
    best = None
    n_tiles = len(range(TILE_SEARCH_PX + 20, img.shape[0] - TILE_PX - TILE_SEARCH_PX - 20 + 1, TILE_PX)) *         len(range(TILE_SEARCH_PX + 20, img.shape[1] - TILE_PX - TILE_SEARCH_PX - 20 + 1, TILE_PX))
    cands = coarse_rotations(img, model, max_deg)
    for floor in TILE_NCC_FLOORS:
        for th0 in cands:
            # The organizer's export puts design and image in one frame (rotation
            # about the centre, no offset). If a search CAD arrives in another
            # frame, the offset is large and unambiguous over the whole field.
            tx, ty, _ = frame_translation(img, _rotate(model, th0))
            shift = (tx, ty) if math.hypot(tx, ty) > TRANSLATION_BELIEVE_PX else (0.0, 0.0)
            try:
                fit = fine_rotation(img, model, th0, shift=shift, min_ncc=floor)
            except CadAnchorUnavailable:
                continue
            kept, resid = int(fit[4].sum()), float(fit[5])
            # Many tiles agreeing tightly is the evidence. A wrong candidate can
            # still get a small, internally consistent set of tiles (the
            # organizer's curated sample 14 did: 32 tiles at 1.37 px residual
            # against the right angle's 72 at 0.10 px), so both count and
            # residual decide, and a loose fit ranks last.
            quality = kept * float(np.median(fit[3][fit[4], 4])) * (1.0 if resid < 0.5 else 1e-3)
            if best is None or quality > best[0]:
                best = (quality, th0, shift, fit)
            if kept >= 0.6 * n_tiles and resid < 0.25:
                break                       # a clear consensus: stop early
        if best is not None and int(best[3][4].sum()) >= MIN_TILE_SHARE * n_tiles:
            break                           # this floor answered; do not go lower
    # Either no candidate fitted, or the winner agrees on too few tiles to be
    # a frame fit. The design question is nonetheless already answered
    # exactly: `origin` came from polygon bounding boxes, which no amount of
    # beam noise can move. Report the anchor with the best coarse angle and a
    # score that says the pose is unverified, rather than discarding an exact
    # location and falling back to a matcher with strictly less to work with.
    if best is None or int(best[3][4].sum()) < MIN_TILE_SHARE * n_tiles:
        if not cands:
            raise CadAnchorUnavailable("no rotation candidate aligned the frame")
        theta = float(cands[0]) if best is None else float(best[1])
        px, py = origin[0] + ref_size / 2.0, origin[1] + ref_size / 2.0
        x, y = design_to_search(px, py, theta, img.shape)
        lo, hi = rotation_bounds
        out.found = True
        out.x, out.y = float(x), float(y)
        out.theta = float(np.clip(theta, lo, hi))
        out.scale = float(SEARCH_NM_PER_PX)
        out.theta_coarse = theta
        out.tiles_total = n_tiles
        out.tiles_used = 0 if best is None else int(best[3][4].sum())
        lo_b, span_b = UNVERIFIED_BAND
        out.score = float(lo_b + span_b * min(support, 1.0))
        out.reason = "cad-anchored (pose unverified)"
        return out
    _, th0, shift, (theta, ds, b, pts, keep, resid) = best
    out.theta_coarse = th0

    # Yield fit on the aligned frame, then one more pass on the fitted render.
    vis_rot = np.stack([cv2.GaussianBlur(_rotate(v, theta, shift), (0, 0), 0.6) for v in vis])
    greys, r2 = fit_greys(img, vis_rot)
    fitted = cv2.GaussianBlur(compose(vis, greys), (0, 0), 0.6)
    try:
        theta, ds, b, pts, keep, resid = fine_rotation(img, fitted, theta, iterations=1, shift=shift)
    except CadAnchorUnavailable:
        pass

    mag = 1.0 + ds if abs(ds) > SCALE_BELIEVE else 1.0
    px, py = origin[0] + ref_size / 2.0, origin[1] + ref_size / 2.0
    if shift != (0.0, 0.0):
        shift = (shift[0], shift[1] + b)          # the vertical residual is drift-free
    x, y = design_to_search(px, py, theta, img.shape, magnification=mag, translation_px=shift)
    lo, hi = rotation_bounds
    out.found = True
    out.x, out.y = float(x), float(y)
    out.theta = float(np.clip(theta, lo, hi))
    out.scale = float(SEARCH_NM_PER_PX / mag)
    out.magnification = mag
    out.translation_px = (float(shift[0]), float(shift[1]))
    out.tiles_used, out.tiles_total = int(keep.sum()), int(len(pts))
    out.tile_ncc = float(np.median(pts[keep, 4])) if keep.any() else 0.0
    out.tile_resid_px = resid
    out.yield_r2 = float(r2)
    out.greys = {("background" if i == 0 else i - 1): float(g) for i, g in enumerate(greys)}
    # Calibration: "AUC of your score column, and whether you knew your yield
    # fit was wrong". support is the design evidence, tile_ncc the frame
    # agreement, and r2 the yield fit itself -- half-weighted, so a poor fit
    # discounts a pair without ever dropping it into the unverified band.
    fit_q = float(np.clip(r2, 0.0, 1.0)) if np.isfinite(r2) else 0.0
    quality = min(support, 1.0) * np.clip(out.tile_ncc, 0.0, 1.0) * (0.5 + 0.5 * fit_q)
    lo_b, span_b = VERIFIED_BAND
    out.score = float(lo_b + span_b * quality)
    out.reason = "cad-anchored"
    return out

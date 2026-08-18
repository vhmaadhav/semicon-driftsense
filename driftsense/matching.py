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
import numpy as np
import torch
import torch.nn.functional as F

from driftsense.model import SCALE, STRIDE

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


def make_template(reference: np.ndarray) -> np.ndarray:
    """Reference (1 nm/px) -> template at the search frame's 10 nm/px.

    INTER_AREA because that is the same area-average the physical search
    image underwent; bilinear here measurably softens the match.
    """
    th = max(int(round(reference.shape[0] / SCALE)), 1)
    tw = max(int(round(reference.shape[1] / SCALE)), 1)
    return cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)


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

    Searches +/- `radius` px around the coarse box position and fits a
    parabola through the correlation peak for sub-pixel placement.
    """
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


def zncc_only(reference: np.ndarray, search: np.ndarray) -> dict:
    """Classical multi-scale ZNCC. Used as the fallback path when no trained
    weights are available, so the inference script always returns a result."""
    best = None
    for scale in (9.0, 9.5, 10.0, 10.5, 11.0):
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


@torch.no_grad()
def locate(model, reference: np.ndarray, search: np.ndarray, device,
           refine: bool = True, return_heatmap: bool = False,
           tie_tol: float = TIE_REL_TOL, refine_radius: int = REFINE_RADIUS,
           refine_accept_px: float = 10.0) -> dict:
    """Full inference: reference + search (uint8 grayscale) -> centre (x, y)."""
    model.eval()

    template = make_template(reference)
    th, tw = template.shape

    tpl_n = standardize(template / 255.0)
    sea_n = standardize(search / 255.0)

    t = torch.from_numpy(tpl_n)[None, None].to(device)
    s = torch.from_numpy(pad_to_stride(sea_n))[None, None].to(device)

    out = model(t, s)
    prob = torch.sigmoid(out["logit"])[0, 0].float().cpu().numpy()
    offs = out["offset"][0].float().cpu().numpy()

    i, j, score = select_peak(prob, search.shape, (th, tw), tie_tol)
    cx, cy = response_to_center(i, j, th, tw,
                                float(offs[1, i, j]), float(offs[0, i, j]))
    result = {"x": cx, "y": cy, "score": float(score), "coarse": (cx, cy)}

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


def _dihedral_point_inv(x: float, y: float, size: int, t: int) -> tuple[float, float]:
    """Map a point in the transformed frame back to original coordinates.

    Forward is `k` counter-clockwise rot90s then an optional fliplr, so the
    inverse undoes the flip first and then applies the inverse rotation
    (x, y) -> (size-1-y, x), k times.
    """
    k, flip = t % 4, t // 4
    if flip:
        x = size - 1 - x
    for _ in range(k):
        x, y = size - 1 - y, x
    return x, y


@torch.no_grad()
def locate_tta(model, reference: np.ndarray, search: np.ndarray, device,
               transforms=range(8), cluster_px: float = 6.0,
               refine: bool = True, verify_alpha: float = VERIFY_ALPHA,
               verify_top_k: int = VERIFY_TOP_K) -> dict:
    """Dihedral test-time augmentation with cluster voting.

    Each view proposes a centre; proposals are mapped back to the original
    frame and grouped. The group with the greatest total confidence wins, and
    its members are averaged (confidence-weighted). Voting on *agreement*
    rather than averaging heatmaps is what makes this robust: a wrong-repeat
    proposal is an outlier one period away, and averaging it in would drag the
    answer off, while clustering discards it.
    """
    size = search.shape[0]
    props = []
    for t in transforms:
        r_t = _dihedral_img(reference, t)
        s_t = _dihedral_img(search, t)
        res = locate(model, r_t, s_t, device, refine=False)
        x, y = _dihedral_point_inv(res["x"], res["y"], size, t)
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

    tpl_n = standardize(make_template(reference) / 255.0)
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

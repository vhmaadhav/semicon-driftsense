#!/usr/bin/env python3
"""Phase 3 representation A/B on the 200-sample CAD dataset.

Compares four ways of getting from a Reference .gds to something matchable
against the Search SEM image:

  A. intensity        -- rasterize the GDS with the yield model, ZNCC directly
                         (exactly what baseline_solution/cad_infer.py does)
  B. edges            -- Sobel gradient magnitude on both sides, ZNCC
  C. cad2sem          -- rasterize -> simulate an SEM capture of the design
                         (PSF blur + edge brightening + shot/detector noise),
                         then ZNCC. "First convert it to the SEM image."
  D. cad2sem+edges    -- C followed by the edge transform

Design notes, so this runs ONCE and cheaply on a 16 GB laptop:

  * GDS rendering is the expensive part (1 s/sample). Every representation is
    derived from ONE render per sample, and the renders are cached to an .npz
    unless --no-cache. Re-running is then seconds, not minutes.
  * No rotation search: the generator's `search_rotation_deg` defaults to 0.0,
    so pose is location + scale only. Scale is swept as in the shipped Phase 2
    baseline (9.0..11.0). This is a representation comparison, not a pose hunt.
  * Peak RSS is one 1000x1000 float32 stack at a time.

Scoring follows the published rubric's localisation tiers, and reports the
statistic that actually decides the score column: the separation between
true-match peaks and the best competing peak (this is what AUC measures).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import gdstk
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.cad import yield_model as ym
from src.cad.render import rasterize_cell
from src import sem_imaging

ROOT_DEFAULT = "output/phase3_200"
SCALES = (9.0, 9.5, 10.0, 10.5, 11.0)
REF_SIZE = 1000


# ---------------------------------------------------------------- rendering

def render_gds(path: str, size: int = REF_SIZE) -> np.ndarray:
    """Rasterize a reference .gds with the yield model -- arm A's input."""
    lib = gdstk.read_gds(path)
    cells = lib.top_level()
    if not cells:
        raise ValueError(f"{path}: no cells")
    cell = cells[0]
    polys = cell.get_polygons()
    if not polys:
        raise ValueError(f"{path}: no polygons")
    num_layers = max(p.layer for p in polys) + 1
    return rasterize_cell(cell, size, num_layers)


def edges(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def edge_brighten(img: np.ndarray, strength: float) -> np.ndarray:
    """Our repo's forward model (driftsense/generate.py:281), reused verbatim.

    Kept identical so the CAD->SEM arm is not inventing new physics; the
    generator that produced the Search images is the same one that documented
    this effect.
    """
    if strength <= 0.0:
        return img
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    peak = float(mag.max())
    if peak < 1e-6:
        return img
    return np.clip(f + strength * 255.0 * (mag / peak), 0.0, 255.0).astype(np.uint8)


def cad_to_sem(raster: np.ndarray, seed: int, edge_strength: float,
               spot_nm: float = 5.0, dose: float = 200.0,
               noise_sigma: float = 5.0, gamma: float = 1.0) -> np.ndarray:
    """Turn a clean design raster into something SEM-like.

    Mirrors the generator's own forward chain (beam PSF -> edge brightening ->
    detector/shot noise -> gamma) but applied to the DESIGN raster instead of
    the fabricated canvas, which is the whole point of the "convert it to an
    SEM image" arm: recover the shared appearance, then match in it.
    """
    rng = np.random.default_rng(seed)
    img = sem_imaging.gaussian_psf_blur(raster.astype(np.float32), spot_nm, 1.0)
    img = edge_brighten(img, edge_strength)
    img = sem_imaging.add_shot_noise(img, dose, rng)
    img = sem_imaging.add_detector_noise(img, noise_sigma, rng)
    if gamma != 1.0:
        img = sem_imaging.apply_gamma(img, gamma)
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- matching

def zncc_peak(search: np.ndarray, tmpl: np.ndarray):
    """Best ZNCC over the scale sweep. Returns (score, x, y, second_score).

    `second_score` is the best peak outside a suppression radius around the
    winner -- the quantity a score-threshold / AUC actually depends on.
    """
    best = None
    radius = max(int(tmpl.shape[0] / SCALES[-1]), 10)
    for s in SCALES:
        tw = max(int(round(tmpl.shape[1] / s)), 1)
        th = max(int(round(tmpl.shape[0] / s)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        t = cv2.resize(tmpl, (tw, th), interpolation=cv2.INTER_AREA)
        if float(t.std()) < 1e-6:
            continue
        res = cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED)
        _, sc, _, loc = cv2.minMaxLoc(res)
        cx, cy = loc[0] + tw / 2.0, loc[1] + th / 2.0
        if best is None or sc > best[0]:
            # suppress around the winner and take the runner-up
            lx, ly = int(loc[0]), int(loc[1])
            x0, y0 = max(lx - radius, 0), max(ly - radius, 0)
            x1, y1 = min(lx + radius, res.shape[1]), min(ly + radius, res.shape[0])
            r2 = res.copy()
            r2[y0:y1, x0:x1] = -1.0
            second = float(r2.max()) if r2.size else -1.0
            best = (float(sc), cx, cy, second)
    return best or (0.0, 500.0, 500.0, 0.0)


# ---------------------------------------------------------------- driver

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT_DEFAULT)
    ap.add_argument("--limit", type=int, default=0, help="0 = all present pairs")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--edge-strength", type=float, default=0.30,
                    help="0.15-0.35 is the documented realistic band")
    args = ap.parse_args()

    cache = os.path.join(args.root, "cad_raster_cache.npz")
    rows = list(csv.DictReader(open(os.path.join(args.root, "manifest.csv"))))
    present = [r for r in rows if r["match_found"] == "True"]
    if args.limit:
        present = present[:args.limit]

    # ---- cache every reference render exactly once
    rasters = {}
    if os.path.exists(cache) and not args.no_cache:
        z = np.load(cache)
        rasters = {k: z[k] for k in z.files}
        print(f"[cache] loaded {len(rasters)} renders from {cache}")
    todo = [r for r in present if r["id"] not in rasters]
    if todo:
        print(f"[render] {len(todo)} GDS -> raster (one-time, ~1 s each)")
        t0 = time.time()
        for n, r in enumerate(todo, 1):
            rasters[r["id"]] = render_gds(os.path.join(args.root, r["reference_gds_path"]))
            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                print(f"  {n}/{len(todo)}  {el:.0f}s elapsed, ~{el/n*(len(todo)-n):.0f}s left")
        np.savez_compressed(cache, **rasters)
        print(f"[cache] wrote {cache}")

    # ---- evaluate
    ARMS = ("A_intensity", "B_edges", "C_cad2sem", "D_cad2sem_edges")
    stats = {a: {"err": [], "score": [], "margin": []} for a in ARMS}

    for r in present:
        ref = rasters[r["id"]]
        sea = cv2.imread(os.path.join(args.root, r["search_path"]), cv2.IMREAD_GRAYSCALE)
        if sea is None:
            continue
        gx, gy = float(r["gt_x"]), float(r["gt_y"])

        sem = cad_to_sem(ref, seed=int(r["id"]), edge_strength=args.edge_strength)
        inputs = {
            "A_intensity":    (ref, sea),
            "B_edges":        (edges(ref), edges(sea)),
            "C_cad2sem":      (sem, sea),
            "D_cad2sem_edges": (edges(sem), edges(sea)),
        }
        for arm, (tmpl, srch) in inputs.items():
            sc, x, y, second = zncc_peak(srch, tmpl)
            stats[arm]["err"].append(float(np.hypot(x - gx, y - gy)))
            stats[arm]["score"].append(sc)
            stats[arm]["margin"].append(sc - second)

    # ---- report
    print(f"\nn = {len(stats['A_intensity']['err'])} present pairs, "
          f"edge_strength = {args.edge_strength}\n")
    hdr = (f"{'arm':>16} {'mean err':>9} {'med err':>8} "
           f"{'<=1px':>6} {'<=2px':>6} {'<=3px':>6} {'<=5px':>6} "
           f"{'med score':>10} {'med margin':>11}")
    print(hdr)
    print("-" * len(hdr))
    for a in ARMS:
        e = np.array(stats[a]["err"])
        print(f"{a:>16} {e.mean():>9.2f} {np.median(e):>8.2f} "
              f"{100*(e<=1).mean():>5.0f}% {100*(e<=2).mean():>5.0f}% "
              f"{100*(e<=3).mean():>5.0f}% {100*(e<=5).mean():>5.0f}% "
              f"{np.median(stats[a]['score']):>10.4f} "
              f"{np.median(stats[a]['margin']):>11.4f}")

    print("\nNOTE: 'margin' = best-peak minus best-competing-peak. The 15-pt "
          "rejection F1 and 10-pt calibration AUC both depend on it, so a low "
          "score that is well-separated beats a high score that is not.")

    np.savez_compressed(os.path.join(args.root, "ab_results.npz"),
                        **{f"{a}_{k}": np.array(v) for a, d in stats.items()
                           for k, v in d.items()})
    print(f"[saved] {os.path.join(args.root, 'ab_results.npz')}")


if __name__ == "__main__":
    main()

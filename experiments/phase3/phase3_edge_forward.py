#!/usr/bin/env python3
"""The decisive test: does the deck's premise hold when the SEM actually
brightens edges?

Why this file exists
--------------------
`src/sem_imaging.py` in the mentor generator has NO edge-brightening term:

    gaussian_psf_blur -> downsample -> drift -> barrel -> shot -> detector
    -> speckle -> salt&pepper -> vignette -> gamma -> charging_streaks

but the deck says, verbatim: *"The design has no shading, and the SEM brightens
edges it knows nothing about. Edges are what both sides agree on."*

So the earlier A/B -- which found edge matching losing -- was run on data that
does not contain the phenomenon edge matching is designed to exploit. That is a
confound, and it means the earlier conclusion was stated too strongly.

This script removes the confound. It regenerates the SEARCH images with edge
brightening added to the forward model at the physically correct place (on the
specimen surface, BEFORE beam blur and before the 10x downsample, so the effect
is imaged the way a real edge-brightened capture would be), then re-runs the
arm comparison on:

  * `off`  -- the dataset as the mentor generator produces it (edge=0.0)
  * `on`   -- edge brightening at strengths the repo documents as realistic
              (`--edge-brightening` help text: "0.15-0.35 is a visible,
              realistic range")

If edges win under `on`, the deck is right and the earlier result was an
artifact of the generator's missing physics. If edges lose under both, the
conclusion holds on real evidence.

Cost control: the 1000x1000 float Sobel on the 10000x10000 fine canvas is the
only expensive step, so the fine canvas is rendered ONCE per sample and reused
for every edge strength. Peak memory stays ~1 canvas at a time.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phase3_ab import edges, zncc_peak  # noqa: E402
from src import sem_imaging
from src.cad.cad_zones import rasterize_zone_canvas
from src.cad_pipeline import (
    FINE_CANVAS_SIZE_PX,
    PIXEL_SIZE_REF_NM,
    PIXEL_SIZE_SEARCH_NM,
    CadGenerationParams,
    build_cad_geometry,
)

ARMS = ("A_intensity", "B_edges")
STRENGTHS = (0.0, 0.20, 0.35)


def edge_brighten(img: np.ndarray, strength: float) -> np.ndarray:
    """Our repo's forward model (driftsense/generate.py:281), memory-bounded.

    The shipped version allocates img/gx/gy/mag/f simultaneously in float32.
    On the 10000x10000 fine canvas that is ~2 GB peak, which is not acceptable
    on a 16 GB laptop. Here the gradient magnitude is accumulated in row
    blocks, so peak extra allocation is O(block) rather than O(image), and the
    result is numerically identical (same Sobel 3x3 kernel, same peak scaling).
    """
    if strength <= 0.0:
        return img
    f = img.astype(np.float32)
    # Row-block the gradient so we never hold five full-canvas float32 arrays.
    block = max(1, min(f.shape[0], int(2e7 // max(f.shape[1], 1))))
    bands = []
    for y0 in range(0, f.shape[0], block):
        y1 = min(y0 + block, f.shape[0])
        # Pad by 1 so the 3x3 Sobel sees real neighbours across block seams.
        py0, py1 = max(y0 - 1, 0), min(y1 + 1, f.shape[0])
        sub = f[py0:py1]
        gx = cv2.Sobel(sub, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(sub, cv2.CV_32F, 0, 1, ksize=3)
        # np.hypot, not cv2.magnitude: the shipped function uses np.hypot, and
        # the two differ in the last float bits (measured max 1.8e-7 of peak),
        # which can flip a grey level by 1. Match the shipped arithmetic.
        m = np.hypot(gx, gy)
        del gx, gy
        bands.append(m[y0 - py0:y1 - py0])
        del m, sub
    mag = np.concatenate(bands, axis=0)
    del bands
    peak = float(mag.max())
    if peak < 1e-6:
        return img
    out = np.clip(f + strength * 255.0 * (mag / peak), 0.0, 255.0).astype(np.uint8)
    del f, mag
    return out


def search_with_edges(fine_canvas: np.ndarray, params, seed: int,
                      strength: float) -> np.ndarray:
    """The generator's own search chain, with edge brightening inserted at the
    specimen surface (before blur, before downsample)."""
    rng = np.random.default_rng(seed)
    canvas = edge_brighten(fine_canvas, strength) if strength > 0 else fine_canvas
    return sem_imaging.image_search(
        canvas,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_search,
        rng=rng,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
        astigmatism_ratio=params.astigmatism_ratio,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=80,
                    help="samples to regenerate (fine canvas is 10000x10000)")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--out", default="output/edge_forward.npz")
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out) and not args.rebuild:
        z = np.load(args.out, allow_pickle=True)
        data = {k: z[k] for k in z.files}
        print(f"[cache] loaded {args.out}")
    else:
        params = CadGenerationParams()
        rng = np.random.default_rng(args.seed)
        data = {"gt": [], "gds": [], **{f"sea_{s}": [] for s in STRENGTHS},
                **{f"ref_{s}": [] for s in STRENGTHS}}
        t0 = time.perf_counter()
        for i in range(args.limit):
            kind = "dram" if i % 2 == 0 else "finfet"
            geom = build_cad_geometry(kind, rng, params)
            if not geom["match_found"]:
                continue
            # ONE fine canvas render, reused for every edge strength
            fine = rasterize_zone_canvas(
                FINE_CANVAS_SIZE_PX, geom["mats"], geom["strip_rects"],
                np.random.default_rng(geom["strip_rng_seed"]), None,
                min_layer=0,
            )
            from src.cad.render import rasterize_cell
            for s in STRENGTHS:
                sea = search_with_edges(fine, params, geom["strip_rng_seed"] + 1 + int(s * 100), s)
                data[f"sea_{s}"].append(sea)
                # Reference: the DESIGN raster, edge-brightened the same way,
                # so both sides share the appearance (that is the deck's claim).
                ref = rasterize_cell(geom["reference_cell"], 1000, geom["num_layers"])
                data[f"ref_{s}"].append(edge_brighten(ref, s) if s > 0 else ref)
            data["gt"].append((geom["gt_x"], geom["gt_y"]))
            data["gds"].append(kind)
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{args.limit}  {time.perf_counter()-t0:.0f}s")
        data = {k: np.array(v, dtype=object) if k in ("gt", "gds") else np.array(v)
                for k, v in data.items()}
        np.savez_compressed(args.out, **data)
        print(f"[saved] {args.out}")

    n = len(data["gt"])
    print(f"\nn = {n} present pairs\n")
    print("=" * 78)
    print("Does edge-based matching win once the SEM actually brightens edges?")
    print("=" * 78)
    hdr = f"{'edge strength':>14} {'arm':>13} {'<=1px':>7} {'<=2px':>7} {'med err':>9} {'med score':>10}"
    print(hdr); print("-" * len(hdr))
    for s in STRENGTHS:
        sea = data[f"sea_{s}"]
        ref = data[f"ref_{s}"]
        errs = {"A_intensity": [], "B_edges": []}
        scs = {"A_intensity": [], "B_edges": []}
        for i in range(n):
            gx, gy = data["gt"][i]
            if s == 0.0:
                # reference handed to the matcher is the plain design raster
                tA, tB = ref[i], edges(ref[i])
            else:
                tA, tB = ref[i], edges(ref[i])
            sA, xA, yA, _ = zncc_peak(sea[i], tA)
            sB, xB, yB, _ = zncc_peak(edges(sea[i]), tB)
            errs["A_intensity"].append(np.hypot(xA - gx, yA - gy))
            errs["B_edges"].append(np.hypot(xB - gx, yB - gy))
            scs["A_intensity"].append(sA); scs["B_edges"].append(sB)
        for a in ARMS:
            e = np.array(errs[a])
            print(f"{s:>14.2f} {a:>13} {100*(e<=1).mean():>6.0f}% {100*(e<=2).mean():>6.0f}% "
                  f"{np.median(e):>9.2f} {np.median(scs[a]):>10.4f}")
        print()


if __name__ == "__main__":
    main()

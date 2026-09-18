#!/usr/bin/env python3
"""Phase 3 CAD dataset with per-sample pose and severity variation.

The upstream CLI (generate_cad_dataset.py) applies ONE parameter set to every
sample: search rotation 0, every SEM artefact off. The Phase 3 brief promises
chuck rotation on the search side, and the upstream GUI (app.py) exposes it
together with the artefacts. This wrapper draws a fresh CadGenerationParams per
sample from the GUI's own "Randomize" bands (app.py, _randomize_cad_params),
then calls the upstream pipeline unmodified.

Two things the upstream CLI does not record, and this does:

  gt_theta  the rotation actually applied. The pipeline draws it from
            np.random.default_rng(strip_rng_seed + 2).uniform(-r, r) and never
            returns it, so it is recomputed here from the same seed --
            exactly, not approximately (tests pin this against gt_x/gt_y).
  gt_scale  always 10: the CAD pipeline has no scale knob (search 10 nm/px,
            reference 1 nm/px, SCALE_FACTOR fixed).

The rotation convention is cv2.getRotationMatrix2D's: positive = counter-
clockwise on screen, the same convention driftsense.matching uses for theta.

Per-layer grey levels and the cell structure (mat size, strip width, contact
shape) are varied too: the Phase 3 brief says brightness is "yours to infer,
per layer", and the GUI exposes both as sliders.

Writes, per sample, what the organizer's own bulk export (app.py,
_build_bookmarks_zip) writes: reference/<id>.gds, search/<id>.gds (the whole
search-side CAD, app.py's _full_canvas_cell), search/<id>.png, the reference
preview, and params/<id>.json. manifest.csv carries the per-sample parameters
as extra columns, so generate_parallel.py can shard and merge it like the
upstream CLIs.

Example:
    python generate_cad_varied.py --num-samples 20 --output-dir ./output --split cad_varied --seed 7
"""

import argparse
import csv
import json
import os

import cv2
import gdstk
import numpy as np

import src.cad_pipeline as _cad_pipeline
from cad_export import applied_rotation_deg, full_canvas_cell  # noqa: F401
from src.cad import cad_zones as _cad_zones
from src.cad import yield_model
from src.cad_pipeline import (
    MAX_SEARCH_ROTATION_DEG, SCALE_FACTOR, CadGenerationParams, build_cad_geometry, render_cad_sample,
)


def _safe_fab_distortion(design_cell, num_layers, rng, polygon_scale_prob=0.0, polygon_scale_range=0.0,
                         linewidth_bias_nm=0.0, corner_rounding_px=0.0):
    """src/cad/fab_distortion.py's apply_fab_distortion, with one guard.

    gdstk's Polygon.fillet segfaults on some offset polygons (Windows access
    violation; reproducible with --seed 30004, sample 21: DRAM, linewidth
    bias -0.68 nm, corner rounding 2.79 px -- ordinary values inside the
    GUI's Randomize bands). A segfault cannot be caught, so corner rounding
    is done as a morphological opening instead. Scaling and linewidth bias
    are exactly upstream's, with the same RNG draws in the same order.
    """
    out = gdstk.Cell(f"{design_cell.name}_FAB")
    for layer in range(num_layers):
        polygons = design_cell.get_polygons(layer=layer, datatype=0)
        if not polygons:
            continue
        if polygon_scale_prob > 0 and polygon_scale_range > 0:
            for poly in polygons:
                if rng.random() < polygon_scale_prob:
                    factor = 1.0 + rng.uniform(-polygon_scale_range, polygon_scale_range)
                    (xmin, ymin), (xmax, ymax) = poly.bounding_box()
                    poly.scale(factor, center=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))
        if abs(linewidth_bias_nm) >= 1e-9 and polygons:
            polygons = gdstk.offset(polygons, linewidth_bias_nm / 2.0, layer=layer, datatype=0)
        if corner_rounding_px >= 0.5 and polygons:
            # Morphological opening (shrink by r, grow back by r with round
            # joins) instead of Polygon.fillet: same radius on convex corners,
            # computed by Clipper, which does not crash.
            r = float(corner_rounding_px)
            shrunk = gdstk.offset(polygons, -r, layer=layer, datatype=0)
            rounded = gdstk.offset(shrunk, r, join="round", layer=layer, datatype=0) if shrunk else []
            polygons = rounded or polygons
        for poly in polygons:
            out.add(poly)
    return out


# Both upstream importers bound the name at import time; rebind both.
_cad_zones.apply_fab_distortion = _safe_fab_distortion
_cad_pipeline.apply_fab_distortion = _safe_fab_distortion

ARCHITECTURE_KINDS = ["dram", "finfet"]

# app.py's Randomize bands: "a moderate sub-band of each slider's full range",
# chosen upstream so every draw stays a plausible capture. Rotation is the
# magnitude bound r; the pipeline then draws the angle uniformly in [-r, r].
GUI_BANDS = {
    "dose_search": (120.0, 900.0),
    "beam_spot_size_nm": (2.0, 9.0),
    "shear_amplitude_px": (0.0, 2.5),
    "drift_jitter_px": (0.0, 1.2),
    "detector_noise_sigma_search": (0.0, 7.0),
    "astigmatism_ratio": (0.8, 1.6),
    "search_rotation_deg": (0.0, 8.0),
    # Off: the organizer stated the Phase 3 evaluation sets carry no barrel
    # distortion ("we will not have a data set which has a barrel
    # distortion"), so it is not sampled here either.
    "barrel_distortion_k": (0.0, 0.0),
    "vignette_strength": (0.0, 0.3),
    "gamma": (0.7, 1.5),
    "charging_streak_prob": (0.0, 1.5),
    "charging_streak_intensity": (0.0, 1.0),
    "speckle_sigma": (0.0, 0.3),
    "salt_pepper_prob": (0.0, 0.015),
    "polygon_scale_prob": (0.0, 0.4),
    "polygon_scale_range": (0.0, 0.25),
    "linewidth_bias_nm": (-8.0, 8.0),
    "corner_rounding_px": (0.0, 8.0),
}
# Structure: moderate sub-bands of the GUI sliders (the GUI's Randomize leaves
# these at their defaults; the brief's "more layers, denser patterns" makes
# them worth covering). Contact angle is a GUI selectbox.
STRUCTURE_BANDS = {
    "mat_size_nm": (1800.0, 3400.0),
    "strip_width_nm": (200.0, 480.0),
    "contact_aspect_ratio": (1.0, 2.2),
    "contact_thickness_factor": (0.7, 1.5),
}
CONTACT_ANGLES = (0.0, 45.0, 90.0)
# Per-layer grey level: the yield-model default times a factor in this band.
# The organizer described nudging a layer "from 51 to 40" while keeping the
# layers in proportion -- about +/-20%.
INTENSITY_JITTER = (0.8, 1.2)
VARIED_FIELDS = list(GUI_BANDS) + list(STRUCTURE_BANDS) + ["contact_angle_deg"]


def draw_params(rng: np.random.Generator, max_rotation_deg: float, no_match_prob: float,
                vary_structure: bool = True) -> CadGenerationParams:
    """One sample's parameters. Draw order is fixed (GUI_BANDS order, then
    structure), so a seed reproduces the dataset."""
    kw = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in GUI_BANDS.items()}
    kw["search_rotation_deg"] = float(rng.uniform(0.0, max_rotation_deg))
    if vary_structure:
        kw.update({k: float(rng.uniform(lo, hi)) for k, (lo, hi) in STRUCTURE_BANDS.items()})
        kw["contact_angle_deg"] = float(CONTACT_ANGLES[int(rng.integers(0, len(CONTACT_ANGLES)))])
    return CadGenerationParams(no_match_prob=no_match_prob, **kw)


def draw_layer_intensities(rng: np.random.Generator, num_layers: int) -> dict:
    lo, hi = INTENSITY_JITTER
    return {i: int(np.clip(round(yield_model.yield_to_intensity(yield_model.layer_yield(i, num_layers))
                                 * rng.uniform(lo, hi)), 0, 255))
            for i in range(num_layers)}


def generate_sample(kind: str, rng: np.random.Generator, max_rotation_deg: float, no_match_prob: float,
                    vary_structure: bool = True, vary_intensity: bool = True) -> dict:
    params = draw_params(rng, max_rotation_deg, no_match_prob, vary_structure)
    geometry = build_cad_geometry(kind, rng, params)
    intensities = draw_layer_intensities(rng, geometry["num_layers"]) if vary_intensity else None
    sample = render_cad_sample(geometry, params, layer_intensities=intensities)
    sample["layer_intensities"] = intensities
    sample["gt_theta"] = applied_rotation_deg(geometry, params) if sample["match_found"] else 0.0
    sample["gt_scale"] = float(SCALE_FACTOR)
    sample["geometry"] = geometry
    return sample


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--architectures", nargs="+", default=ARCHITECTURE_KINDS, choices=ARCHITECTURE_KINDS)
    p.add_argument("--split", default="cad_varied")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rotation-deg", type=float, default=GUI_BANDS["search_rotation_deg"][1],
                   help="upper bound of the per-sample rotation magnitude (default: the GUI's "
                        f"Randomize band, %(default)s; upstream cap {MAX_SEARCH_ROTATION_DEG})")
    p.add_argument("--no-match-prob", type=float, default=CadGenerationParams.no_match_prob)
    p.add_argument("--fixed-structure", action="store_true", help="keep mat/strip/contact shape at defaults")
    p.add_argument("--fixed-intensity", action="store_true", help="keep per-layer grey levels at defaults")
    a = p.parse_args()
    if not 0.0 <= a.max_rotation_deg <= MAX_SEARCH_ROTATION_DEG:
        p.error(f"--max-rotation-deg must be within [0, {MAX_SEARCH_ROTATION_DEG}] (upstream cap)")
    return a


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    params_dir = os.path.join(split_dir, "params")
    for d in (ref_dir, search_dir, params_dir):
        os.makedirs(d, exist_ok=True)

    fieldnames = [
        "id", "reference_gds_path", "reference_preview_path", "search_path", "search_gds_path",
        "params_json_path",
        "match_found", "gt_x", "gt_y", "gt_theta", "gt_scale",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "architecture_kind", "num_layers", *VARIED_FIELDS, "no_match_prob", "seed",
    ]
    with open(os.path.join(split_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i in range(args.num_samples):
            kind = args.architectures[int(rng.integers(0, len(args.architectures)))]
            sample = generate_sample(kind, rng, args.max_rotation_deg, args.no_match_prob,
                                     vary_structure=not args.fixed_structure,
                                     vary_intensity=not args.fixed_intensity)
            geom = sample["geometry"]

            for cell, path in ((sample["reference_cell"], os.path.join(ref_dir, f"{i:05d}.gds")),
                               (full_canvas_cell(geom["mats"], geom["num_layers"]),
                                os.path.join(search_dir, f"{i:05d}.gds"))):
                lib = gdstk.Library()
                lib.add(cell)
                lib.write_gds(path)
            cv2.imwrite(os.path.join(ref_dir, f"{i:05d}_preview.png"), sample["reference_preview"])
            cv2.imwrite(os.path.join(search_dir, f"{i:05d}.png"), sample["search_img"])
            with open(os.path.join(params_dir, f"{i:05d}.json"), "w") as jf:
                json.dump({
                    "match_found": bool(sample["match_found"]),
                    "gt_x": sample["gt_x"], "gt_y": sample["gt_y"], "gt_box": sample["gt_box"],
                    "gt_theta": sample["gt_theta"], "gt_scale": sample["gt_scale"],
                    "architecture_kind": kind, "num_layers": sample["num_layers"],
                    "crop_origin_nm": [geom["x0"], geom["y0"]],
                    "search_layer_intensities": sample["layer_intensities"],
                    "params": sample["params"],
                }, jf, indent=2, default=float)

            if sample["match_found"]:
                gx0, gy0, gw, gh = sample["gt_box"]
                gt = {"gt_x": sample["gt_x"], "gt_y": sample["gt_y"],
                      "gt_theta": sample["gt_theta"], "gt_scale": sample["gt_scale"]}
            else:
                gx0 = gy0 = gw = gh = ""
                gt = {"gt_x": "", "gt_y": "", "gt_theta": "", "gt_scale": ""}
            writer.writerow({
                "id": i,
                "reference_gds_path": os.path.join("reference", f"{i:05d}.gds"),
                "reference_preview_path": os.path.join("reference", f"{i:05d}_preview.png"),
                "search_path": os.path.join("search", f"{i:05d}.png"),
                "search_gds_path": os.path.join("search", f"{i:05d}.gds"),
                "params_json_path": os.path.join("params", f"{i:05d}.json"),
                "match_found": sample["match_found"], **gt,
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture_kind": kind, "num_layers": sample["num_layers"],
                **sample["params"], "seed": args.seed,
            })
            state = (f"({sample['gt_x']:.1f}, {sample['gt_y']:.1f}, {sample['gt_theta']:+.2f} deg)"
                     if sample["match_found"] else "NO MATCH")
            print(f"[{i + 1}/{args.num_samples}] {kind} -> {state}")

    print(f"Wrote {args.num_samples} samples to {split_dir}")


if __name__ == "__main__":
    main()

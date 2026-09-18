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

Writes the upstream layout (reference/<id>.gds, reference/<id>_preview.png,
search/<id>.png, manifest.csv) with the per-sample parameters as extra
columns, so generate_parallel.py can shard and merge it like the upstream CLIs.

Example:
    python generate_cad_varied.py --num-samples 20 --output-dir ./output --split cad_varied --seed 7
"""

import argparse
import csv
import os

import cv2
import gdstk
import numpy as np

from src.cad_pipeline import (
    MAX_SEARCH_ROTATION_DEG, SCALE_FACTOR, CadGenerationParams, build_cad_geometry, render_cad_sample,
)

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
    "barrel_distortion_k": (-0.05, 0.05),
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
VARIED_FIELDS = list(GUI_BANDS)


def draw_params(rng: np.random.Generator, max_rotation_deg: float, no_match_prob: float) -> CadGenerationParams:
    """One sample's parameters. Draw order is fixed (GUI_BANDS order), so a
    seed reproduces the dataset."""
    kw = {k: float(rng.uniform(lo, hi)) for k, (lo, hi) in GUI_BANDS.items()}
    kw["search_rotation_deg"] = float(rng.uniform(0.0, max_rotation_deg))
    return CadGenerationParams(no_match_prob=no_match_prob, **kw)


def applied_rotation_deg(geometry: dict, params: CadGenerationParams) -> float:
    """The angle render_cad_sample applies, from the pipeline's own stream."""
    if params.search_rotation_deg <= 0:
        return 0.0
    rot_rng = np.random.default_rng(geometry["strip_rng_seed"] + 2)
    return float(rot_rng.uniform(-params.search_rotation_deg, params.search_rotation_deg))


def generate_sample(kind: str, rng: np.random.Generator, max_rotation_deg: float, no_match_prob: float) -> dict:
    params = draw_params(rng, max_rotation_deg, no_match_prob)
    geometry = build_cad_geometry(kind, rng, params)
    sample = render_cad_sample(geometry, params)
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
                        "Randomize band, %(default)s; upstream cap %s)" % MAX_SEARCH_ROTATION_DEG)
    p.add_argument("--no-match-prob", type=float, default=CadGenerationParams.no_match_prob)
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
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    fieldnames = [
        "id", "reference_gds_path", "reference_preview_path", "search_path",
        "match_found", "gt_x", "gt_y", "gt_theta", "gt_scale",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "architecture_kind", "num_layers", *VARIED_FIELDS, "no_match_prob", "seed",
    ]
    with open(os.path.join(split_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for i in range(args.num_samples):
            kind = args.architectures[int(rng.integers(0, len(args.architectures)))]
            sample = generate_sample(kind, rng, args.max_rotation_deg, args.no_match_prob)

            lib = gdstk.Library()
            lib.add(sample["reference_cell"])
            lib.write_gds(os.path.join(ref_dir, f"{i:05d}.gds"))
            cv2.imwrite(os.path.join(ref_dir, f"{i:05d}_preview.png"), sample["reference_preview"])
            cv2.imwrite(os.path.join(search_dir, f"{i:05d}.png"), sample["search_img"])

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

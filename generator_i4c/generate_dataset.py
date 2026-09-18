#!/usr/bin/env python3
"""CLI to generate a Drift-Sense synthetic dataset split.

Example:
    python generate_dataset.py --num-samples 20 --split train \
        --architectures dram_1x finfet_10nm --output-dir ./output --seed 42
"""

import argparse
import csv
import os

import cv2
import numpy as np

from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()), choices=list(PRESETS.keys()))
    p.add_argument("--split", default="train")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam-spot-size-nm", type=float, default=GenerationParams.beam_spot_size_nm)
    p.add_argument("--collapse-threshold-nm", type=float, default=GenerationParams.collapse_threshold_nm)
    p.add_argument("--dose-reference", type=float, default=GenerationParams.dose_reference)
    p.add_argument("--dose-search", type=float, default=GenerationParams.dose_search)
    p.add_argument("--shear-amplitude-px", type=float, default=GenerationParams.shear_amplitude_px)
    p.add_argument("--drift-jitter-px", type=float, default=GenerationParams.drift_jitter_px)
    p.add_argument("--astigmatism-ratio", type=float, default=GenerationParams.astigmatism_ratio)
    p.add_argument("--vignette-strength", type=float, default=GenerationParams.vignette_strength)
    p.add_argument("--gamma", type=float, default=GenerationParams.gamma)
    p.add_argument("--barrel-distortion-k", type=float, default=GenerationParams.barrel_distortion_k)
    p.add_argument("--charging-streak-prob", type=float, default=GenerationParams.charging_streak_prob)
    p.add_argument("--charging-streak-intensity", type=float, default=GenerationParams.charging_streak_intensity)
    p.add_argument("--speckle-sigma", type=float, default=GenerationParams.speckle_sigma)
    p.add_argument("--salt-pepper-prob", type=float, default=GenerationParams.salt_pepper_prob)
    p.add_argument("--linewidth-bias-nm", type=float, default=GenerationParams.linewidth_bias_nm)
    p.add_argument("--corner-rounding-px", type=float, default=GenerationParams.corner_rounding_px)
    p.add_argument("--mat-size-nm", type=float, default=GenerationParams.mat_size_nm)
    p.add_argument("--strip-width-nm", type=float, default=GenerationParams.strip_width_nm)
    p.add_argument("--boundary-bias", type=float, default=GenerationParams.boundary_bias)
    p.add_argument("--reference-rotation-deg", type=float, default=GenerationParams.reference_rotation_deg,
                    help="max +/- degrees the captured Reference is randomly rotated by (0 disables)")
    p.add_argument("--polygon-scale-prob", type=float, default=GenerationParams.polygon_scale_prob,
                    help="probability an individual contact/line/fin/gate gets an extra outlier size scale")
    p.add_argument("--polygon-scale-range", type=float, default=GenerationParams.polygon_scale_range,
                    help="+/- fraction of that outlier scale, e.g. 0.10 = up to 10%% bigger/smaller")
    p.add_argument("--no-match-prob", type=float, default=GenerationParams.no_match_prob,
                    help="probability a sample's Reference is cropped from an unrelated canvas (no match in Search)")
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    params = GenerationParams(
        beam_spot_size_nm=args.beam_spot_size_nm,
        collapse_threshold_nm=args.collapse_threshold_nm,
        dose_reference=args.dose_reference,
        dose_search=args.dose_search,
        shear_amplitude_px=args.shear_amplitude_px,
        drift_jitter_px=args.drift_jitter_px,
        astigmatism_ratio=args.astigmatism_ratio,
        vignette_strength=args.vignette_strength,
        gamma=args.gamma,
        barrel_distortion_k=args.barrel_distortion_k,
        charging_streak_prob=args.charging_streak_prob,
        charging_streak_intensity=args.charging_streak_intensity,
        speckle_sigma=args.speckle_sigma,
        salt_pepper_prob=args.salt_pepper_prob,
        linewidth_bias_nm=args.linewidth_bias_nm,
        corner_rounding_px=args.corner_rounding_px,
        mat_size_nm=args.mat_size_nm,
        strip_width_nm=args.strip_width_nm,
        boundary_bias=args.boundary_bias,
        reference_rotation_deg=args.reference_rotation_deg,
        polygon_scale_prob=args.polygon_scale_prob,
        polygon_scale_range=args.polygon_scale_range,
        no_match_prob=args.no_match_prob,
    )

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(split_dir, "manifest.csv")
    fieldnames = [
        "id", "reference_path", "search_path", "match_found", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h", "architecture",
        "beam_spot_size_nm", "collapse_threshold_nm", "dose_reference",
        "dose_search", "shear_amplitude_px", "drift_jitter_px",
        "astigmatism_ratio", "vignette_strength", "gamma", "barrel_distortion_k",
        "charging_streak_prob", "charging_streak_intensity",
        "speckle_sigma", "salt_pepper_prob",
        "linewidth_bias_nm", "corner_rounding_px",
        "mat_size_nm", "strip_width_nm", "boundary_bias", "seed",
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i in range(args.num_samples):
            architecture = args.architectures[int(rng.integers(0, len(args.architectures)))]
            sample = generate_sample(architecture, rng, params)

            ref_path = os.path.join(ref_dir, f"{i:05d}.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")
            cv2.imwrite(ref_path, sample["reference_img"])
            cv2.imwrite(search_path, sample["search_img"])

            # Store paths relative to the manifest's own directory (split_dir),
            # not the CWD -- makes the split portable (zip it up, hand it to
            # someone else, it still resolves) and matches the convention
            # validate_submission.py expects.
            manifest_ref_path = os.path.join("reference", f"{i:05d}.png")
            manifest_search_path = os.path.join("search", f"{i:05d}.png")

            if sample["match_found"]:
                gx0, gy0, gw, gh = sample["gt_box"]
                gt_x, gt_y = sample["gt_x"], sample["gt_y"]
            else:
                gx0 = gy0 = gw = gh = gt_x = gt_y = ""

            writer.writerow({
                "id": i,
                "reference_path": manifest_ref_path,
                "search_path": manifest_search_path,
                "match_found": sample["match_found"],
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture": architecture,
                **sample["params"],
                "seed": args.seed,
            })
            gt_str = f"({sample['gt_x']:.1f}, {sample['gt_y']:.1f})" if sample["match_found"] else "NO MATCH"
            print(f"[{i + 1}/{args.num_samples}] {architecture} -> gt={gt_str}")

    print(f"Wrote {args.num_samples} samples to {split_dir}")


if __name__ == "__main__":
    main()

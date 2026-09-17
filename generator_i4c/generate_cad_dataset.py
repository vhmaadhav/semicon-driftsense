#!/usr/bin/env python3
"""Phase 3 CLI: generate a Drift-Sense dataset where the Reference is a
real multi-layer GDSII CAD file (not a rendered image) and the Search is
still a noisy SEM-imaged raster, exactly as in Phase 1/2.

Each sample writes:
  reference/<id>.gds          -- the actual CAD file (5 layers, real GDSII)
  reference/<id>_preview.png  -- a clean render of that CAD, for quick
                                  eyeballing without a GDS viewer (NOT part
                                  of the matching task -- students match
                                  against the .gds, this is just for QA)
  search/<id>.png              -- the noisy Search image

Example:
    python generate_cad_dataset.py --num-samples 20 --output-dir ./output --split cad_train --seed 7
"""

import argparse
import csv
import os

import cv2
import gdstk
import numpy as np

from src.cad_pipeline import CadGenerationParams, generate_cad_sample

ARCHITECTURE_KINDS = ["dram", "finfet"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--architectures", nargs="+", default=ARCHITECTURE_KINDS, choices=ARCHITECTURE_KINDS)
    p.add_argument("--split", default="cad_train")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--beam-spot-size-nm", type=float, default=CadGenerationParams.beam_spot_size_nm)
    p.add_argument("--dose-search", type=float, default=CadGenerationParams.dose_search)
    p.add_argument("--shear-amplitude-px", type=float, default=CadGenerationParams.shear_amplitude_px)
    p.add_argument("--drift-jitter-px", type=float, default=CadGenerationParams.drift_jitter_px)
    p.add_argument("--detector-noise-sigma-search", type=float, default=CadGenerationParams.detector_noise_sigma_search)
    p.add_argument("--mat-size-nm", type=float, default=CadGenerationParams.mat_size_nm)
    p.add_argument("--strip-width-nm", type=float, default=CadGenerationParams.strip_width_nm)
    p.add_argument("--polygon-scale-prob", type=float, default=CadGenerationParams.polygon_scale_prob)
    p.add_argument("--polygon-scale-range", type=float, default=CadGenerationParams.polygon_scale_range)
    p.add_argument("--no-match-prob", type=float, default=CadGenerationParams.no_match_prob)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    params = CadGenerationParams(
        beam_spot_size_nm=args.beam_spot_size_nm,
        dose_search=args.dose_search,
        shear_amplitude_px=args.shear_amplitude_px,
        drift_jitter_px=args.drift_jitter_px,
        detector_noise_sigma_search=args.detector_noise_sigma_search,
        mat_size_nm=args.mat_size_nm,
        strip_width_nm=args.strip_width_nm,
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
        "id", "reference_gds_path", "reference_preview_path", "search_path",
        "match_found", "gt_x", "gt_y", "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "architecture_kind", "num_layers",
        "beam_spot_size_nm", "dose_search", "shear_amplitude_px", "drift_jitter_px",
        "detector_noise_sigma_search", "mat_size_nm", "strip_width_nm",
        "polygon_scale_prob", "polygon_scale_range", "no_match_prob", "seed",
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i in range(args.num_samples):
            kind = args.architectures[int(rng.integers(0, len(args.architectures)))]
            sample = generate_cad_sample(kind, rng, params)

            gds_path = os.path.join(ref_dir, f"{i:05d}.gds")
            preview_path = os.path.join(ref_dir, f"{i:05d}_preview.png")
            search_path = os.path.join(search_dir, f"{i:05d}.png")

            lib = gdstk.Library()
            lib.add(sample["reference_cell"])
            lib.write_gds(gds_path)
            cv2.imwrite(preview_path, sample["reference_preview"])
            cv2.imwrite(search_path, sample["search_img"])

            manifest_ref_gds = os.path.join("reference", f"{i:05d}.gds")
            manifest_ref_preview = os.path.join("reference", f"{i:05d}_preview.png")
            manifest_search = os.path.join("search", f"{i:05d}.png")

            if sample["match_found"]:
                gx0, gy0, gw, gh = sample["gt_box"]
                gt_x, gt_y = sample["gt_x"], sample["gt_y"]
            else:
                gx0 = gy0 = gw = gh = gt_x = gt_y = ""

            writer.writerow({
                "id": i,
                "reference_gds_path": manifest_ref_gds,
                "reference_preview_path": manifest_ref_preview,
                "search_path": manifest_search,
                "match_found": sample["match_found"],
                "gt_x": gt_x, "gt_y": gt_y,
                "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                "architecture_kind": kind,
                "num_layers": sample["num_layers"],
                **sample["params"],
                "seed": args.seed,
            })
            gt_str = f"({sample['gt_x']:.1f}, {sample['gt_y']:.1f})" if sample["match_found"] else "NO MATCH"
            print(f"[{i + 1}/{args.num_samples}] {kind} -> gt={gt_str}")

    print(f"Wrote {args.num_samples} samples to {split_dir}")


if __name__ == "__main__":
    main()

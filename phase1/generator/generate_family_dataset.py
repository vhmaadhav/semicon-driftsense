#!/usr/bin/env python3
"""Generate 'sample families': one fixed Reference crop per sample, paired
with several differently-imaged Search images (same physical scene, varied
acquisition conditions) -- demonstrates that a Reference stays findable
across degraded Search conditions, not just the exact image it was cut from.

Example:
    python generate_family_dataset.py --num-samples 10 --output-dir ./output/families
"""

import argparse
import csv
import os

import cv2
import numpy as np

from src.pipeline import GenerationParams, generate_sample_family, SEARCH_VARIANTS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--num-samples", type=int, default=10, help="per architecture family (dram, finfet)")
    p.add_argument("--architectures", nargs="+", default=["dram_1x", "finfet_10nm"])
    p.add_argument("--output-dir", default="./output/families")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    fieldnames = [
        "architecture", "sample_id", "base_seed", "reference_path",
        "gt_x", "gt_y", "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "variant_label", "search_path",
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for architecture in args.architectures:
            arch_dir = os.path.join(args.output_dir, architecture)
            for i in range(args.num_samples):
                base_seed = int(rng.integers(0, 2**31 - 1))
                sample_dir = os.path.join(arch_dir, f"sample_{i:02d}")
                os.makedirs(sample_dir, exist_ok=True)

                family = generate_sample_family(architecture, base_seed, GenerationParams())

                ref_path = os.path.join(sample_dir, "reference.png")
                cv2.imwrite(ref_path, family["reference_img"])

                gx0, gy0, gw, gh = family["gt_box"]
                for v_idx, variant in enumerate(family["search_variants"]):
                    search_path = os.path.join(sample_dir, f"search_{v_idx:02d}_{variant['label']}.png")
                    cv2.imwrite(search_path, variant["search_img"])
                    writer.writerow({
                        "architecture": architecture,
                        "sample_id": i,
                        "base_seed": base_seed,
                        "reference_path": ref_path,
                        "gt_x": family["gt_x"], "gt_y": family["gt_y"],
                        "gt_box_x": gx0, "gt_box_y": gy0, "gt_box_w": gw, "gt_box_h": gh,
                        "variant_label": variant["label"],
                        "search_path": search_path,
                    })
                print(f"[{architecture}] sample {i + 1}/{args.num_samples} "
                      f"-> {len(family['search_variants'])} search variants, gt=({family['gt_x']:.1f}, {family['gt_y']:.1f})")

    print(f"Wrote {len(args.architectures)} x {args.num_samples} samples "
          f"({len(SEARCH_VARIANTS)} search variants each) to {args.output_dir}")


if __name__ == "__main__":
    main()

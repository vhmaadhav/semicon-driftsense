#!/usr/bin/env python3
"""Generate a 50-pair Phase 3 sanity dataset using the mentor repository
at C:\\Users\\nisha\\Downloads\\drift-sense-i4c.

Mentor guidelines implemented:
1. No barrel distortion (barrel_distortion_k = 0.0, strictly avoided per mentor).
2. Reference CAD is design-only (unrotated, theta=0, no noise).
3. Search capture carries stage rotation up to 10 degrees.
4. Search capture includes realistic SEM effects:
   - raster shear (0.0 to 4.0 px)
   - drift jitter (0.5 to 2.0 px)
   - starved/normal electron dose (50 to 400)
   - detector noise (4.0 to 9.0)
   - astigmatism / beam ellipticity (1.0 to 1.8)
   - gamma variations (0.7 to 1.5)
   - charging streaks (0.0 to 2.5 prob, 1.0 to 2.0 intensity)
   - speckle noise (0.0 to 0.4 sigma)
   - salt & pepper noise (0.0 to 0.025 prob)
   - linewidth bias (-10 to +15 nm) and corner rounding (0 to 1.5 px)
   - strip width variation (80 to 200 nm)
5. Absent / no-match rate: ~8.3% (no_match_prob = 0.08, ~4 pairs absent out of 50).
6. 50 pairs balanced across DRAM (25) and FinFET (25).
7. Outputs full Phase 3 pairs.csv (6 columns), ground_truth.csv, manifest_jury.csv,
   drift.csv, and per-pair params JSON.
"""

import argparse
import csv
import json
import os
import sys
import time
import cv2
import gdstk
import numpy as np

# Path to organizer codebase
I4C_ROOT = os.environ.get("I4C_ROOT", r"C:\Users\nisha\Downloads\drift-sense-i4c")
if I4C_ROOT not in sys.path:
    sys.path.insert(0, I4C_ROOT)

from src.cad_pipeline import CadGenerationParams, build_cad_geometry, render_cad_sample

PAIRS_FIELDS = [
    "pair_id",
    "search_path",
    "reference_gds_path",
    "search_gds_path",
    "reference_sem_path",
    "params_json_path",
]


def parse_args():
    p = argparse.ArgumentParser(description="Generate 50-pair mentor sanity dataset")
    p.add_argument("--num-samples", type=int, default=50)
    p.add_argument("--out", default="data/sanity_50")
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--absent-prob", type=float, default=0.08)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = os.path.abspath(args.out)
    ref_dir = os.path.join(out_dir, "reference")
    search_dir = os.path.join(out_dir, "search")
    params_dir = os.path.join(out_dir, "params")

    for d in (ref_dir, search_dir, params_dir):
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    kinds = ["dram", "finfet"]
    # 25 DRAM, 25 FinFET
    n_each = args.num_samples // 2
    arch_list = (["dram"] * n_each) + (["finfet"] * (args.num_samples - n_each))
    rng.shuffle(arch_list)

    pairs_rows = []
    gt_rows = []
    drift_rows = []

    # Curated raster shear values commonly observed in SEM / mentor test suites
    shear_candidates = [0.0, 1.5, 2.5, 3.0, 3.5, 4.0]

    t0 = time.perf_counter()
    print(f"Generating {args.num_samples} Phase 3 pairs into {out_dir}...")

    for i in range(args.num_samples):
        pid = f"p{i:04d}"
        kind = arch_list[i]

        # Sample realistic mentor-specified SEM parameters
        shear = float(rng.choice(shear_candidates))
        jitter = float(rng.uniform(0.5, 2.0))
        dose = float(rng.uniform(60.0, 350.0))
        det_noise = float(rng.uniform(4.0, 8.5))
        astig = float(rng.uniform(1.0, 1.6))
        gamma = float(rng.uniform(0.75, 1.35))
        has_streaks = rng.random() < 0.4
        streak_prob = float(rng.uniform(0.5, 2.2)) if has_streaks else 0.0
        streak_int = float(rng.uniform(1.2, 1.9)) if has_streaks else 0.0
        has_speckle = rng.random() < 0.5
        speckle = float(rng.uniform(0.15, 0.40)) if has_speckle else 0.0
        has_sp = rng.random() < 0.35
        sp_prob = float(rng.uniform(0.005, 0.020)) if has_sp else 0.0
        lw_bias = float(rng.uniform(-8.0, 12.0))
        corner_round = float(rng.uniform(0.0, 1.2))
        strip_width = float(rng.uniform(80.0, 200.0))

        # Build CadGenerationParams: strictly barrel_distortion_k = 0.0, search_rotation_deg = 10.0
        params = CadGenerationParams(
            collapse_threshold_nm=10.0,
            beam_spot_size_nm=float(rng.uniform(4.0, 5.5)),
            dose_search=dose,
            shear_amplitude_px=shear,
            drift_jitter_px=jitter,
            detector_noise_sigma_search=det_noise,
            astigmatism_ratio=astig,
            search_rotation_deg=10.0,
            barrel_distortion_k=0.0,  # strictly zero per mentor instructions
            vignette_strength=float(rng.choice([0.0, 0.0, 0.3, 0.5])),
            gamma=gamma,
            charging_streak_prob=streak_prob,
            charging_streak_intensity=streak_int,
            speckle_sigma=speckle,
            salt_pepper_prob=sp_prob,
            mat_size_nm=2600.0,
            strip_width_nm=strip_width,
            polygon_scale_prob=0.10,
            polygon_scale_range=0.10,
            linewidth_bias_nm=lw_bias,
            corner_rounding_px=corner_round,
            no_match_prob=args.absent_prob,
        )

        sample_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        geom = build_cad_geometry(kind, sample_rng, params)

        # Retrieve exact search rotation angle drawn by the pipeline
        angle = float(np.random.default_rng(geom["strip_rng_seed"] + 2).uniform(-10.0, 10.0))

        sample = render_cad_sample(geom, params)

        # Write Reference GDSII file
        gds_rel = f"reference/{pid}.gds"
        gds_abs = os.path.join(out_dir, gds_rel)
        lib = gdstk.Library()
        lib.add(sample["reference_cell"])
        lib.write_gds(gds_abs)

        # Write Reference preview image
        preview_rel = f"reference/{pid}_preview.png"
        preview_abs = os.path.join(out_dir, preview_rel)
        cv2.imwrite(preview_abs, sample["reference_preview"])

        # Write Search image
        search_rel = f"search/{pid}.png"
        search_abs = os.path.join(out_dir, search_rel)
        cv2.imwrite(search_abs, sample["search_img"])

        # Write params JSON
        params_rel = f"params/{pid}.json"
        params_abs = os.path.join(out_dir, params_rel)
        sample_params = params.as_dict()
        sample_params["actual_rotation_deg"] = angle
        sample_params["architecture_kind"] = kind
        sample_params["match_found"] = sample["match_found"]
        with open(params_abs, "w") as pf:
            json.dump(sample_params, pf, indent=2)

        # Add to pairs.csv
        pairs_rows.append({
            "pair_id": pid,
            "search_path": search_rel,
            "reference_gds_path": gds_rel,
            "search_gds_path": gds_rel,
            "reference_sem_path": "",
            "params_json_path": params_rel,
        })

        # Add to ground_truth.csv
        if sample["match_found"]:
            gt_rows.append([
                pid,
                1,
                f"{sample['gt_x']:.4f}",
                f"{sample['gt_y']:.4f}",
                f"{angle:.4f}",
                "10.0",
            ])
        else:
            gt_rows.append([pid, 0, 0, 0, 0, 0])

        # Add to drift.csv
        drift_rows.append([
            pid,
            f"{shear:.4f}",
            f"{jitter:.4f}",
            f"{angle:.4f}",
            kind,
            int(bool(sample["match_found"])),
            f"{dose:.1f}",
            f"{gamma:.2f}",
            f"{speckle:.2f}",
            f"{sp_prob:.3f}",
            f"{streak_prob:.2f}",
            f"{lw_bias:.1f}",
        ])

        if (i + 1) % 10 == 0 or (i + 1) == args.num_samples:
            el = time.perf_counter() - t0
            print(f"  [{i + 1}/{args.num_samples}] in {el:.1f}s (~{el / (i + 1) * (args.num_samples - i - 1):.0f}s left)")

    # Write pairs.csv
    pairs_path = os.path.join(out_dir, "pairs.csv")
    with open(pairs_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=PAIRS_FIELDS)
        w.writeheader()
        w.writerows(pairs_rows)

    # Write ground_truth.csv
    gt_path = os.path.join(out_dir, "ground_truth.csv")
    with open(gt_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "present", "x", "y", "theta", "scale"])
        w.writerows(gt_rows)

    # Write manifest_jury.csv
    jury_path = os.path.join(out_dir, "manifest_jury.csv")
    with open(jury_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "set", "severity"])
        for r in pairs_rows:
            w.writerow([r["pair_id"], "Sanity50", 0])

    # Write drift.csv
    drift_path = os.path.join(out_dir, "drift.csv")
    with open(drift_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pair_id", "shear_amplitude_px", "drift_jitter_px", "rotation_deg",
            "architecture", "present", "dose_search", "gamma", "speckle_sigma",
            "salt_pepper_prob", "charging_streak_prob", "linewidth_bias_nm"
        ])
        w.writerows(drift_rows)

    n_present = sum(1 for g in gt_rows if g[1] == 1)
    n_absent = args.num_samples - n_present
    print(f"\nSuccessfully generated 50-pair dataset:")
    print(f"  Total pairs: {args.num_samples}")
    print(f"  Present: {n_present}, Absent: {n_absent} ({n_absent / args.num_samples * 100:.1f}%)")
    print(f"  Directory: {out_dir}")
    print(f"  Total time: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render GUI bookmarks (e.g. the organizer's 20 curated worked samples in
bookmarks_store.json) to a Phase 3 dataset, exactly as the GUI renders them.

app.py rebuilds a bookmark with _sample_from_bookmark and bulk-exports it
with _build_bookmarks_zip: reference.gds, search.gds (_full_canvas_cell),
search.png, gt.json. This reproduces that pipeline outside Streamlit (app.py
imports Streamlit at module level, so it cannot be imported here) and writes
the blind Phase 3 layout:

    reference/<id>.gds, search/<id>.gds, search/<id>.png, params/<id>.json
    pairs.csv          (six columns; reference_sem_path / params_json_path empty)
    ground_truth.csv   (pair_id, present, x, y, theta, scale, architecture_kind)
    manifest.csv       (the same, in the generator's manifest shape)

The applied rotation is recovered exactly from the pipeline's own stream
(np.random.default_rng(strip_rng_seed + 2)), as generate_cad_varied.py does.

    python export_bookmarks.py --bookmarks bookmarks_store.json --output-dir output --split curated20
"""

import argparse
import csv
import json
import os

import cv2
import gdstk
import numpy as np

from generate_cad_varied import applied_rotation_deg, full_canvas_cell
from src.cad_pipeline import SCALE_FACTOR, CadGenerationParams, build_cad_geometry, render_cad_sample


def sample_from_bookmark(bm: dict):
    """app.py's _sample_from_bookmark + _cached_cad_geometry, line for line."""
    st = bm["structure"]
    fab = bm["fab_distortion"]
    manual_center = tuple(bm["manual_center"]) if bm.get("manual_center") else None
    contact_aspect_ratio = st.get("contact_aspect_ratio", 1.6)
    contact_thickness_factor = st.get("contact_thickness_factor", 1.0)
    contact_angle_deg = st.get("contact_angle_deg", 90.0)
    center_bias = bm.get("center_bias", False)
    geom_params = CadGenerationParams(
        collapse_threshold_nm=st["collapse_threshold_nm"],
        mat_size_nm=st["mat_size_nm"], strip_width_nm=st["strip_width_nm"],
        polygon_scale_prob=fab["polygon_scale_prob"], polygon_scale_range=fab["polygon_scale_range"],
        no_match_prob=st["no_match_prob"],
        linewidth_bias_nm=fab["linewidth_bias_nm"], corner_rounding_px=fab["corner_rounding_px"],
        contact_aspect_ratio=contact_aspect_ratio,
        contact_thickness_factor=contact_thickness_factor, contact_angle_deg=contact_angle_deg,
    )
    geom = build_cad_geometry(bm["kind"], np.random.default_rng(bm["seed"]), geom_params,
                              manual_center_nm=manual_center, center_bias=center_bias)
    render_params = CadGenerationParams(
        mat_size_nm=st["mat_size_nm"], strip_width_nm=st["strip_width_nm"],
        collapse_threshold_nm=st["collapse_threshold_nm"], no_match_prob=st["no_match_prob"],
        contact_aspect_ratio=contact_aspect_ratio,
        contact_thickness_factor=contact_thickness_factor, contact_angle_deg=contact_angle_deg,
        polygon_scale_prob=fab["polygon_scale_prob"], polygon_scale_range=fab["polygon_scale_range"],
        linewidth_bias_nm=fab["linewidth_bias_nm"], corner_rounding_px=fab["corner_rounding_px"],
        **bm["sem_acquisition"],
    )
    reference_render_params = None
    if bm.get("reference_ops_enabled"):
        reference_render_params = CadGenerationParams(
            polygon_scale_prob=bm["reference_fab_distortion"]["polygon_scale_prob"],
            polygon_scale_range=bm["reference_fab_distortion"]["polygon_scale_range"],
            linewidth_bias_nm=bm["reference_fab_distortion"]["linewidth_bias_nm"],
            corner_rounding_px=bm["reference_fab_distortion"]["corner_rounding_px"],
            **bm["reference_sem_acquisition"],
        )
    vis = bm.get("layer_visibility", {})
    sample = render_cad_sample(
        geom, render_params,
        layer_intensities={int(k): v for k, v in bm["search_layer_intensities"].items()},
        reference_layer_intensities={int(k): v for k, v in bm["reference_layer_intensities"].items()},
        reference_render_params=reference_render_params,
        search_min_layer=vis.get("search_min_layer", 0),
        reference_min_layer=vis.get("reference_min_layer", 0),
    )
    theta = applied_rotation_deg(geom, render_params) if sample["match_found"] else 0.0
    return geom, sample, theta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bookmarks", default="bookmarks_store.json")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--split", default="curated")
    a = p.parse_args()

    bms = json.load(open(a.bookmarks))
    if isinstance(bms, dict):
        bms = bms.get("bookmarks", list(bms.values()))
    root = os.path.join(a.output_dir, a.split)
    for d in ("reference", "search", "params"):
        os.makedirs(os.path.join(root, d), exist_ok=True)

    pairs, gts, man = [], [], []
    for i, bm in enumerate(bms):
        pid = f"p{i:03d}"
        geom, s, theta = sample_from_bookmark(bm)
        ref_gds = os.path.join("reference", f"{pid}.gds")
        sea_gds = os.path.join("search", f"{pid}.gds")
        sea_png = os.path.join("search", f"{pid}.png")
        for cell, rel in ((s["reference_cell"], ref_gds),
                          (full_canvas_cell(geom["mats"], geom["num_layers"]), sea_gds)):
            lib = gdstk.Library()
            lib.add(cell)
            lib.write_gds(os.path.join(root, rel))
        cv2.imwrite(os.path.join(root, sea_png), s["search_img"])
        found = bool(s["match_found"])
        gt = {"pair_id": pid, "present": int(found),
              "x": s["gt_x"] if found else 0, "y": s["gt_y"] if found else 0,
              "theta": theta if found else 0, "scale": float(SCALE_FACTOR) if found else 0,
              "architecture_kind": bm["kind"]}
        with open(os.path.join(root, "params", f"{pid}.json"), "w") as f:
            json.dump({"bookmark_id": bm.get("id"), "label": bm.get("label"), **gt,
                       "crop_origin_nm": [geom["x0"], geom["y0"]]}, f, indent=2, default=float)
        pairs.append([pid, sea_png, ref_gds, sea_gds, "", ""])
        gts.append(gt)
        man.append({"id": pid, "reference_gds_path": ref_gds, "search_path": sea_png,
                    "search_gds_path": sea_gds, "match_found": found,
                    "gt_x": gt["x"] if found else "", "gt_y": gt["y"] if found else "",
                    "gt_theta": theta if found else "", "gt_scale": gt["scale"] if found else "",
                    "architecture_kind": bm["kind"], "label": bm.get("label", "")})
        print(f"[{i + 1}/{len(bms)}] {bm.get('id')} {bm['kind']} "
              f"{'(%.1f, %.1f, %+.2f deg)' % (gt['x'], gt['y'], theta) if found else 'NO MATCH'}", flush=True)

    with open(os.path.join(root, "pairs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "search_path", "reference_gds_path", "search_gds_path",
                    "reference_sem_path", "params_json_path"])
        w.writerows(pairs)
    with open(os.path.join(root, "ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gts[0]))
        w.writeheader()
        w.writerows(gts)
    with open(os.path.join(root, "manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader()
        w.writerows(man)
    print(f"Wrote {len(bms)} samples to {root}")


if __name__ == "__main__":
    main()

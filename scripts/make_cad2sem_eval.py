#!/usr/bin/env python3
"""Build a CAD2SEM evaluation set with the organizer's own pipeline.

The organizer's dataset CLI (``generate_cad_dataset.py``) writes no search-side
GDS and never passes ``search_rotation_deg``, so it cannot produce a set that
exercises the Phase 3 read path. This drives ``build_cad_geometry`` /
``render_cad_sample`` directly and exports the search CAD through the same
``_full_canvas_cell`` merge that ``app.py``'s download button uses, so every
byte comes from unmodified upstream code.

Two profiles, sharing geometry and ground truth so they are paired:

  default  a clean capture -- the regime the Phase 3 dev sets already cover
  --harsh  the same sites imaged badly (starved dose, high detector noise, a
           wide beam, heavy shear, streaks and salt-and-pepper). This is the
           regime that exposed the tile-floor cliff in driftsense.cad_anchor.

Usage:
    python scripts/make_cad2sem_eval.py --out eval20
    python scripts/make_cad2sem_eval.py --out eval20_harsh --harsh
"""
import argparse
import csv
import os
import sys

import cv2
import gdstk
import numpy as np


def _load_i4c(root: str):
    sys.path.insert(0, os.path.abspath(root))
    from src.cad_pipeline import (  # noqa: E402
        CadGenerationParams, MAX_SEARCH_ROTATION_DEG, build_cad_geometry, render_cad_sample,
    )
    from src.cad import yield_model  # noqa: E402
    return CadGenerationParams, build_cad_geometry, render_cad_sample, yield_model, MAX_SEARCH_ROTATION_DEG


def full_canvas_cell(mats: list, num_layers: int) -> gdstk.Cell:
    """The search-side CAD export, identical to app.py's _full_canvas_cell:
    every mat's *design* polygons (never fab_cell) merged at canvas coords."""
    out = gdstk.Cell("SEARCH_SIDE_CAD")
    for mat in mats:
        mx0, my0 = mat["x0"], mat["y0"]
        for layer in range(num_layers):
            for poly in mat["design_cell"].get_polygons(layer=layer, datatype=0):
                poly.translate(mx0, my0)
                out.add(poly)
    return out


def gds_write(cell: gdstk.Cell, path: str) -> None:
    lib = gdstk.Library()
    lib.add(cell)
    lib.write_gds(path)


def draw_params(Params, rng, harsh: bool, max_rot: float, absent: bool):
    """Imaging draws differ between profiles; every draw consumes exactly one
    value either way, so both profiles see the same RNG stream and therefore
    the same geometry and ground truth."""
    u = rng.uniform
    return Params(
        beam_spot_size_nm=float(u(9.0, 16.0) if harsh else u(4.0, 8.0)),
        dose_search=float(u(15.0, 70.0) if harsh else u(120.0, 700.0)),
        shear_amplitude_px=float(u(2.0, 6.0) if harsh else u(0.0, 4.0)),
        drift_jitter_px=float(u(0.0, 1.2)),
        detector_noise_sigma_search=float(u(14.0, 32.0) if harsh else u(2.0, 6.0)),
        astigmatism_ratio=float(u(1.0, 1.5)),
        search_rotation_deg=float(u(1.0, max_rot)),
        barrel_distortion_k=0.0,                  # organizer: not in the evaluation sets
        vignette_strength=float(u(0.2, 0.55) if harsh else u(0.0, 0.2)),
        gamma=float(u(0.9, 1.15)),
        charging_streak_prob=float(u(0.8, 2.5) if harsh else u(0.0, 1.1)),
        charging_streak_intensity=float(u(0.15, 0.45) if harsh else u(0.0, 0.16)),
        speckle_sigma=float(u(0.01, 0.06) if harsh else u(0.0, 0.01)),
        salt_pepper_prob=float(u(0.2, 1.2) if harsh else u(0.0, 0.25)),
        mat_size_nm=float(u(2000.0, 2300.0)),
        strip_width_nm=float(u(300.0, 400.0)),
        contact_aspect_ratio=float(u(1.3, 1.7)),
        contact_thickness_factor=float(u(0.9, 1.5)),
        polygon_scale_prob=float(u(0.05, 0.25)),
        polygon_scale_range=float(u(0.05, 0.25)),
        linewidth_bias_nm=float(u(-5.0, 4.0)),
        corner_rounding_px=0.0,                   # gdstk's fillet() segfaults upstream
        no_match_prob=1.0 if absent else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--n", type=int, default=20, help="number of pairs (default 20)")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--harsh", action="store_true", help="degrade only the imaging")
    ap.add_argument("--i4c-root", default=os.path.join(os.path.dirname(__file__), "..", "generator_i4c"),
                    help="checkout of the organizer generator (default: the vendored one)")
    a = ap.parse_args()

    Params, build_geom, render, yield_model, max_rot = _load_i4c(a.i4c_root)
    rng = np.random.default_rng(a.seed)
    for d in ("reference", "search"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)

    # ~1 in 12 sites carry no true match, as the brief states.
    absent_at = set(range(4, a.n, 9))
    pairs, gts = [], []
    for i in range(a.n):
        pid = f"p{i + 1:03d}"
        kind = "dram" if i % 2 == 0 else "finfet"
        p = draw_params(Params, rng, a.harsh, max_rot, i in absent_at)
        geom = build_geom(kind, rng, p)
        nl = geom["num_layers"]

        # Half the set sits on the default yield model and half is pushed off
        # it, so "did you know your yield fit was wrong" is a gradable axis.
        base = {L: yield_model.yield_to_intensity(yield_model.layer_yield(L, nl)) for L in range(nl)}
        if i % 2 == 0:
            greys = {L: int(np.clip(v + rng.normal(0, 3), 0, 255)) for L, v in base.items()}
        else:
            greys = {L: int(np.clip(v * rng.uniform(0.55, 1.45) + rng.normal(0, 18), 0, 255))
                     for L, v in base.items()}

        sample = render(geom, p, layer_intensities=greys)
        # The pipeline draws the rotation internally and never returns it;
        # reproduce the identical stream (cad_pipeline.py, render_cad_sample).
        rot_rng = np.random.default_rng(geom["strip_rng_seed"] + 2)
        angle = float(rot_rng.uniform(-p.search_rotation_deg, p.search_rotation_deg))

        gds_write(sample["reference_cell"], os.path.join(a.out, "reference", f"{pid}.gds"))
        gds_write(full_canvas_cell(geom["mats"], nl), os.path.join(a.out, "search", f"{pid}.gds"))
        cv2.imwrite(os.path.join(a.out, "search", f"{pid}.png"), sample["search_img"])

        pairs.append({"pair_id": pid, "search_path": f"search/{pid}.png",
                      "reference_gds_path": f"reference/{pid}.gds",
                      "search_gds_path": f"search/{pid}.gds",
                      "reference_sem_path": "", "params_json_path": ""})
        present = int(sample["match_found"])
        gts.append({"pair_id": pid, "present": present,
                    "x": f'{sample["gt_x"]:.6f}' if present else "0.0",
                    "y": f'{sample["gt_y"]:.6f}' if present else "0.0",
                    "theta": f"{angle:.6f}" if present else "0.0",
                    "scale": "10.00" if present else "0.0"})
        print(f"{pid} {kind:6s} present={present} theta={angle:+6.2f} "
              f"shear={p.shear_amplitude_px:4.2f} dose={p.dose_search:6.1f}")

    with open(os.path.join(a.out, "pairs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0])); w.writeheader(); w.writerows(pairs)
    with open(os.path.join(a.out, "ground_truth.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(gts[0])); w.writeheader(); w.writerows(gts)
    print(f"\nWrote {len(pairs)} pairs to {a.out}/")


if __name__ == "__main__":
    main()

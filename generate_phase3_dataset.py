#!/usr/bin/env python3
"""Generate a Phase 3 dataset: GDS reference + search image + params.

Each training site produces the **five artefacts** the Phase 3 briefing names:

    reference/<id>.gds          reference CAD (design geometry)
    reference/<id>_preview.png  clean render of that CAD (QA eyeballing only --
                                NOT part of the matching task)
    search/<id>.png             noisy Search SEM image @ 10 nm/px
    reference_sem/<id>.png      SEM-style capture of the reference
    params/<id>.json            generation parameters, incl. per-layer brightness

and a ``pairs.csv`` in the Phase 3 six-column layout. The blind split leaves
``reference_sem_path`` and ``params_json_path`` empty -- that is the whole point
of the layout, and ``--blind`` writes it that way.

Absent pairs default to **8%** (about one site in twelve), which is Phase 3's
disclosed rate and *not* Phase 2's 20%.

    python generate_phase3_dataset.py --num-pairs 200 --output-dir ./output

The search image is produced by the same imaging chain Phase 1/2 use, so a
Phase 3 search frame is directly comparable with a Phase 2 one. Only the
reference changes: an image becomes a design file.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from driftsense import gds as gds_read  # noqa: E402
from driftsense import gds_layers  # noqa: E402
from driftsense import params3  # noqa: E402
from driftsense.presets import architecture_presets  # noqa: E402

REFERENCE_SIZE_PX = 1000
FINE_CANVAS_SIZE_PX = 10000
# Mat/strip geometry. Phase 1/2 samples these per canvas from
# mat_size_nm ~ U(1800, 4000) and strip_width_nm ~ U(200, 500)
# (driftsense/generate.py); the defaults here sit in that range.
#
# Mat size matters for MORE than looks: a 1000 nm reference taken from deep
# inside a single periodic array has ~4 near-identical aliases within 0.005
# ZNCC of the true peak, which makes localisation ambiguous by construction.
# Smaller mats put more mat/strip boundaries in the canvas, and since adjacent
# mats draw different presets (different pitch) a boundary in the window is
# what actually disambiguates it.
MAT_SIZE_NM = 2000.0
STRIP_WIDTH_NM = 420.0

PAIRS_FIELDS = ["pair_id", "search_path", "reference_gds_path",
                "search_gds_path", "reference_sem_path", "params_json_path"]


def _preset(name: str) -> dict:
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "generator"))
    from src.presets import get_preset  # noqa: PLC0415
    return get_preset(name)


def _zone_grid(size_px, rng, mat_lo=1800.0, mat_hi=4000.0,
               strip_lo=200.0, strip_hi=500.0):
    """Irregular mat/strip partition of the canvas.

    Sizes are **randomised per span**, not fixed. A regular grid is itself
    periodic: with mat=2000 and strip=420 the period is 2420 nm = 242 px in the
    search frame, and a reference that contains a mat boundary then matches
    every other mat boundary equally well -- measured as a consistent +240 px
    alias, i.e. exactly one grid period. Randomising the spans removes the
    global period, which is what makes a boundary-containing reference
    informative rather than just repeatable.

    This mirrors the Phase 1/2 generator, which samples
    ``mat_size_nm ~ U(1800, 4000)`` and ``strip_width_nm ~ U(200, 500)``
    (driftsense/generate.py:172-173) for the same reason.
    """
    spans, pos, is_mat = [], 0.0, True
    while pos < size_px:
        if is_mat:
            span = float(rng.uniform(mat_lo, mat_hi))
        else:
            span = float(rng.uniform(strip_lo, strip_hi))
        end = min(pos + span, size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos = end
        is_mat = not is_mat
    return spans


def _interior_crop(big, rng):
    """A crop fully inside one mat and fully inside the canvas.

    Tries mats in random order and returns the first that admits a legal
    window, so a mat against the canvas edge never produces an out-of-range
    crop origin.
    """
    order = list(range(len(big)))
    rng.shuffle(order)
    for i in order:
        m = big[i]
        lo_x = max(m["x0"], 0)
        hi_x = min(m["x0"] + m["w"], FINE_CANVAS_SIZE_PX) - REFERENCE_SIZE_PX
        lo_y = max(m["y0"], 0)
        hi_y = min(m["y0"] + m["h"], FINE_CANVAS_SIZE_PX) - REFERENCE_SIZE_PX
        if hi_x < lo_x or hi_y < lo_y:
            continue
        x0 = int(rng.integers(lo_x, hi_x + 1)) if hi_x > lo_x else int(lo_x)
        y0 = int(rng.integers(lo_y, hi_y + 1)) if hi_y > lo_y else int(lo_y)
        return m, x0, y0
    raise RuntimeError(
        f"no mat admits a {REFERENCE_SIZE_PX} nm window inside the canvas; "
        f"increase MAT size bounds or reduce REFERENCE_SIZE_PX")


def build_site(architecture: str, preset_name: str, rng,
               collapse_threshold_nm: float = 10.0,
               boundary_bias: float = 1.0):
    """Build one site's mat layout and its 8-layer GDS cells.

    Returns ``(design_cell, mats, reference_cell, x0, y0)`` where ``mats`` is
    the layout used to rasterize the search canvas, and ``reference_cell`` is
    the clipped window that becomes ``reference/<id>.gds``.

    **The crop deliberately straddles a mat/strip boundary.** A crop taken
    entirely inside one mat sees only the repeating cell array, and because the
    array is periodic every repeat scores almost identically under correlation
    -- on a 242 nm pitch with a 1000 nm reference that is ~4 aliases within
    0.02 of the true peak, which makes localisation ambiguous by construction
    rather than by difficulty. Including a boundary puts a non-periodic feature
    in the reference, which is what actually disambiguates it. The Phase 1/2
    generator does the same thing for the same reason
    (``_pick_visible_crop_origin`` / ``boundary_bias``).
    """
    preset = _preset(preset_name)
    rows = _zone_grid(FINE_CANVAS_SIZE_PX, rng)
    cols = _zone_grid(FINE_CANVAS_SIZE_PX, rng)

    mats = []
    for row_is_mat, y0, y1 in rows:
        for col_is_mat, x0, x1 in cols:
            if not (row_is_mat and col_is_mat and y1 > y0 and x1 > x0):
                continue
            w, h = x1 - x0, y1 - y0
            child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            # A DIFFERENT preset per mat: mats are independently-generated
            # array blocks, so the layout is not globally periodic even before
            # the strips are considered.
            per_mat_preset = architecture_presets(architecture)[
                int(child.integers(0, len(architecture_presets(architecture))))]
            # Build the cell at the mat's LARGER dimension so its pattern
            # covers the whole span; the raster step then paints only the
            # (w, h) window, which is the correct sub-region of the pattern
            # rather than a square pattern cropped to fit.
            built = gds_layers.build_gds(architecture, max(w, h),
                                         _preset(per_mat_preset),
                                         collapse_threshold_nm, child)
            # The pattern is laid out over a square max(w,h) field, but only
            # the (w,h) window is this mat. Clip, so the design contains
            # exactly what the raster paints.
            cell = gds_layers.clip_cell_to_rect(built, w, h)
            mats.append({"cell": cell, "x0": x0, "y0": y0, "w": w, "h": h})
    if not mats:
        raise RuntimeError("no mats built -- check MAT_SIZE_NM/STRIP_WIDTH_NM")

    strips = _strip_cells(rows, cols, rng)

    big = [m for m in mats
           if m["w"] >= REFERENCE_SIZE_PX and m["h"] >= REFERENCE_SIZE_PX]
    if not big:
        raise RuntimeError(
            f"no mat is at least {REFERENCE_SIZE_PX} nm wide and tall; "
            f"reduce MAT_SIZE_NM or REFERENCE_SIZE_PX")

    # Prefer a crop that straddles this mat's right or bottom edge, so the
    # window contains part of the neighbouring strip. Fall back to an interior
    # crop when the geometry leaves no room for one.
    #
    # The window is deliberately NOT centred on the edge. Centring makes the
    # reference ~50% flat separator, and a reference that is half featureless
    # matches a wrong location almost as well as the right one (measured: the
    # true peak scored 0.86 while a periodic alias scored 0.90). Offsetting
    # toward the mat keeps the strip a visible minority -- which is what
    # disambiguates the window -- while leaving the majority of it device
    # geometry, which is what makes the match well-posed.
    STRIP_SHARE = 0.22   # target fraction of the window over the strip
    if boundary_bias > 0 and rng.random() < boundary_bias:
        edge_pull = REFERENCE_SIZE_PX * (0.5 - STRIP_SHARE)
        cands = []
        for m in big:
            for axis in ("x", "y"):
                if axis == "x":
                    cx = m["x0"] + m["w"] - edge_pull + float(rng.uniform(-120, 120))
                    cy = m["y0"] + m["h"] / 2.0 + float(rng.uniform(-300, 300))
                else:
                    cx = m["x0"] + m["w"] / 2.0 + float(rng.uniform(-300, 300))
                    cy = m["y0"] + m["h"] - edge_pull + float(rng.uniform(-120, 120))
                x0 = int(round(cx - REFERENCE_SIZE_PX / 2.0))
                y0 = int(round(cy - REFERENCE_SIZE_PX / 2.0))
                # The window must lie fully inside the CANVAS. Clamping would
                # silently relocate the crop and invalidate the ground truth
                # (the crop origin IS the label), so candidates that need
                # clamping are rejected instead of nudged. A window pinned at
                # the canvas edge cannot hold a full reference at all.
                if not 0 <= x0 <= FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX:
                    continue
                if not 0 <= y0 <= FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX:
                    continue
                cands.append((m, x0, y0))
        if cands:
            mat, x0, y0 = cands[int(rng.integers(0, len(cands)))]
        else:
            mat, x0, y0 = _interior_crop(big, rng)
    else:
        mat, x0, y0 = _interior_crop(big, rng)

    ref_cell = gds_layers.clip_multi_cell(
        list(mats) + list(strips), x0, y0, REFERENCE_SIZE_PX,
        gds_layers.NUM_LAYERS)
    return mat["cell"], mats, strips, ref_cell, x0, y0


def _layer_intensities(architecture: str, rng, num_layers=8) -> dict:
    """Sample a per-layer brightness vector.

    Brightness must VARY per layer for "infer it per layer" to be a learnable
    task; the organizer-side default ladder is used as the centre and each
    layer is jittered around it. The background/field value is kept below every
    drawn layer so a design never disappears into its own background.
    """
    out = {}
    for i in range(num_layers):
        base = gds_read.yield_to_intensity(gds_read.layer_yield(i, num_layers))
        jitter = int(rng.normal(0, 18))
        lo = gds_read.background_intensity() + 6
        out[i] = int(np.clip(base + jitter, lo, 255))
    return out


def _strip_cells(rows, cols, rng):
    """Every non-mat region of the grid, as real design geometry."""
    mats_xy = set()
    for row_is_mat, y0, y1 in rows:
        for col_is_mat, x0, x1 in cols:
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                mats_xy.add((x0, y0))
    cells = []
    for row_is_mat, y0, y1 in rows:
        for col_is_mat, x0, x1 in cols:
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                continue
            if x1 <= x0 or y1 <= y0:
                continue
            child = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
            cells.append({"cell": gds_layers.build_strip_cell(
                x0, y0, x1 - x0, y1 - y0, rng=child),
                "x0": x0, "y0": y0, "w": x1 - x0, "h": y1 - y0})
    return cells


def render_search_canvas(mats, strips, num_layers, layer_intensities, min_layer=0):
    """Rasterize the full layout -- mat AND strip cells -- into the fine canvas.

    Every drawn region comes from GDS geometry, so the search raster and the
    reference clipped from it derive from one source. Each cell is rasterized
    at its own ``(w, h)``: a mat is not square, and a square raster sliced to
    ``[:h, :w]`` would drop the geometry beyond the slice.
    """
    canvas = np.full((FINE_CANVAS_SIZE_PX, FINE_CANVAS_SIZE_PX),
                     gds_read.background_intensity(), dtype=np.uint8)
    for cell in list(strips) + list(mats):
        sub = gds_read.rasterize_layers(
            _cell_polygons(cell["cell"], num_layers),
            num_layers, size=(cell["w"], cell["h"]),
            layer_intensities=layer_intensities, min_layer=min_layer)
        h, w = cell["h"], cell["w"]
        canvas[cell["y0"]:cell["y0"] + h, cell["x0"]:cell["x0"] + w] = sub[:h, :w]
    return canvas


def _cell_polygons(cell, num_layers):
    """Per-layer polygon points from a gdstk Cell, datatype 0 only."""
    return {
        L: [np.asarray(p.points, dtype=np.float64)
            for p in cell.get_polygons(layer=L, datatype=0)]
        for L in range(num_layers)
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-pairs", type=int, default=20)
    ap.add_argument("--output-dir", default="./output_phase3")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--architectures", nargs="+", default=["dram", "finfet"],
                    choices=["dram", "finfet"])
    ap.add_argument("--absent-frac", type=float, default=0.08,
                    help="Phase 3 discloses ~1 site in 12 with no true match "
                         "(0.083); this is NOT Phase 2's 0.20")
    ap.add_argument("--blind", action="store_true",
                    help="write the blind-split pairs.csv (withheld columns empty)")
    args = ap.parse_args(argv)

    if not 0.0 <= args.absent_frac <= 1.0:
        ap.error("--absent-frac must be in [0, 1]")

    import cv2  # noqa: PLC0415
    from src import sem_imaging  # noqa: PLC0415

    root = os.path.join(args.output_dir, args.split)
    dirs = {k: os.path.join(root, k) for k in
            ("reference", "search", "reference_sem", "params")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pairs_rows = []
    t0 = time.perf_counter()

    for i in range(args.num_pairs):
        pid = f"p{i:04d}"
        arch = args.architectures[int(rng.integers(0, len(args.architectures)))]
        preset_name = architecture_presets(arch)[
            int(rng.integers(0, len(architecture_presets(arch))))]
        present = not (args.absent_frac > 0 and rng.random() < args.absent_frac)

        design_cell, mats, strips, ref_cell, x0, y0 = build_site(arch, preset_name, rng)
        num_layers = gds_layers.NUM_LAYERS
        intens = _layer_intensities(arch, rng, num_layers)

        # --- reference GDS
        gds_path = os.path.join(dirs["reference"], f"{pid}.gds")
        import gdstk  # noqa: PLC0415
        lib = gdstk.Library()
        lib.add(ref_cell)
        lib.write_gds(gds_path)

        # --- search image, from the same geometry the reference was clipped out of
        fine = render_search_canvas(mats, strips, num_layers, intens)
        search_rng = np.random.default_rng(int(rng.integers(0, 2**31 - 1)))
        search_img = sem_imaging.image_search(
            fine, pixel_size_ref_nm=1, pixel_size_search_nm=10,
            spot_size_nm=5.0, dose=200.0, rng=search_rng,
            shear_amplitude_px=1.5, drift_jitter_px=0.5,
            detector_noise_sigma=5.0)
        search_path = os.path.join(dirs["search"], f"{pid}.png")
        cv2.imwrite(search_path, search_img)

        # --- reference SEM capture (training only)
        ref_raster = gds_read.render_reference(
            gds_path, size=REFERENCE_SIZE_PX, layer_intensities=intens)
        ref_sem = sem_imaging.image_reference(
            ref_raster, pixel_size_nm=1, spot_size_nm=5.0, dose=2000.0,
            rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
            detector_noise_sigma=2.0, drift_jitter_px=0.5)
        ref_sem_path = os.path.join(dirs["reference_sem"], f"{pid}.png")
        cv2.imwrite(ref_sem_path, ref_sem)

        # --- preview (QA only, not a task artefact)
        cv2.imwrite(os.path.join(dirs["reference"], f"{pid}_preview.png"), ref_raster)

        # --- params JSON
        params = params3.build_params(
            arch, num_layers, intens,
            reference_gds_path=os.path.join("reference", f"{pid}.gds"),
            search_gds_path=os.path.join("search", f"{pid}.gds"),
            present=present, scale=10.0,
            generation={"preset": preset_name, "seed": args.seed,
                        "crop_x0": int(x0), "crop_y0": int(y0)})
        params_path = os.path.join(dirs["params"], f"{pid}.json")
        params3.write_params(params_path, params)

        rel = lambda p: os.path.relpath(p, root)  # noqa: E731
        # Ground truth: the reference is a REFERENCE_SIZE_PX window whose
        # origin is (x0, y0) in fine-canvas nm. At 10 nm/px the search frame is
        # the fine canvas downsampled 10x, so the window centre lands at
        # (x0 + 500)/10, (y0 + 500)/10 -- the same conversion the Phase 1/2
        # pipeline documents.
        gt_cx = (x0 + REFERENCE_SIZE_PX / 2.0) / 10.0
        gt_cy = (y0 + REFERENCE_SIZE_PX / 2.0) / 10.0
        pairs_rows.append({
            "pair_id": pid,
            "search_path": rel(search_path),
            "reference_gds_path": rel(gds_path),
            "search_gds_path": rel(gds_path),   # same site frame
            "reference_sem_path": "" if args.blind else rel(ref_sem_path),
            "params_json_path": "" if args.blind else rel(params_path),
            "_present": present,
            "_gt_x": gt_cx,
            "_gt_y": gt_cy,
            "_arch": arch,
        })
        if (i + 1) % 10 == 0 or i + 1 == args.num_pairs:
            el = time.perf_counter() - t0
            print(f"  [{i+1}/{args.num_pairs}] {el:.0f}s "
                  f"(~{el/(i+1)*(args.num_pairs-i-1):.0f}s left)", flush=True)

    pairs_csv = os.path.join(root, "pairs.csv")
    with open(pairs_csv, "w", newline="") as f:
        # extrasaction="ignore": the internal _gt_*/_present keys carried for
        # ground_truth.csv must not leak into the six-column Phase 3 layout.
        w = csv.DictWriter(f, fieldnames=PAIRS_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(pairs_rows)

    # ground_truth.csv: same columns and order as Phase 2, so the Phase 2
    # reader and scorer work on it unmodified. Absent pairs carry present=0 and
    # zeros in the pose columns.
    gt_csv = os.path.join(root, "ground_truth.csv")
    with open(gt_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "present", "x", "y", "theta", "scale"])
        for row in pairs_rows:
            if row["_present"]:
                w.writerow([row["pair_id"], 1,
                            f'{row["_gt_x"]:.4f}', f'{row["_gt_y"]:.4f}',
                            0.0, 10.0])
            else:
                w.writerow([row["pair_id"], 0, 0, 0, 0, 0])

    n_absent = sum(1 for r in pairs_rows if not r["params_json_path"])
    print(f"\nwrote {len(pairs_rows)} pairs to {root}")
    print(f"  blind split: {bool(args.blind)}"
          + (f" ({n_absent} without params -- absent by construction)"
             if args.blind else ""))
    print(f"  pairs.csv, ground_truth.csv, reference/, search/, "
          f"reference_sem/, params/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

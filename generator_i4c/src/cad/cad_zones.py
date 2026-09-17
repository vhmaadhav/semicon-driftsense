"""
Large-scale zone composition for Phase 3's CAD-derived fine canvas -- same
mat/strip tiling idea as src/patterns/zones.py, but each mat is built as a
real gdstk.Cell (src/cad/dram_gds.py or finfet_gds.py) and rasterized via
the yield model, rather than painted directly as a raster mask.

Each mat carries TWO cells: `design_cell` (the ideal, as-drawn geometry --
what the Reference GDS and every CAD viewer show) and `fab_cell` (the same
geometry with fabrication/imaging distortion applied -- CD bias, corner
rounding, polygon-scale outliers; see src/cad/fab_distortion.py). Only the
Search-side render (rasterize_zone_canvas below) uses `fab_cell`; the
Reference is never distorted, because a design database doesn't encode how
a particular fab run happened to come out.
"""

from __future__ import annotations

import numpy as np

from src.cad.dram_gds import build_dram_mat_gds, NUM_LAYERS as DRAM_LAYERS
from src.cad.finfet_gds import build_finfet_mat_gds, NUM_LAYERS as FINFET_LAYERS
from src.cad.fab_distortion import apply_fab_distortion
from src.cad.render import rasterize_cell
from src.cad import yield_model
from src.presets import presets_for_kind

_BUILDERS = {"dram": build_dram_mat_gds, "finfet": build_finfet_mat_gds}
_NUM_LAYERS = {"dram": DRAM_LAYERS, "finfet": FINFET_LAYERS}

STRIP_LINE_PITCH_NM = 220
STRIP_LINE_WIDTH_NM = 9
STRIP_BASE_INTENSITY = yield_model.background_intensity()
STRIP_LINE_INTENSITY = yield_model.yield_to_intensity(0.45)


def _strip_routing_texture(size_px: int, rng: np.random.Generator):
    import numpy as np
    canvas = np.full((size_px, size_px), STRIP_BASE_INTENSITY, dtype=np.uint8)
    half = STRIP_LINE_WIDTH_NM / 2.0
    for axis_positions, is_row in (
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), True),
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), False),
    ):
        for center in axis_positions:
            lo = max(int(round(center - half)), 0)
            hi = min(int(round(center + half)), size_px)
            if is_row:
                canvas[lo:hi, :] = STRIP_LINE_INTENSITY
            else:
                canvas[:, lo:hi] = STRIP_LINE_INTENSITY
    return canvas


def _zone_grid(size_px: int, mat_size_nm: float, strip_width_nm: float):
    spans = []
    pos = 0.0
    is_mat = True
    while pos < size_px:
        span_len = mat_size_nm if is_mat else strip_width_nm
        end = min(pos + span_len, size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos = end
        is_mat = not is_mat
    return spans


def build_cad_mats(
    size_px: int,
    kind: str,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    mat_size_nm: float = 2600.0,
    strip_width_nm: float = 320.0,
    polygon_scale_prob: float = 0.0,
    polygon_scale_range: float = 0.0,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
    contact_aspect_ratio: float = 1.6,
    contact_thickness_factor: float = 1.0,
    contact_angle_deg: float = 90.0,
) -> dict:
    """Build the mat/strip layout's *geometry only* -- no rasterization, no
    intensity choice yet. Separated from rendering so a caller (the app) can
    let a user inspect/choose per-layer intensities before any image gets
    produced, without re-rolling the random layout every time they nudge a
    slider.

    `polygon_scale_prob/range`, `linewidth_bias_nm`, `corner_rounding_px`
    are fabrication/imaging effects -- they only ever shape `fab_cell`
    (used for the Search-side render). `design_cell` is always the clean,
    undistorted geometry the builder produces, unconditionally.
    `contact_aspect_ratio`/`contact_thickness_factor`/`contact_angle_deg`
    are structural (how elongated/thick the tablet-shaped contacts/vias
    are, and at what angle) -- they shape both cells identically, since
    that's a real design choice, not a fab defect.
    """
    builder = _BUILDERS[kind]
    num_layers = _NUM_LAYERS[kind]
    presets = presets_for_kind(kind)

    row_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)
    col_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)

    mats = []  # list of dicts: {design_cell, fab_cell, x0, y0, w, h, kind, num_layers}
    strip_rects = []

    for row_is_mat, y0, y1 in row_spans:
        for col_is_mat, x0, x1 in col_spans:
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                mat_h, mat_w = y1 - y0, x1 - x0
                preset = presets[int(rng.integers(0, len(presets)))]
                child_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
                mat_size = max(mat_h, mat_w)
                design_cell = builder(
                    mat_size, preset, collapse_threshold_nm, child_rng,
                    contact_aspect_ratio=contact_aspect_ratio,
                    contact_thickness_factor=contact_thickness_factor, contact_angle_deg=contact_angle_deg,
                )
                fab_cell = apply_fab_distortion(
                    design_cell, num_layers, child_rng,
                    polygon_scale_prob=polygon_scale_prob, polygon_scale_range=polygon_scale_range,
                    linewidth_bias_nm=linewidth_bias_nm, corner_rounding_px=corner_rounding_px,
                )
                mats.append({
                    "design_cell": design_cell, "fab_cell": fab_cell, "x0": x0, "y0": y0, "w": mat_w, "h": mat_h,
                    "kind": kind, "num_layers": num_layers,
                })
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))

    return {"mats": mats, "strip_rects": strip_rects, "num_layers": num_layers}


def rasterize_zone_canvas(
    size_px: int,
    mats: list,
    strip_rects: list,
    rng: np.random.Generator,
    layer_intensities: dict | None = None,
    min_layer: int = 0,
) -> np.ndarray:
    """Composite the mats (each rasterized with `layer_intensities`, falling
    back to the yield model for any layer not given) plus strip texture into
    the full fine canvas. `rng` only drives the strip routing-line texture --
    pass the *same* rng state used when the mats were built (e.g. a fresh
    default_rng seeded the same way) if you want the strips to look
    identical across re-renders with different intensities.

    Renders from `fab_cell` (fabrication/imaging-distorted), since this
    feeds the Search image -- the only place that distortion should be
    visible. The Reference always renders from `design_cell` instead (see
    src/cad/render.py's clip_multi_mat_reference / rasterize_full_canvas_colored).

    `min_layer` excludes any layer below that index -- deeper/more-buried
    layers (see yield_model's layer-number convention) that a coarser,
    wide-FOV Search capture realistically wouldn't resolve.
    """
    canvas = _strip_routing_texture(size_px, rng)
    for mat in mats:
        mat_size = max(mat["w"], mat["h"])
        mat_canvas = rasterize_cell(
            mat["fab_cell"], mat_size, mat["num_layers"], layer_intensities=layer_intensities, min_layer=min_layer,
        )
        canvas[mat["y0"]:mat["y0"] + mat["h"], mat["x0"]:mat["x0"] + mat["w"]] = mat_canvas[:mat["h"], :mat["w"]]
    return canvas


def generate_cad_zone_canvas(
    size_px: int,
    kind: str,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    mat_size_nm: float = 2600.0,
    strip_width_nm: float = 320.0,
    polygon_scale_prob: float = 0.0,
    polygon_scale_range: float = 0.0,
    layer_intensities: dict | None = None,
) -> dict:
    """Convenience one-shot wrapper (build + rasterize) for callers that
    don't need the intermediate geometry -- generate_cad_dataset.py, tests.

    NOTE: geometry (mat presets/positions) is now built before the strip
    texture is drawn, whereas the original single-pass implementation drew
    the strip texture first -- both consume the same rng, just in a
    different order, so a given seed now produces different (but equally
    valid) pixel content than before this function was split. Nothing
    depends on exact pixel reproducibility across that change.
    """
    geometry = build_cad_mats(
        size_px, kind, collapse_threshold_nm, rng,
        mat_size_nm=mat_size_nm, strip_width_nm=strip_width_nm,
        polygon_scale_prob=polygon_scale_prob, polygon_scale_range=polygon_scale_range,
    )
    canvas = rasterize_zone_canvas(size_px, geometry["mats"], geometry["strip_rects"], rng, layer_intensities)
    return {"canvas": canvas, "mats": geometry["mats"], "strip_rects": geometry["strip_rects"]}

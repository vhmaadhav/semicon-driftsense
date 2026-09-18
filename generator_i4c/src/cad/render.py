"""
Rasterize a gdstk.Cell into the same 1 nm/px grayscale convention the rest
of the pipeline uses, with per-layer brightness *derived* from the layer
number via yield_model.layer_yield() -- see that module's docstring.
"""

from __future__ import annotations

import cv2
import gdstk
import numpy as np

from src.cad import yield_model

# Fixed, distinct palette for the layer-colored CAD viewer (not the yield
# model -- this is purely so a human can tell layers apart when inspecting
# geometry, before any intensity has been chosen). BGR, since it's drawn
# with cv2 and converted for display by the caller.
VIEWER_PALETTE_BGR = [
    (70, 60, 55), (255, 140, 70), (110, 210, 90), (60, 90, 245),
    (200, 200, 60), (180, 80, 200), (80, 200, 200), (140, 140, 140),
]
# Distinct fixed color for strip/separator regions in the full-canvas
# viewer -- material boundaries should read clearly as "not a device layer".
SEPARATOR_VIEWER_COLOR = (40, 40, 40)

# KLayout-style layer view: semi-transparent fills (so a layer underneath
# stays visible through the one on top of it, e.g. the DRAM active islands
# under the word/bit lines) plus a dashed outline on every polygon --
# cosmetic, inspection-only choices, not physical -- the yield-based
# rasterize_cell() render stays an opaque painter's-algorithm composite,
# since a real SEM only ever sees the top surface.
KLAYOUT_FILL_ALPHA = 0.45
KLAYOUT_DASH_LEN_PX = 6.0
KLAYOUT_GAP_LEN_PX = 4.0
KLAYOUT_OUTLINE_THICKNESS = 1


def _alpha_blend_fill(
    canvas: np.ndarray, pts_list: list, color: tuple, alpha: float, mask_buf: np.ndarray | None = None,
) -> None:
    """In-place: alpha-blend `color` onto `canvas` wherever the filled
    polygons land, leaving every other pixel untouched -- unlike a plain
    cv2.fillPoly, this lets a layer already painted underneath keep showing
    through instead of being fully occluded.

    Built from cv2 whole-array primitives (addWeighted, copyTo) rather than
    numpy boolean fancy indexing (canvas[hit] = ...) or np.where, both of
    which turned out ~2-3x slower here -- this runs once per layer per mat
    over the whole fine canvas, so the constant-factor matters.
    `mask_buf`, if given, is a scratch (H, W) uint8 array reused across
    layers of the same cell instead of reallocating one per call.
    """
    if not pts_list:
        return
    mask = mask_buf if mask_buf is not None else np.zeros(canvas.shape[:2], dtype=np.uint8)
    mask[:] = 0
    cv2.fillPoly(mask, pts_list, color=255)
    overlay = np.full_like(canvas, color)
    blended = cv2.addWeighted(canvas, 1.0 - alpha, overlay, alpha, 0)
    cv2.copyTo(blended, mask, canvas)


def _draw_dashed_polygon(
    canvas: np.ndarray, pts: np.ndarray, color: tuple, dash_len: float, gap_len: float, thickness: int,
) -> None:
    """Draw a closed polygon's outline as a dashed line. OpenCV has no
    native dashed-polyline primitive, so walk each edge in turn and
    alternately draw dash-length segments / skip gap-length ones, carrying
    the leftover dash/gap phase across each vertex so the pattern stays even
    all the way around instead of resetting at every edge.
    """
    n = len(pts)
    if n < 2:
        return
    period = dash_len + gap_len
    phase = 0.0
    for i in range(n):
        p0 = pts[i].astype(np.float64)
        p1 = pts[(i + 1) % n].astype(np.float64)
        seg_vec = p1 - p0
        seg_len = float(np.hypot(*seg_vec))
        if seg_len < 1e-9:
            continue
        direction = seg_vec / seg_len
        pos = 0.0
        while pos < seg_len:
            cycle_pos = phase % period
            drawing = cycle_pos < dash_len
            remaining_in_state = (dash_len - cycle_pos) if drawing else (period - cycle_pos)
            step = min(remaining_in_state, seg_len - pos)
            if drawing:
                a = p0 + direction * pos
                b = p0 + direction * (pos + step)
                cv2.line(
                    canvas, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)),
                    color, thickness, cv2.LINE_AA,
                )
            pos += step
            phase += step


def rasterize_cell(
    cell: gdstk.Cell,
    size_px: int,
    num_layers: int,
    offset_nm: tuple = (0.0, 0.0),
    layer_intensities: dict | None = None,
    min_layer: int = 0,
) -> np.ndarray:
    """Paint layers 0..num_layers-1 bottom-to-top (painter's algorithm) onto
    a size_px x size_px canvas, at 1 nm/px. `offset_nm` shifts the cell's
    coordinates before rasterizing -- used to render a sub-window of a
    larger mat without needing a separately-clipped cell.

    `layer_intensities`, if given, is a {layer_number: 0-255 intensity} map
    that overrides the yield-model default for that layer -- lets a user
    pick brightness per layer explicitly instead of accepting the derived
    value. Layers not present in the dict still fall back to the model.

    `min_layer` skips any layer below that index entirely -- models a
    resolution/FOV limit where the deepest, most-buried layers (see
    yield_model's layer-number convention: 0 = most buried) don't resolve,
    e.g. a coarser wide-FOV Search capture vs. a close-up Reference.
    """
    layer_intensities = layer_intensities or {}
    canvas = np.full((size_px, size_px), yield_model.background_intensity(), dtype=np.uint8)
    ox, oy = offset_nm

    for layer in range(min_layer, num_layers):
        polygons = cell.get_polygons(layer=layer, datatype=0)
        if not polygons:
            continue
        intensity = layer_intensities.get(layer)
        if intensity is None:
            intensity = yield_model.yield_to_intensity(yield_model.layer_yield(layer, num_layers))
        pts_list = []
        for poly in polygons:
            pts = np.round(poly.points - np.array([ox, oy])).astype(np.int32)
            pts_list.append(pts)
        cv2.fillPoly(canvas, pts_list, color=int(intensity))

    return canvas


def rasterize_cell_layers_colored(
    cell: gdstk.Cell,
    size_px: int,
    num_layers: int,
    offset_nm: tuple = (0.0, 0.0),
    dashed: bool = True,
    min_layer: int = 0,
) -> np.ndarray:
    """CAD-viewer render: each layer in its own fixed, distinct color (BGR),
    independent of any yield/intensity choice -- for inspecting geometry
    before deciding brightness, the way a real layout viewer (e.g. KLayout)
    shows layers: semi-transparent fills so a layer underneath remains
    visible through the one on top, plus (when `dashed`) a dashed polygon
    outline. `dashed=False` skips the per-edge outline pass -- useful for a
    render that will be downscaled a lot before display (e.g. the full-canvas
    overview), where a dash drawn at native resolution would shrink to
    sub-pixel and just be wasted work.

    `min_layer` skips any layer below that index -- see rasterize_cell.
    """
    canvas = np.zeros((size_px, size_px, 3), dtype=np.uint8)
    mask_buf = np.zeros((size_px, size_px), dtype=np.uint8)
    ox, oy = offset_nm

    for layer in range(min_layer, num_layers):
        polygons = cell.get_polygons(layer=layer, datatype=0)
        if not polygons:
            continue
        color = VIEWER_PALETTE_BGR[layer % len(VIEWER_PALETTE_BGR)]
        pts_list = [np.round(poly.points - np.array([ox, oy])).astype(np.int32) for poly in polygons]
        _alpha_blend_fill(canvas, pts_list, color, KLAYOUT_FILL_ALPHA, mask_buf=mask_buf)
        if dashed:
            for pts in pts_list:
                _draw_dashed_polygon(canvas, pts, color, KLAYOUT_DASH_LEN_PX, KLAYOUT_GAP_LEN_PX, KLAYOUT_OUTLINE_THICKNESS)

    return canvas


def clip_and_translate(cell: gdstk.Cell, x0_nm: float, y0_nm: float, size_nm: float, num_layers: int) -> gdstk.Cell:
    """Return a new Cell containing only the polygons (clipped) within the
    [x0, x0+size) x [y0, y0+size) window, translated so that window's
    origin becomes (0, 0) -- what actually gets written as reference.gds.
    `x0_nm`/`y0_nm` are in the cell's own (local) coordinate frame.
    """
    window = gdstk.rectangle((x0_nm, y0_nm), (x0_nm + size_nm, y0_nm + size_nm))
    out = gdstk.Cell(f"REF_{cell.name}")
    for layer in range(num_layers):
        polygons = cell.get_polygons(layer=layer, datatype=0)
        if not polygons:
            continue
        clipped = gdstk.boolean(polygons, [window], "and", layer=layer)
        for poly in clipped:
            poly.translate(-x0_nm, -y0_nm)
            out.add(poly)
    return out


def clip_multi_mat_reference(mats: list, x0: float, y0: float, size: float, num_layers: int) -> gdstk.Cell:
    """Like clip_and_translate, but merges polygons from every mat whose
    bounding box overlaps the [x0, x0+size) x [y0, y0+size) *canvas-coordinate*
    window -- needed once a reference crop is allowed to straddle a mat/strip
    boundary, since its footprint can then span more than one mat. Each
    mat's cell holds polygons in its own local (0,0)-(w,h) frame, so they're
    shifted into canvas coordinates before clipping, then shifted back to be
    relative to (x0, y0). A crop region with no mat underneath it at all
    (pure strip) legitimately contributes no polygons for that area -- a
    real GDS can have empty space too.

    Always reads `design_cell` -- the Reference is the exact, as-drawn
    design, never the fabrication-distorted `fab_cell` used for the Search.
    """
    window = gdstk.rectangle((x0, y0), (x0 + size, y0 + size))
    out = gdstk.Cell(f"REF_MULTI_{int(x0)}_{int(y0)}")
    for mat in mats:
        mx0, my0, mw, mh = mat["x0"], mat["y0"], mat["w"], mat["h"]
        if mx0 + mw <= x0 or mx0 >= x0 + size or my0 + mh <= y0 or my0 >= y0 + size:
            continue  # bounding boxes don't overlap -- skip
        cell = mat["design_cell"]
        for layer in range(num_layers):
            polygons = cell.get_polygons(layer=layer, datatype=0)
            if not polygons:
                continue
            for poly in polygons:
                poly.translate(mx0, my0)
            clipped = gdstk.boolean(polygons, [window], "and", layer=layer)
            for poly in clipped:
                poly.translate(-x0, -y0)
                out.add(poly)
    return out


def rasterize_full_canvas_colored(
    mats: list, strip_rects: list, size_px: int, num_layers: int, min_layer: int = 0,
) -> np.ndarray:
    """Layer-colored CAD-viewer render of the *entire* fine canvas -- every
    mat plus the strip/separator regions between them, each strip drawn in
    a fixed, distinct "separator" color so material boundaries are visible
    in the viewer, not just individual device layers.

    Always reads `design_cell` -- this is a CAD/design-inspection viewer
    ("no intensity applied yet"), so it shows the ideal, as-drawn geometry,
    the same as the Reference. Fabrication distortion only ever shows up in
    the rendered Search image, not in a CAD viewer. `min_layer` skips any
    layer below that index, same convention as rasterize_cell -- lets this
    viewer preview what the Search side would actually resolve.
    """
    canvas = np.full((size_px, size_px, 3), SEPARATOR_VIEWER_COLOR, dtype=np.uint8)
    for sx, sy, sw, sh in strip_rects:
        canvas[sy:sy + sh, sx:sx + sw] = SEPARATOR_VIEWER_COLOR
    for mat in mats:
        mat_size = max(mat["w"], mat["h"])
        mat_colored = rasterize_cell_layers_colored(
            mat["design_cell"], mat_size, num_layers, dashed=False, min_layer=min_layer,
        )
        canvas[mat["y0"]:mat["y0"] + mat["h"], mat["x0"]:mat["x0"] + mat["w"]] = mat_colored[:mat["h"], :mat["w"]]
    return canvas

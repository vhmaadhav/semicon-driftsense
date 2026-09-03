"""
Large-scale zone composition for the fine canvas.

Real chips are not one uniform repeating pattern across a whole field of
view: a memory device is built from discrete sub-array "mats" (blocks of
the periodic cell array), separated by strips of visually distinct material
-- peripheral circuitry, sense-amp/decoder rows, global routing, scribe
lines. This module tiles independently-generated mats (optionally using
different presets of the same architecture kind, e.g. a dense array mat
next to a legacy/relaxed one) across the canvas, separated by flatter strip
regions with sparse routing lines.

The payoff for the dataset: a wide search image with real large-scale
structure instead of one endless grid, and a reference crop that can be
deliberately biased to straddle a mat/strip boundary -- a harder and more
realistic matching scenario than always sampling from deep inside a
uniform field.
"""

from __future__ import annotations

import numpy as np

from src.patterns.dram import generate_dram_canvas
from src.patterns.finfet import generate_finfet_canvas
from src.presets import presets_for_kind

_GENERATORS = {"dram": generate_dram_canvas, "finfet": generate_finfet_canvas}

STRIP_BASE_VAL = 95
STRIP_LINE_VAL = 128
STRIP_LINE_PITCH_NM = 220
STRIP_LINE_WIDTH_NM = 9


def _strip_routing_texture(size_px: int, rng: np.random.Generator) -> np.ndarray:
    """Flat mid-gray fill with sparse orthogonal routing lines -- a stand-in
    for global interconnect/peripheral material, visually distinct from the
    dense periodic mats it separates."""
    canvas = np.full((size_px, size_px), STRIP_BASE_VAL, dtype=np.uint8)

    half = STRIP_LINE_WIDTH_NM / 2.0
    for axis_positions, is_row in (
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), True),
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), False),
    ):
        for center in axis_positions:
            lo = max(int(round(center - half)), 0)
            hi = min(int(round(center + half)), size_px)
            if is_row:
                canvas[lo:hi, :] = STRIP_LINE_VAL
            else:
                canvas[:, lo:hi] = STRIP_LINE_VAL
    return canvas


def _zone_grid(size_px: int, mat_size_nm: float, strip_width_nm: float):
    """Compute alternating [mat, strip, mat, strip, ...] spans covering size_px."""
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


def generate_zone_canvas(
    size_px: int,
    kind: str,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    mat_size_nm: float = 2600.0,
    strip_width_nm: float = 320.0,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
) -> dict:
    """Tile independently-generated mats of `kind` across the canvas,
    separated by strip material. Returns dict with the canvas plus mat/strip
    rectangles in px for crop-placement logic downstream.
    """
    generator = _GENERATORS[kind]
    presets = presets_for_kind(kind)

    canvas = _strip_routing_texture(size_px, rng)

    row_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)
    col_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)

    mat_rects = []
    strip_rects = []

    for row_is_mat, y0, y1 in row_spans:
        for col_is_mat, x0, x1 in col_spans:
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                mat_h, mat_w = y1 - y0, x1 - x0
                preset = presets[int(rng.integers(0, len(presets)))]
                child_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
                mat_size = max(mat_h, mat_w)
                mat_canvas = generator(
                    mat_size, preset, collapse_threshold_nm, child_rng,
                    linewidth_bias_nm=linewidth_bias_nm, corner_rounding_px=corner_rounding_px,
                )
                canvas[y0:y1, x0:x1] = mat_canvas[:mat_h, :mat_w]
                mat_rects.append((x0, y0, mat_w, mat_h))
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))

    return {"canvas": canvas, "mat_rects": mat_rects, "strip_rects": strip_rects}

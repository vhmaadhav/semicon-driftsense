"""
DRAM-style structure generator.

Draws a folded-bitline 6F^2 cell array: horizontal word lines, vertical bit
lines, and a checkerboard of storage-node contacts (one contact per 2 cells,
matching real folded-bitline layouts rather than a naive full grid). Runs at
1 nm/px, so nanometer preset values map 1:1 to pixel offsets.

Rendering is vectorized (1D row/column masks broadcast across the canvas)
so it stays fast even at the 10000x10000 px fine-canvas size used for a
single sample.
"""

import cv2
import numpy as np

from src.structural_defects import maybe_collapse_gap

BACKGROUND = 40
WORD_LINE_VAL = 150
BIT_LINE_VAL = 170
CONTACT_VAL = 225

POSITION_JITTER_NM = 1.5
# Per-instance linewidth variation as a fraction of nominal width (CD variation
# / line-edge roughness proxy) -- "scaling of polygons" knob: each drawn
# element gets its own slightly perturbed size rather than being identical.
WIDTH_JITTER_FRACTION = 0.10


def _line_positions(size_px: int, pitch_nm: float, rng: np.random.Generator) -> np.ndarray:
    positions = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, POSITION_JITTER_NM)
    return np.array(positions)


def _line_mask(
    size_px: int,
    positions: np.ndarray,
    width_nm: float,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    width_jitter_fraction: float = WIDTH_JITTER_FRACTION,
    linewidth_bias_nm: float = 0.0,
) -> np.ndarray:
    """1D boolean mask marking line + any bridged (collapsed) gaps.

    `linewidth_bias_nm` is a deterministic global CD bias (simulates
    over/under-exposure or etch bias) applied on top of the per-instance
    random jitter -- positive grows every line, negative shrinks them.
    """
    mask = np.zeros(size_px, dtype=bool)
    biased_width_nm = max(width_nm + linewidth_bias_nm, 1.0)
    widths = biased_width_nm * (1.0 + rng.normal(0, width_jitter_fraction, size=len(positions)))
    widths = np.clip(widths, biased_width_nm * 0.5, biased_width_nm * 1.5)
    for i, center in enumerate(positions):
        half_w = widths[i] / 2.0
        lo = int(round(center - half_w))
        hi = int(round(center + half_w))
        mask[max(lo, 0):min(hi, size_px)] = True

        if i + 1 < len(positions):
            next_center = positions[i + 1]
            next_half_w = widths[i + 1] / 2.0
            gap_nm = (next_center - next_half_w) - (center + half_w)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                bridge_lo = int(round(center + half_w))
                bridge_hi = int(round(next_center - next_half_w))
                mask[max(bridge_lo, 0):min(bridge_hi, size_px)] = True
    return mask


def generate_dram_canvas(
    size_px: int,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
    return_layers: bool = False,
):
    """Render the DRAM cell array.

    By default returns just the flattened grayscale composite (what a real
    SEM would actually image -- it only sees the top exposed surface, not a
    color-coded stack). With `return_layers=True`, also returns a dict of
    the individual process layers (substrate, word_line, bit_line,
    storage_contact) as separate masks -- useful for a teaching "exploded
    stack" visualization or for QA'ing one layer at a time. The flattened
    composite is bit-identical either way; layers are purely additive info.
    """
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)

    word_positions = _line_positions(size_px, preset["word_line_pitch_nm"], rng)
    bit_positions = _line_positions(size_px, preset["bit_line_pitch_nm"], rng)

    row_mask = _line_mask(
        size_px, word_positions, preset["word_line_width_nm"], collapse_threshold_nm, rng,
        linewidth_bias_nm=linewidth_bias_nm,
    )
    col_mask = _line_mask(
        size_px, bit_positions, preset["bit_line_width_nm"], collapse_threshold_nm, rng,
        linewidth_bias_nm=linewidth_bias_nm,
    )

    word_line_layer = np.zeros((size_px, size_px), dtype=np.uint8)
    word_line_layer[row_mask, :] = WORD_LINE_VAL
    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], WORD_LINE_VAL)

    bit_line_layer = np.zeros((size_px, size_px), dtype=np.uint8)
    bit_line_layer[:, col_mask] = BIT_LINE_VAL
    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], BIT_LINE_VAL)

    contact_layer = np.zeros((size_px, size_px), dtype=np.uint8)
    base_radius = max(preset["contact_diameter_nm"] + linewidth_bias_nm, 1.0) / 2.0
    for i, wl in enumerate(word_positions):
        for j, bl in enumerate(bit_positions):
            if (i + j) % 2 == 0:
                radius = max(1, int(round(base_radius * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)))))
                cv2.circle(canvas, (int(round(bl)), int(round(wl))), radius, CONTACT_VAL, -1)
                cv2.circle(contact_layer, (int(round(bl)), int(round(wl))), radius, CONTACT_VAL, -1)

    if corner_rounding_px >= 0.5:
        k = max(1, int(round(corner_rounding_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)
        if return_layers:
            for layer in (word_line_layer, bit_line_layer, contact_layer):
                layer[:] = cv2.morphologyEx(
                    cv2.morphologyEx(layer, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel
                )

    if return_layers:
        layers = {
            "substrate": np.full((size_px, size_px), BACKGROUND, dtype=np.uint8),
            "word_line": word_line_layer,
            "bit_line": bit_line_layer,
            "storage_contact": contact_layer,
        }
        return canvas, layers

    return canvas

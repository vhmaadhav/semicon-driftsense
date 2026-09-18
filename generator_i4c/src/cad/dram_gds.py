"""
DRAM CAD (GDSII) geometry builder -- Phase 3.

Builds a real multi-layer gdstk.Cell for a DRAM mat, on 8 numbered layers
(the render step derives brightness from the layer *number* via
src/cad/yield_model.py -- nothing here assigns a grayscale value):

  0  active/diffusion    -- islands between word lines and bit lines
  1  word_line (poly gate)
  2  bit_line_contact    -- diffusion-to-bitline via
  3  bit_line_metal
  4  storage_contact     -- storage-node landing pad
  5  storage_capacitor   -- the actual memory element, above its landing pad
  6  via1                -- periodic bitline-to-strap via
  7  metal2_strap        -- periodic strap over a subset of bitlines
                             (real DRAMs strap M1 bitlines with a wider,
                             sparser M2 to cut resistance)

This builds the *design* -- the ideal, as-drawn geometry a student's
Reference GDS comes from. CD bias, corner rounding, and polygon-scale
outliers are NOT applied here: those are fabrication/imaging artifacts
(see src/cad/fab_distortion.py), and a design database never encodes how a
particular fab run happened to come out -- only the Search-side render sees
them. This mirrors src/patterns/dram.py's geometry (same preset fields,
same position-jitter/collapse-bridging model) but emits vector polygons
instead of a raster mask, since the deliverable is now an actual .gds file.
Illustrative of real folded-bitline DRAM topology, not an exact fab layout.
"""

from __future__ import annotations

import gdstk
import numpy as np

from src.structural_defects import maybe_collapse_gap
from src.cad.shapes import tablet

LAYER_ACTIVE = 0
LAYER_WORD_LINE = 1
LAYER_BIT_LINE_CONTACT = 2
LAYER_BIT_LINE_METAL = 3
LAYER_STORAGE_CONTACT = 4
LAYER_STORAGE_CAPACITOR = 5
LAYER_VIA1 = 6
LAYER_METAL2_STRAP = 7
NUM_LAYERS = 8

POSITION_JITTER_NM = 1.5
WIDTH_JITTER_FRACTION = 0.10

STRAP_PERIOD = 4            # one metal2 strap every 4 bit lines
STRAP_WIDTH_FACTOR = 2.5    # metal2 strap width, relative to bit line width
CAPACITOR_FILL_FACTOR = 1.8  # capacitor radius, relative to the storage via's


def _positions(size_nm: float, pitch_nm: float, rng: np.random.Generator) -> np.ndarray:
    positions = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_nm:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, POSITION_JITTER_NM)
    return np.array(positions)


def _jittered_widths(n: int, width_nm: float, rng: np.random.Generator) -> np.ndarray:
    widths = width_nm * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION, size=n))
    return np.clip(widths, width_nm * 0.5, width_nm * 1.5)


def build_dram_mat_gds(
    size_nm: float,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    contact_aspect_ratio: float = 1.6,
    contact_thickness_factor: float = 1.0,
    contact_angle_deg: float = 90.0,
) -> gdstk.Cell:
    cell = gdstk.Cell(f"DRAM_MAT_{rng.integers(0, 2**31 - 1)}")

    word_positions = _positions(size_nm, preset["word_line_pitch_nm"], rng)
    bit_positions = _positions(size_nm, preset["bit_line_pitch_nm"], rng)
    word_widths = _jittered_widths(len(word_positions), preset["word_line_width_nm"], rng)
    bit_widths = _jittered_widths(len(bit_positions), preset["bit_line_width_nm"], rng)

    # Layer 0: active/diffusion islands, one per (word-gap x bit-gap) cell --
    # isolated by STI from neighboring cells in both directions, matching
    # real DRAM active-area topology (not a full-width strip: that would
    # necessarily touch every bit line crossing it). "Always big" and
    # "never touching a neighbor" are geometrically in tension the moment
    # you try to grow past the natural gap -- the gap's edge *is* the
    # neighboring line's edge, so growing beyond it always means overlap.
    # ACTIVE_FILL_FRACTION resolves that: each island fills a large but
    # strictly-interior fraction of its available cell footprint, so it's
    # as big as possible while guaranteeing clearance on all four sides.
    ACTIVE_FILL_FRACTION = 0.88
    for i in range(len(word_positions) - 1):
        lo_w = word_positions[i] + word_widths[i] / 2.0
        hi_w = word_positions[i + 1] - word_widths[i + 1] / 2.0
        if hi_w <= lo_w:
            continue
        cy = (lo_w + hi_w) / 2.0
        ry = (hi_w - lo_w) / 2.0 * ACTIVE_FILL_FRACTION
        for j in range(len(bit_positions) - 1):
            lo_b = bit_positions[j] + bit_widths[j] / 2.0
            hi_b = bit_positions[j + 1] - bit_widths[j + 1] / 2.0
            if hi_b <= lo_b:
                continue
            cx = (lo_b + hi_b) / 2.0
            rx = (hi_b - lo_b) / 2.0 * ACTIVE_FILL_FRACTION
            cell.add(gdstk.rectangle((cx - rx, cy - ry), (cx + rx, cy + ry), layer=LAYER_ACTIVE))

    # Layer 1: word lines (+ collapse bridging between adjacent word lines)
    for i, y in enumerate(word_positions):
        half = word_widths[i] / 2.0
        cell.add(gdstk.rectangle((0, y - half), (size_nm, y + half), layer=LAYER_WORD_LINE))
        if i + 1 < len(word_positions):
            next_half = word_widths[i + 1] / 2.0
            gap_nm = (word_positions[i + 1] - next_half) - (y + half)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                cell.add(gdstk.rectangle((0, y + half), (size_nm, word_positions[i + 1] - next_half), layer=LAYER_WORD_LINE))

    # Layer 3: bit lines (+ collapse bridging)
    for j, x in enumerate(bit_positions):
        half = bit_widths[j] / 2.0
        cell.add(gdstk.rectangle((x - half, 0), (x + half, size_nm), layer=LAYER_BIT_LINE_METAL))
        if j + 1 < len(bit_positions):
            next_half = bit_widths[j + 1] / 2.0
            gap_nm = (bit_positions[j + 1] - next_half) - (x + half)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                cell.add(gdstk.rectangle((x + half, 0), (bit_positions[j + 1] - next_half, size_nm), layer=LAYER_BIT_LINE_METAL))

    # Layers 2, 4 & 5: checkerboard vias (storage contact / bitline contact
    # alternate, one per cell, matching real folded-bitline topology) plus
    # the storage capacitor sitting above each storage contact. Contacts
    # and vias are tablet (capsule) shaped, not circles or squares -- see
    # src/cad/shapes.py.
    base_r = preset["contact_diameter_nm"] / 2.0
    for i, wy in enumerate(word_positions):
        for j, bx in enumerate(bit_positions):
            r = max(base_r * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)), 1.0)
            is_storage = (i + j) % 2 == 0
            layer = LAYER_STORAGE_CONTACT if is_storage else LAYER_BIT_LINE_CONTACT
            cell.add(tablet(bx, wy, r, contact_aspect_ratio, contact_thickness_factor, contact_angle_deg, layer))
            if is_storage:
                cap_r = r * CAPACITOR_FILL_FACTOR
                cell.add(tablet(bx, wy, cap_r, contact_aspect_ratio, contact_thickness_factor, contact_angle_deg, LAYER_STORAGE_CAPACITOR))

    # Layers 6 & 7: via1 + metal2 strap -- periodic reinforcement over a
    # sparse subset of bit lines, the way a real DRAM straps M1 bitlines
    # with a wider, coarser-pitch M2 to cut resistance.
    strap_half_width = (preset["bit_line_width_nm"] * STRAP_WIDTH_FACTOR) / 2.0
    via1_r = base_r * 0.6
    for j, bx in enumerate(bit_positions):
        if j % STRAP_PERIOD != 0:
            continue
        cell.add(gdstk.rectangle((bx - strap_half_width, 0), (bx + strap_half_width, size_nm), layer=LAYER_METAL2_STRAP))
        for wy in word_positions:
            cell.add(tablet(bx, wy, via1_r, contact_aspect_ratio, contact_thickness_factor, contact_angle_deg, LAYER_VIA1))

    return cell

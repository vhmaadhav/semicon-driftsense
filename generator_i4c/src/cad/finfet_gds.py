"""
FinFET CAD (GDSII) geometry builder -- Phase 3. See dram_gds.py for the
overall approach (vector polygons, layer number -> yield, not a hardcoded
grayscale per layer) and for why CD bias / corner rounding / polygon-scale
outliers are NOT applied here -- this builds the ideal, as-drawn design;
those are fabrication/imaging artifacts applied only to the Search-side
render (see src/cad/fab_distortion.py).

8 numbered layers:
  0  fin (active)
  1  gate (poly)
  2  spacer      -- flanks each gate on both sides
  3  contact     -- source/drain contact
  4  via0        -- contact-to-metal1 via
  5  metal1      -- M1 routing over each contact row
  6  via1        -- metal1-to-metal2 via
  7  metal2      -- routed orthogonal to M1 (vertical bars vs. M1's
                    horizontal bands), real BEOL alternates routing
                    direction layer to layer

Illustrative of real FinFET process ordering (fin -> gate -> spacer ->
contact -> BEOL), not an exact fab layout.
"""

from __future__ import annotations

import gdstk
import numpy as np

from src.structural_defects import maybe_collapse_gap
from src.cad.shapes import tablet

LAYER_FIN = 0
LAYER_GATE = 1
LAYER_SPACER = 2
LAYER_CONTACT = 3
LAYER_VIA0 = 4
LAYER_METAL1 = 5
LAYER_VIA1 = 6
LAYER_METAL2 = 7
NUM_LAYERS = 8

POSITION_JITTER_NM = 1.0
WIDTH_JITTER_FRACTION = 0.10
SPACER_WIDTH_FRACTION = 0.15  # spacer width as a fraction of gate_length_nm
METAL1_HEIGHT_FRACTION = 0.35  # metal1 band half-height as a fraction of contact half-size

METAL2_PERIOD = 3           # one metal2 bar every 3 fin columns
METAL2_WIDTH_FACTOR = 2.5  # metal2 bar width, relative to fin width


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


def build_finfet_mat_gds(
    size_nm: float,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    contact_aspect_ratio: float = 1.6,
    contact_thickness_factor: float = 1.0,
    contact_angle_deg: float = 90.0,
) -> gdstk.Cell:
    cell = gdstk.Cell(f"FINFET_MAT_{rng.integers(0, 2**31 - 1)}")

    fin_positions = _positions(size_nm, preset["fin_pitch_nm"], rng)
    gate_positions = _positions(size_nm, preset["gate_pitch_nm"], rng)
    fin_widths = _jittered_widths(len(fin_positions), preset["fin_width_nm"], rng)
    gate_widths = _jittered_widths(len(gate_positions), preset["gate_length_nm"], rng)

    # Layer 0: fins (+ collapse bridging between adjacent fins)
    for i, x in enumerate(fin_positions):
        half = fin_widths[i] / 2.0
        cell.add(gdstk.rectangle((x - half, 0), (x + half, size_nm), layer=LAYER_FIN))
        if i + 1 < len(fin_positions):
            next_half = fin_widths[i + 1] / 2.0
            gap_nm = (fin_positions[i + 1] - next_half) - (x + half)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                cell.add(gdstk.rectangle((x + half, 0), (fin_positions[i + 1] - next_half, size_nm), layer=LAYER_FIN))

    # Layers 1 & 2: gates + flanking spacers
    spacer_width = preset["gate_length_nm"] * SPACER_WIDTH_FRACTION
    for j, y in enumerate(gate_positions):
        half = gate_widths[j] / 2.0
        cell.add(gdstk.rectangle((0, y - half), (size_nm, y + half), layer=LAYER_GATE))
        cell.add(gdstk.rectangle((0, y - half - spacer_width), (size_nm, y - half), layer=LAYER_SPACER))
        cell.add(gdstk.rectangle((0, y + half), (size_nm, y + half + spacer_width), layer=LAYER_SPACER))
        if j + 1 < len(gate_positions):
            next_half = gate_widths[j + 1] / 2.0
            gap_nm = (gate_positions[j + 1] - next_half) - (y + half)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                cell.add(gdstk.rectangle((0, y + half), (size_nm, gate_positions[j + 1] - next_half), layer=LAYER_GATE))

    # Layers 3, 4 & 5: source/drain contacts (checkerboard, one per 2
    # fin/gap cells), their via0 up to M1, and an M1 routing line over each
    # contact row. Contacts/vias are tablet (capsule) shaped -- see
    # src/cad/shapes.py.
    base_half = preset["contact_size_nm"] / 2.0
    via0_r = base_half * 0.6
    for j in range(len(gate_positions) - 1):
        row_mid_y = (gate_positions[j] + gate_positions[j + 1]) / 2.0
        row_has_contact = False
        for i, fx in enumerate(fin_positions):
            if (i + j) % 2 == 0:
                half = max(base_half * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)), 1.0)
                cell.add(tablet(fx, row_mid_y, half, contact_aspect_ratio, contact_thickness_factor, contact_angle_deg, LAYER_CONTACT))
                cell.add(tablet(fx, row_mid_y, via0_r, contact_aspect_ratio, contact_thickness_factor, contact_angle_deg, LAYER_VIA0))
                row_has_contact = True
        if row_has_contact:
            m1_half = base_half * METAL1_HEIGHT_FRACTION
            cell.add(gdstk.rectangle((0, row_mid_y - m1_half), (size_nm, row_mid_y + m1_half), layer=LAYER_METAL1))

    # Layers 6 & 7: via1 + metal2 -- routed orthogonal to M1 (vertical bars
    # instead of horizontal bands), periodic reinforcement over a sparse
    # subset of fin columns, mirroring real BEOL layer-to-layer routing
    # direction alternation.
    m2_half_width = (preset["fin_width_nm"] * METAL2_WIDTH_FACTOR) / 2.0
    via1_r = base_half * 0.5
    for i, fx in enumerate(fin_positions):
        if i % METAL2_PERIOD != 0:
            continue
        cell.add(gdstk.rectangle((fx - m2_half_width, 0), (fx + m2_half_width, size_nm), layer=LAYER_METAL2))
        for j in range(len(gate_positions) - 1):
            row_mid_y = (gate_positions[j] + gate_positions[j + 1]) / 2.0
            cell.add(tablet(fx, row_mid_y, via1_r, contact_aspect_ratio, contact_thickness_factor, contact_angle_deg, LAYER_VIA1))

    return cell

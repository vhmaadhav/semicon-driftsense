"""
Fabrication/imaging distortion for Phase 3 -- applied *only* to the copy of
the CAD geometry that feeds the Search-side render, never to the design
itself. The Reference GDS a student receives is always the exact, as-drawn
design intent (see dram_gds.py / finfet_gds.py); a design database doesn't
encode how a particular fab run or imaging session happened to come out.

Each effect here models something that happens *between* "as designed" and
"what the SEM actually images":
  - per-polygon size outliers -- local CD scatter from etch/litho loading
  - a deterministic global CD/etch bias -- systematic over/under-exposure
  - corner rounding -- real litho/etch never draws a perfectly sharp corner
"""

from __future__ import annotations

import gdstk
import numpy as np


def apply_fab_distortion(
    design_cell: gdstk.Cell,
    num_layers: int,
    rng: np.random.Generator,
    polygon_scale_prob: float = 0.0,
    polygon_scale_range: float = 0.0,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
) -> gdstk.Cell:
    """Return a NEW cell built from `design_cell`'s polygons with
    fabrication-style distortion applied. `design_cell` is never mutated --
    the Reference (and anything else built from it) keeps rendering the
    exact, undistorted design.
    """
    out = gdstk.Cell(f"{design_cell.name}_FAB")
    for layer in range(num_layers):
        polygons = design_cell.get_polygons(layer=layer, datatype=0)  # copies
        if not polygons:
            continue

        if polygon_scale_prob > 0 and polygon_scale_range > 0:
            for poly in polygons:
                if rng.random() < polygon_scale_prob:
                    factor = 1.0 + rng.uniform(-polygon_scale_range, polygon_scale_range)
                    (xmin, ymin), (xmax, ymax) = poly.bounding_box()
                    poly.scale(factor, center=((xmin + xmax) / 2.0, (ymin + ymax) / 2.0))

        if abs(linewidth_bias_nm) >= 1e-9 and polygons:
            polygons = gdstk.offset(polygons, linewidth_bias_nm / 2.0, layer=layer, datatype=0)

        if corner_rounding_px >= 0.5:
            for poly in polygons:
                poly.fillet(corner_rounding_px)

        for poly in polygons:
            out.add(poly)
    return out

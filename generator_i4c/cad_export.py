"""Side-effect-free helpers shared by the dataset writers.

Importing this module changes nothing in the upstream pipeline -- unlike
generate_cad_varied.py, which swaps in a crash-safe corner-rounding step at
import time. export_bookmarks.py imports from here so the organizer's curated
samples are rendered with upstream's exact code.
"""

import gdstk
import numpy as np


def full_canvas_cell(mats: list, num_layers: int) -> gdstk.Cell:
    """The whole search-side CAD, exactly as app.py's _full_canvas_cell builds
    it for the organizer's export (design geometry, canvas coordinates, no
    strip routing texture). Reimplemented, not imported: app.py pulls in
    Streamlit at import time."""
    out = gdstk.Cell("SEARCH_SIDE_CAD")
    for mat in mats:
        mx0, my0 = mat["x0"], mat["y0"]
        for layer in range(num_layers):
            for poly in mat["design_cell"].get_polygons(layer=layer, datatype=0):
                poly.translate(mx0, my0)
                out.add(poly)
    return out


def applied_rotation_deg(geometry: dict, params) -> float:
    """The angle render_cad_sample applies, from the pipeline's own stream
    (np.random.default_rng(strip_rng_seed + 2).uniform(-r, r)); the pipeline
    draws it but never returns it."""
    if params.search_rotation_deg <= 0:
        return 0.0
    rot_rng = np.random.default_rng(geometry["strip_rng_seed"] + 2)
    return float(rot_rng.uniform(-params.search_rotation_deg, params.search_rotation_deg))

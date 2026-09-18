#!/usr/bin/env python3
"""Phase 3 baseline: the Reference is a .gds CAD file, not an image, so it
has to be rendered to a raster before any image-matching technique can be
applied at all -- this is the step Phase 1/2's infer.py didn't need.

Rendering reads layer numbers directly out of the GDS and derives their
brightness from src/cad/yield_model.py (the same rule used to generate the
data), which is why this works on a Reference from either architecture
without being told which one: the render doesn't care what a layer is
*called*, only its number in the stack.

Once rendered, matching is identical to Phase 1/2 -- reuses zncc_match()
unmodified.

Example:
    python baseline_solution/cad_infer.py --reference ref.gds --search search.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import gdstk
import numpy as np

from src.cad.render import rasterize_cell
from baseline_solution.zncc import zncc_match, shift_report

REFERENCE_SIZE_PX = 1000  # fixed by the problem statement, same as Phase 1/2


def render_reference_gds(gds_path: str) -> np.ndarray:
    lib = gdstk.read_gds(gds_path)
    cells = lib.top_level()
    if not cells:
        raise ValueError(f"{gds_path} has no cells")
    cell = cells[0]
    polygons = cell.get_polygons()
    if not polygons:
        raise ValueError(f"{gds_path} has no polygons")
    num_layers = max(p.layer for p in polygons) + 1
    return rasterize_cell(cell, REFERENCE_SIZE_PX, num_layers)


def predict(gds_path: str, search_path: str) -> dict:
    reference = render_reference_gds(gds_path)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    if search is None:
        raise ValueError(f"Could not read search image: {search_path}")
    return zncc_match(reference, search)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, help="path to reference .gds")
    p.add_argument("--search", required=True)
    p.add_argument("--gt-x", type=float, default=None)
    p.add_argument("--gt-y", type=float, default=None)
    args = p.parse_args()

    match = predict(args.reference, args.search)
    print(f"predicted: x={match['x']:.2f} y={match['y']:.2f} zncc_score={match['score']:.4f} scale={match['scale']}")

    if args.gt_x is not None and args.gt_y is not None:
        shift = shift_report(match["x"], match["y"], args.gt_x, args.gt_y)
        print(f"shift (predicted - actual): dx={shift['dx']:+.2f} dy={shift['dy']:+.2f} "
              f"distance={shift['distance_px']:.2f} px")


if __name__ == "__main__":
    main()

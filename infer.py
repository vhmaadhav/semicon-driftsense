#!/usr/bin/env python3
"""Legacy Phase 1 single-pair compatibility CLI.

Phase 2 is the canonical Drift-Sense interface and is run with::

    python register.py --input pairs.csv --output predictions.csv

This file remains only for Phase 1 compatibility and historical experiments.
It accepts one fixed-pose Reference/Search pair and prints ``x,y``. Shared
runtime helpers live in ``driftsense.runtime`` so the Phase 2 submission path
does not depend on this legacy CLI.

Usage
-----
    python infer.py --reference path/to/reference.png --search path/to/search.png
    python infer.py reference.png search.png
    python infer.py --reference r.png --search s.png --json

Weights load from ``weights/driftsense.pt``. If the learned model cannot load,
the compatibility CLI falls back to classical multi-scale ZNCC.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# Re-exported for compatibility with older scripts/tests that imported these
# helpers from infer.py. New code should import driftsense.runtime directly.
from driftsense.runtime import (  # noqa: E402,F401
    DEFAULT_WEIGHTS,
    load_model,
    read_gray,
    zncc_fallback,
)
from driftsense.policy import ROUTE_THRESHOLD  # noqa: E402,F401


def predict(reference_path: str, search_path: str, weights_path: str = DEFAULT_WEIGHTS,
            want_heatmap: bool = False, tta: bool = True,
            route_threshold: float = ROUTE_THRESHOLD) -> dict:
    reference = read_gray(reference_path)
    search = read_gray(search_path)

    loaded = load_model(weights_path)
    if loaded is None:
        return zncc_fallback(reference, search)

    model, device = loaded

    # Historical Phase 1 decode. Phase 2 uses register.py + locate_phase2.
    from driftsense.policy import predict_policy
    return predict_policy(model, reference, search, device, tta=tta,
                          want_heatmap=want_heatmap,
                          route_threshold=route_threshold)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("positional", nargs="*", metavar="REFERENCE SEARCH",
                   help="reference and search image paths (alternative to the flags)")
    p.add_argument("--reference", "-r", help="path to the reference image (1 nm/px)")
    p.add_argument("--search", "-s", help="path to the search image (10 nm/px)")
    p.add_argument("--weights", "-w", default=DEFAULT_WEIGHTS)
    p.add_argument("--json", action="store_true", help="print a JSON object instead of x,y")
    p.add_argument("--save-heatmap", default="",
                   help="optional path to write the response map (implies --no-tta, "
                        "since TTA aggregates over eight views)")
    p.add_argument("--no-tta", action="store_true",
                   help="single view instead of 8-way dihedral voting")
    args = p.parse_args()

    ref, sea = args.reference, args.search
    if ref is None or sea is None:
        if len(args.positional) == 2:
            ref, sea = args.positional
        else:
            p.error("provide --reference and --search (or two positional paths)")
    args.reference, args.search = ref, sea
    return args


def main():
    args = parse_args()
    res = predict(args.reference, args.search, args.weights,
                  want_heatmap=bool(args.save_heatmap), tta=not args.no_tta)

    if args.save_heatmap and "heatmap" in res:
        hm = res["heatmap"]
        hm = (255 * (hm - hm.min()) / max(float(np.ptp(hm)), 1e-9)).astype(np.uint8)
        cv2.imwrite(args.save_heatmap, cv2.applyColorMap(hm, cv2.COLORMAP_INFERNO))

    if args.json:
        print(json.dumps({k: v for k, v in res.items() if k != "heatmap"}))
    else:
        print(f"{res['x']:.2f},{res['y']:.2f}")


if __name__ == "__main__":
    main()

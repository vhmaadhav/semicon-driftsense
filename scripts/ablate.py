#!/usr/bin/env python3
"""Ablate the inference-time decisions on the validation split.

Three things in the decode path were chosen on reasoning rather than
measurement, so they get measured here before being baked into the shipped
defaults:

  refine       -- does the ZNCC sub-pixel snap actually help, or does it drag
                  predictions onto neighbouring repeats?
  tie_tol      -- the problem statement says to prefer the centre-most match
                  among ties, but the true location is uniform over the frame,
                  so too generous a tie window will actively cost accuracy.
  accept_px    -- how far the snap is allowed to move the network's choice.

    python scripts/ablate.py --split data/val --limit 150
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.dataset import load_manifest  # noqa: E402
from driftsense.matching import locate  # noqa: E402
from driftsense.model import DriftSenseNet  # noqa: E402


def run(model, device, rows, split, **kw) -> dict:
    d = []
    for r in rows:
        ref = cv2.imread(os.path.join(split, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        sea = cv2.imread(os.path.join(split, r["search_path"]), cv2.IMREAD_GRAYSCALE)
        res = locate(model, ref, sea, device, **kw)
        d.append(np.hypot(res["x"] - float(r["gt_x_corr"]), res["y"] - float(r["gt_y_corr"])))
    d = np.array(d)
    return {"median": float(np.median(d)), "mean": float(d.mean()),
            "acc1": float((d <= 1).mean()), "acc2": float((d <= 2).mean()),
            "acc5": float((d <= 5).mean()), "acc10": float((d <= 10).mean())}


def line(name, s):
    print(f"  {name:<34} median {s['median']:7.3f}  mean {s['mean']:8.2f}  "
          f"acc@1 {s['acc1']:.3f}  acc@2 {s['acc2']:.3f}  "
          f"acc@5 {s['acc5']:.3f}  acc@10 {s['acc10']:.3f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="data/val")
    p.add_argument("--weights", default="weights/driftsense.pt")
    p.add_argument("--limit", type=int, default=150)
    args = p.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = DriftSenseNet()
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()

    rows = load_manifest(args.split)[:args.limit]
    print(f"{len(rows)} samples from {args.split}\n")

    print("ZNCC sub-pixel refinement:")
    line("off (heatmap + offset head only)", run(model, device, rows, args.split, refine=False))
    line("on  (default)", run(model, device, rows, args.split, refine=True))

    print("\ntie-break window (centre-preference among near-equal peaks):")
    for tol in (0.0, 0.01, 0.02, 0.04, 0.10):
        line(f"tie_tol = {tol:.2f}", run(model, device, rows, args.split, tie_tol=tol))

    print("\nrefinement search radius / accept threshold:")
    for rad, acc in ((4, 10.0), (8, 10.0), (12, 10.0), (8, 5.0), (8, 20.0)):
        line(f"radius {rad:2d} px, accept {acc:4.1f} px",
             run(model, device, rows, args.split, refine_radius=rad, refine_accept_px=acc))


if __name__ == "__main__":
    main()

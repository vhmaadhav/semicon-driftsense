#!/usr/bin/env python3
"""Render a localisation result: reference, search with boxes, and heatmap.

    python scripts/visualize.py --split data/test --id 0 --out results/vis_000.png
    python scripts/visualize.py --split data/test --ids 0 1 2 3 --out results/panel.png

Green box = ground truth, red = prediction. The heatmap panel is what makes
the periodicity visible: on a hard sample it shows a lattice of near-equal
decoy peaks with the true site only slightly brighter.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.dataset import load_manifest  # noqa: E402
from driftsense.matching import locate, locate_tta, zncc_only  # noqa: E402
from driftsense.model import DriftSenseNet, TEMPLATE_SIZE  # noqa: E402


def draw(ax, img, title):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def box(ax, cx, cy, color, label, pad=0.0, style="-"):
    """`pad` insets/outsets the box so a correct prediction does not completely
    hide the ground-truth box underneath it."""
    size = TEMPLATE_SIZE + 2 * pad
    ax.add_patch(plt.Rectangle((cx - size / 2, cy - size / 2), size, size,
                               fill=False, edgecolor=color, linewidth=1.6,
                               linestyle=style, label=label))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="data/test")
    p.add_argument("--ids", type=int, nargs="+", default=None)
    p.add_argument("--id", type=int, default=0)
    p.add_argument("--weights", default="weights/driftsense.pt")
    p.add_argument("--out", default="results/visualization.png")
    args = p.parse_args()

    ids = args.ids if args.ids is not None else [args.id]
    rows = load_manifest(args.split)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = DriftSenseNet()
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()

    fig, axes = plt.subplots(len(ids), 3, figsize=(13, 4.3 * len(ids)), squeeze=False)

    for k, idx in enumerate(ids):
        r = rows[idx]
        ref = cv2.imread(os.path.join(args.split, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        sea = cv2.imread(os.path.join(args.split, r["search_path"]), cv2.IMREAD_GRAYSCALE)
        gx, gy = float(r["gt_x_corr"]), float(r["gt_y_corr"])

        # Heatmap comes from a single view (TTA aggregates eight, so there is
        # no one map to show); the reported error is the shipped TTA decode, so
        # the figure matches the numbers in the README.
        res = locate(model, ref, sea, device, refine=True, return_heatmap=True)
        res_tta = locate_tta(model, ref, sea, device, refine=True)
        z = zncc_only(ref, sea)
        err = float(np.hypot(res_tta["x"] - gx, res_tta["y"] - gy))
        err_sv = float(np.hypot(res["x"] - gx, res["y"] - gy))
        zerr = float(np.hypot(z["x"] - gx, z["y"] - gy))

        draw(axes[k][0], ref, f"Reference 1000x1000 @1nm/px\n{r['architecture']}")

        draw(axes[k][1], sea, f"Search 1000x1000 @10nm/px\n"
                              f"ours {err:.2f}px (TTA)  |  1-view {err_sv:.2f}px  |  ZNCC {zerr:.1f}px")
        box(axes[k][1], gx, gy, "#00ff66", "ground truth", pad=7, style="--")
        box(axes[k][1], res_tta["x"], res_tta["y"], "#ff3333", "prediction")
        axes[k][1].legend(loc="lower right", fontsize=7, framealpha=0.7)

        hm = res["heatmap"]
        axes[k][2].imshow(hm, cmap="inferno")
        axes[k][2].set_title(f"Response heatmap {hm.shape[0]}x{hm.shape[1]}\n"
                             f"single-view peak {hm.max():.3f}", fontsize=9)
        axes[k][2].axis("off")

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()

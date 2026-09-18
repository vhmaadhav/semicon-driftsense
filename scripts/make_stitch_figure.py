#!/usr/bin/env python3
"""Render docs/how-it-works.png -- what the pipeline does, on real pairs.

Every panel is an original image: the search frames are the generator's own
PNGs, the CAD panels are drawn from the reference `.gds` the pairs file names,
and the boxes are this repo's actual predictions, not illustrations.

The column order follows the organizer's own description of Phase 3 --
"receive the raw CAD polygons, infer the intermediate yield raster (one grey
level per layer), and match it to the real SEM image":

    reference -> yield raster -> search SEM -> result

Phase 2's reference is already an SEM image, so its second panel is the same
template the matcher correlates rather than a rendered one.

    python scripts/make_stitch_figure.py --out docs/how-it-works.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from driftsense import gds  # noqa: E402

# One colour per design layer, dark -> light with depth, so a stack of eight
# reads as a stack rather than a blur.
LAYER_COLOURS = ["#2b3a67", "#3f6fb5", "#4a9fd8", "#5ec2b7",
                 "#8ed081", "#d9c25f", "#e08b4a", "#c0504d"]

BOX_PRED = "#00e5ff"
BOX_TRUE = "#ff4fa3"


def _read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _index(rows, key="pair_id"):
    return {r[key]: r for r in rows}


def cad_polygons_panel(ax, gds_path, size_px, nm_per_px):
    """The raw design: polygon outlines, one colour per layer, nothing filled.

    This is what arrives in `reference_gds_path` -- no brightness of any kind.
    """
    polys_by_layer, num_layers = gds.read_gds_layers(gds_path)
    ax.set_facecolor("#0d1117")
    for layer in sorted(polys_by_layer):
        colour = LAYER_COLOURS[layer % len(LAYER_COLOURS)]
        for poly in polys_by_layer[layer]:
            p = np.asarray(poly, float) / nm_per_px
            ax.fill(p[:, 0], p[:, 1], facecolor=colour, edgecolor=colour,
                    linewidth=0.4, alpha=0.55)
    ax.set_xlim(0, size_px)
    ax.set_ylim(size_px, 0)
    ax.set_aspect("equal")
    return num_layers


def grey_panel(ax, img):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")


def box(ax, cx, cy, side, colour, label, dashed=False):
    """Ground truth is drawn as a thick halo, the prediction as a thin line on
    top. At these errors -- hundredths of a pixel -- two equal-weight outlines
    land on the same screen pixels and the reader sees only one box, which
    reads as a missing annotation rather than as agreement."""
    ax.add_patch(Rectangle((cx - side / 2.0, cy - side / 2.0), side, side,
                           fill=False, edgecolor=colour,
                           linewidth=4.0 if dashed else 1.4,
                           alpha=0.9 if dashed else 1.0,
                           linestyle="-", label=label, zorder=3 if dashed else 4))


def _corner_inset(ax, img, gx, gy, px, py, side, span=6.0):
    """Blow up the top-left corner of both boxes so the reader can see the two
    outlines separately -- at 0.1 px they are otherwise the same line."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    cx, cy = gx - side / 2.0, gy - side / 2.0
    ins = inset_axes(ax, width="34%", height="34%", loc="lower right",
                     borderpad=0.6)
    ins.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ins.set_xlim(cx - span, cx + span)
    ins.set_ylim(cy + span, cy - span)
    ins.add_patch(Rectangle((cx, cy), side, side, fill=False,
                            edgecolor=BOX_TRUE, linewidth=3.0, zorder=3))
    ins.add_patch(Rectangle((px - side / 2.0, py - side / 2.0), side, side,
                            fill=False, edgecolor=BOX_PRED, linewidth=1.2,
                            zorder=4))
    ins.set_xticks([]); ins.set_yticks([])
    for sp in ins.spines.values():
        sp.set_edgecolor("#7ee787"); sp.set_linewidth(1.0)
    ins.set_title(f"±{span:g} px", fontsize=6.5, color="#7ee787", pad=1.5)


def strip(fig, gs_row, panels, row_title):
    """One example: four panels plus a left-hand caption."""
    for i, (ax, draw, title) in enumerate(panels):
        draw(ax)
        ax.set_title(title, fontsize=8.5, color="#d8dee9", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#30363d")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "docs", "how-it-works.png"))
    ap.add_argument("--tiers", default="/Users/sachin/Development/semicon/phase3_tiers")
    ap.add_argument("--preds", default="/tmp/p3run")
    ap.add_argument("--dpi", type=int, default=150)
    a = ap.parse_args(argv)

    examples = _collect(a)
    if not examples:
        raise SystemExit("no examples found -- run the predictions first")

    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(13.2, 3.35 * n),
                             facecolor="#0d1117")
    if n == 1:
        axes = axes[None, :]

    for r, ex in enumerate(examples):
        for c in range(4):
            ax = axes[r, c]
            ax.set_facecolor("#0d1117")
            ex["draw"][c](ax)
            ax.set_title(ex["titles"][c], fontsize=9, color="#e6edf3", pad=5)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#30363d")
        axes[r, 0].set_ylabel(ex["label"], fontsize=9.5, color="#7ee787",
                              labelpad=8)

    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=BOX_TRUE, lw=4,
                               label="ground truth"),
                        Line2D([], [], color=BOX_PRED, lw=1.6,
                               label="Drift-Sense prediction")],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor="#e6edf3", bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Drift-Sense — how a pair is solved.  "
        "Phase 3: raw CAD polygons → inferred yield raster → real SEM capture → answer.",
        fontsize=11.5, color="#e6edf3", y=0.995)
    fig.tight_layout(rect=[0, 0.028, 1, 0.975])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=a.dpi, facecolor=fig.get_facecolor())
    print(f"wrote {a.out}  ({n} examples)")
    return 0


def _collect(a):
    """Pick real pairs: a clean one, a harsh one, and a true absent site."""
    out = []
    want = [("nominal", "present", "Phase 3 · nominal"),
            ("harsh", "present", "Phase 3 · harsh capture"),
            ("harsh", "absent", "Phase 3 · no true match")]

    for tier, kind, label in want:
        root = os.path.join(a.tiers, f"phase3_{tier}")
        pairs = _index(_read_csv(os.path.join(root, "pairs.csv")))
        gt = _index(_read_csv(os.path.join(root, "ground_truth.csv")))
        pred_path = os.path.join(a.preds, f"pred_{tier}.csv")
        if not os.path.exists(pred_path):
            continue
        pred = _index(_read_csv(pred_path))

        # Prefer a visibly rotated example, but never drop the row for it --
        # the nominal tier is generated unrotated, so a hard filter empties it.
        pick = first = None
        for pid, g in gt.items():
            present = g["present"] == "1"
            if (kind == "present") != present:
                continue
            if pid not in pred or pid not in pairs:
                continue
            if first is None:
                first = pid
            if kind == "absent" or float(g["theta"]) >= 2.0:
                pick = pid
                break
        pick = pick or first
        if pick is None:
            continue
        out.append(_phase3_example(root, pairs[pick], gt[pick], pred[pick], label))

    p2 = _phase2_example(a)
    if p2:
        out.append(p2)
    return out


def _phase3_example(root, pair, g, p, label):
    gds_path = os.path.join(root, pair["reference_gds_path"])
    search = cv2.imread(os.path.join(root, pair["search_path"]),
                        cv2.IMREAD_GRAYSCALE)
    raster = gds.render_reference(gds_path, size=gds.REF_SIZE)
    present = g["present"] == "1"
    found = p["found"] == "1"
    side = 100.0          # 1000 nm reference at 10 nm/px

    def d0(ax):
        n = cad_polygons_panel(ax, gds_path, gds.REF_SIZE, 1.0)
        ax._nlayers = n

    def d1(ax):
        grey_panel(ax, raster)

    def d2(ax):
        grey_panel(ax, search)

    def d3(ax):
        grey_panel(ax, search)
        if present:
            box(ax, float(g["x"]), float(g["y"]), side, BOX_TRUE,
                "ground truth", dashed=True)
        if found:
            box(ax, float(p["x"]), float(p["y"]), side, BOX_PRED, "predicted")
        if present and found:
            _corner_inset(ax, search, float(g["x"]), float(g["y"]),
                          float(p["x"]), float(p["y"]), side)

    if present and found:
        err = float(np.hypot(float(p["x"]) - float(g["x"]),
                             float(p["y"]) - float(g["y"])))
        verdict = (f"found · {err:.2f} px  θ {float(p['theta']):+.2f}°"
                   f" (true {float(g['theta']):+.2f}°)")
    elif not present and not found:
        verdict = f"correctly rejected · score {float(p['score']):.3f}"
    else:
        verdict = "MISMATCH"

    return {
        "label": label,
        "titles": ["reference CAD — polygons, no brightness",
                   "inferred yield raster (one grey / layer)",
                   "search SEM — 1000×1000 @ 10 nm/px",
                   verdict],
        "draw": [d0, d1, d2, d3],
    }


def _phase2_example(a):
    root = os.path.join(REPO, "generator", "output")
    if not os.path.exists(os.path.join(root, "pairs.csv")):
        return None
    pairs = _index(_read_csv(os.path.join(root, "pairs.csv")))
    gt = _index(_read_csv(os.path.join(root, "ground_truth.csv")))
    pred_path = os.path.join(a.preds, "pred_p2.csv")
    if not os.path.exists(pred_path):
        return None
    pred = _index(_read_csv(pred_path))

    pick = next((pid for pid, g in gt.items()
                 if g["present"] == "1" and pid in pred), None)
    if pick is None:
        return None
    pair, g, p = pairs[pick], gt[pick], pred[pick]

    ref = cv2.imread(os.path.join(root, pair["reference_path"]),
                     cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(os.path.join(root, pair["search_path"]),
                        cv2.IMREAD_GRAYSCALE)
    scale = float(g["scale"])
    side = ref.shape[0] / scale
    small = cv2.resize(ref, (int(round(side)), int(round(side))),
                       interpolation=cv2.INTER_AREA)
    err = float(np.hypot(float(p["x"]) - float(g["x"]),
                         float(p["y"]) - float(g["y"])))

    return {
        "label": "Phase 2 · SEM reference",
        "titles": ["reference SEM — 1000×1000 fine scan",
                   f"same crop at search scale (÷{scale:g})",
                   "search SEM — 1000×1000 coarse scan",
                   f"found · {err:.2f} px  θ {float(p['theta']):+.2f}°"
                   f" (true {float(g['theta']):+.2f}°)"],
        "draw": [lambda ax: grey_panel(ax, ref),
                 lambda ax: grey_panel(ax, small),
                 lambda ax: grey_panel(ax, search),
                 lambda ax: (grey_panel(ax, search),
                             box(ax, float(g["x"]), float(g["y"]), side,
                                 BOX_TRUE, "ground truth", dashed=True),
                             box(ax, float(p["x"]), float(p["y"]), side,
                                 BOX_PRED, "predicted"))],
    }


if __name__ == "__main__":
    raise SystemExit(main())

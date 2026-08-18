"""
Visualize the individual process layers that make up a rendered structure,
instead of only the flattened grayscale composite a real SEM would image.

Real DRAM cells are built bottom-up: substrate/active area -> STI isolation
-> word-line (gate poly) stack -> bit-line contact pad -> bit-line metal ->
storage-node contact -> storage capacitor. Real FinFETs: substrate -> fin ->
STI -> gate stack -> spacer -> source/drain -> contact -> metal1. This tool
doesn't model every one of those (the generator draws a simplified 3-4 layer
stand-in per architecture), but it renders what IS modeled -- substrate,
word_line/fin, bit_line/gate, contact -- as separate, individually
inspectable layers plus a false-colored composite and a simple exploded
stack view, both for QA (does each layer look right on its own?) and for
teaching (what's actually being drawn, in what order?).

Usage:
    python visualize_layers.py --architecture dram_1x --size 500 --seed 3
    python visualize_layers.py --architecture finfet_10nm --size 500 --seed 3
"""

import argparse
import os

import cv2
import numpy as np
from PIL import Image

from src.presets import get_preset
from src.patterns.dram import generate_dram_canvas
from src.patterns.finfet import generate_finfet_canvas

_GENERATORS = {"dram": generate_dram_canvas, "finfet": generate_finfet_canvas}

# Distinct false color per layer (BGR for cv2), roughly bottom-to-top order.
_LAYER_COLORS = {
    "substrate": (70, 60, 55),
    "word_line": (255, 140, 70),
    "bit_line": (110, 210, 90),
    "storage_contact": (60, 90, 245),
    "fin": (255, 140, 70),
    "gate": (110, 210, 90),
    "contact": (60, 90, 245),
}

_LAYER_ORDER = {
    "dram": ["substrate", "word_line", "bit_line", "storage_contact"],
    "finfet": ["substrate", "fin", "gate", "contact"],
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--architecture", default="dram_1x")
    p.add_argument("--size", type=int, default=500)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--collapse-threshold-nm", type=float, default=10.0)
    p.add_argument("--out-dir", default="./output/layers")
    return p.parse_args()


def false_color(layer: np.ndarray, color_bgr: tuple) -> np.ndarray:
    """Map a single-channel layer (0 = absent, >0 = feature) to a BGR image
    where feature pixels take `color_bgr`, scaled by the layer's own
    intensity (keeps the per-instance jitter/anti-aliasing visible)."""
    alpha = (layer.astype(np.float32) / 255.0)[:, :, None]
    color = np.array(color_bgr, dtype=np.float32)[None, None, :]
    out = alpha * color
    return np.clip(out, 0, 255).astype(np.uint8)


def build_exploded_stack(layers_bgr: list, labels: list, shear_px: int = 46, gap_px: int = 34) -> np.ndarray:
    """Simple axonometric-style exploded view: each layer sheared and offset
    upward/rightward from the one below it, bottom (substrate) drawn first."""
    n = len(layers_bgr)
    h, w = layers_bgr[0].shape[:2]
    canvas_w = w + shear_px * n + 260
    canvas_h = h + gap_px * n + 40
    canvas = np.full((canvas_h, canvas_w, 3), 14, dtype=np.uint8)

    pts_src = np.float32([[0, 0], [w, 0], [0, h]])
    for i, (layer, label) in enumerate(zip(layers_bgr, labels)):
        ox = shear_px * (n - 1 - i) + 20
        oy = canvas_h - h - gap_px * i - 20
        pts_dst = np.float32([
            [ox + shear_px, oy], [ox + shear_px + w, oy], [ox, oy + h],
        ])
        M = cv2.getAffineTransform(pts_src, pts_dst)
        warped = cv2.warpAffine(
            layer, M, (canvas_w, canvas_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        mask = np.any(warped > 4, axis=2)
        # thin "slab" edge for a pseudo-3D thickness cue
        canvas[mask] = warped[mask]
        cv2.putText(
            canvas, label, (canvas_w - 230, oy + h // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (225, 225, 225), 1, cv2.LINE_AA,
        )
        cv2.line(canvas, (canvas_w - 240, oy + h // 2 - 4), (ox + w, oy + int(h * 0.3)), (90, 90, 90), 1, cv2.LINE_AA)

    return canvas


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    preset = get_preset(args.architecture)
    kind = preset["kind"]
    generator = _GENERATORS[kind]
    rng = np.random.default_rng(args.seed)

    canvas, layers = generator(
        args.size, preset, args.collapse_threshold_nm, rng, return_layers=True,
    )

    order = _LAYER_ORDER[kind]
    layers_bgr = []
    for name in order:
        colored = false_color(layers[name], _LAYER_COLORS[name])
        out_path = os.path.join(args.out_dir, f"{args.architecture}_layer_{name}.png")
        Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)).save(out_path)
        print("wrote", out_path)
        layers_bgr.append(colored)

    # False-color flattened composite (all layers, painter's algorithm bottom->top)
    composite = np.zeros((args.size, args.size, 3), dtype=np.uint8)
    for name in order:
        colored = false_color(layers[name], _LAYER_COLORS[name])
        mask = np.any(colored > 4, axis=2)
        composite[mask] = colored[mask]
    composite_path = os.path.join(args.out_dir, f"{args.architecture}_composite_falsecolor.png")
    Image.fromarray(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)).save(composite_path)
    print("wrote", composite_path)

    # Real grayscale composite, for reference
    gray_path = os.path.join(args.out_dir, f"{args.architecture}_composite_grayscale.png")
    Image.fromarray(canvas).save(gray_path)
    print("wrote", gray_path)

    # Exploded stack
    exploded = build_exploded_stack(layers_bgr, [n.replace("_", " ") for n in order])
    exploded_path = os.path.join(args.out_dir, f"{args.architecture}_exploded_stack.png")
    Image.fromarray(cv2.cvtColor(exploded, cv2.COLOR_BGR2RGB)).save(exploded_path)
    print("wrote", exploded_path)


if __name__ == "__main__":
    main()

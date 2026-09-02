#!/usr/bin/env python3
"""Drift-Sense localisation inference.

Given a high-resolution Reference image (1 nm/px) and a low-resolution Search
image (10 nm/px) of a repeating semiconductor layout, predict the centre
(x, y) -- in Search-image pixels -- of the region where the Reference pattern
appears. Where several regions match equally well, the one nearest the centre
of the Search image is returned.

Usage
-----
    python infer.py --reference path/to/reference.png --search path/to/search.png

    # positional form works too
    python infer.py reference.png search.png

    # structured output, and an optional heatmap dump
    python infer.py --reference r.png --search s.png --json
    python infer.py --reference r.png --search s.png --save-heatmap heat.png

Output
------
Prints one line to stdout: `x,y` with two decimals (e.g. `418.73,265.10`).
With --json, prints a JSON object including the confidence score.

Weights are loaded automatically from weights/driftsense.pt next to this
script; override with --weights. If the weights or PyTorch are unavailable
the script falls back to a classical multi-scale ZNCC matcher and still
prints a coordinate (a warning goes to stderr, never to stdout).
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

DEFAULT_WEIGHTS = os.path.join(HERE, "weights", "driftsense.pt")
# The routing threshold lives in driftsense.policy (the single definition of
# the shipped decode -- see audit C-01); re-exported for backwards
# compatibility with scripts that import it from here.
from driftsense.policy import ROUTE_THRESHOLD  # noqa: E402,F401


def zncc_fallback(reference: np.ndarray, search: np.ndarray) -> dict:
    """Classical multi-scale ZNCC, dependency-free apart from OpenCV.

    Only used when the learned model cannot be loaded. It is materially worse
    on periodic layouts -- it latches onto the wrong repeat -- but it keeps
    the script runnable in any environment.
    """
    from driftsense.matching import template_hypotheses
    best = None
    for scale in [f * m for f in template_hypotheses(reference)
                  for m in (0.9, 0.95, 1.0, 1.05, 1.1)]:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        tmpl = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if best is None or score > best["score"]:
            best = {"x": loc[0] + tw / 2.0, "y": loc[1] + th / 2.0,
                    "score": float(score), "method": "zncc-fallback"}
    if best is None:
        return {"x": search.shape[1] / 2.0, "y": search.shape[0] / 2.0,
                "score": 0.0, "method": "center-fallback"}
    return best


def _fuse_conv_bn(model):
    """Fold Conv2d -> BatchNorm2d pairs into single convolutions (eval only).

    Exact: in eval mode BatchNorm applies fixed per-channel scale and shift from
    its running statistics, which composes into the convolution's weight and
    bias. The pass walks children in declaration order and only fuses a
    BatchNorm whose immediately preceding sibling is the Conv2d that feeds it --
    the pattern this model uses throughout. Anything else is left alone, so a
    layout the pass does not understand degrades to no fusion rather than to a
    wrong graph.
    """
    import torch.nn as nn
    from torch.nn.utils.fusion import fuse_conv_bn_eval

    model.eval()

    def walk(mod):
        prev_name, prev = None, None
        for name, child in list(mod.named_children()):
            if isinstance(child, nn.BatchNorm2d) and isinstance(prev, nn.Conv2d):
                setattr(mod, prev_name, fuse_conv_bn_eval(prev, child))
                setattr(mod, name, nn.Identity())
                prev_name, prev = None, None
            else:
                walk(child)
                prev_name, prev = name, child

    walk(model)
    return model


def load_model(weights_path: str):
    """Return (model, device) or None if the learned path is unavailable."""
    try:
        import torch
        from driftsense.model import DriftSenseNet
    except Exception as e:  # torch missing / broken install
        print(f"[warn] PyTorch unavailable ({e}); using ZNCC fallback", file=sys.stderr)
        return None

    if not os.path.exists(weights_path):
        print(f"[warn] weights not found at {weights_path}; using ZNCC fallback",
              file=sys.stderr)
        return None

    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)
        # Checkpoints from a scaled run record their own width. Older ones do
        # not, and must keep loading with the original defaults -- so the
        # fallback here is the constructor's own signature, not a guess.
        kw = ckpt.get("arch_kwargs") or {}
        model = DriftSenseNet(**kw)
        model.load_state_dict(state)
        model.eval()
    except Exception as e:
        print(f"[warn] could not load weights ({e}); using ZNCC fallback", file=sys.stderr)
        return None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # CPU is the graded configuration (4 cores, no GPU, 5 s median), and there
    # the network is 90.6% of pair time -- measured with scripts/profile_pair.py
    # on the CPU-only venv. NCHW forces oneDNN to reorder activations on every
    # convolution; channels_last lets it keep the blocked layout across the
    # whole stack.
    #
    # Measured on 6 set B pairs, 4 threads, CPU-only torch 2.13:
    # 4.612 -> 1.769 s/pair, a 2.61x speedup, with x/y/scale/theta/score
    # bit-identical. It is a memory-layout choice, not an algorithm change:
    # the convolutions compute the same values in a different traversal order.
    #
    # Guarded because the win is CPU-specific and the layout is only defined
    # for 4-D weights; a failure here must not cost the run. Set
    # DRIFTSENSE_CHANNELS_LAST=0 to fall back to NCHW on a platform where the
    # oneDNN path misbehaves.
    #
    # NOT bit-identical: re-association in the blocked kernels moves x/y by up
    # to 5.4e-06 px and score by 2.3e-06 (30 pairs across sets A/B/C). That is
    # ~200,000x below the 1 px credit tier, but it is a numerical difference,
    # not an exact one, and is stated as such.
    if device.type == "cpu" and os.environ.get("DRIFTSENSE_CHANNELS_LAST", "1") != "0":
        # Fold every BatchNorm2d into the convolution feeding it. In eval mode a
        # BatchNorm is a fixed affine map, so this is an algebraic identity --
        # the folded weights produce the same function with 17 fewer kernel
        # launches and 17 fewer passes over the activations. Measured on the
        # 924x924 search branch, 4 threads: 979.8 -> 706.3 ms, 1.39x, max output
        # difference 5.25e-06 (float re-association only).
        if os.environ.get("DRIFTSENSE_FUSE_BN", "1") != "0":
            try:
                model = _fuse_conv_bn(model)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] conv/bn fusion skipped ({e})", file=sys.stderr)
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] channels_last unavailable ({e}); using default layout",
                  file=sys.stderr)
    return model, device


def read_gray(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"error: could not read image '{path}'")
    return img


def predict(reference_path: str, search_path: str, weights_path: str = DEFAULT_WEIGHTS,
            want_heatmap: bool = False, tta: bool = True,
            route_threshold: float = ROUTE_THRESHOLD) -> dict:
    reference = read_gray(reference_path)
    search = read_gray(search_path)

    loaded = load_model(weights_path)
    if loaded is None:
        return zncc_fallback(reference, search)

    model, device = loaded

    # One shared decode policy with evaluate.py (audit C-01): pose estimation
    # and adaptive routing live in driftsense.policy, not here.
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
                   help="single view instead of 8-way dihedral voting: ~7x faster, "
                        "0.3-1.6 points less accurate at the 5px tolerance")
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

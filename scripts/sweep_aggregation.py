#!/usr/bin/env python3
"""Sweep the aggregation rule's hyperparameters over the cached proposals.

Reuses the cache built by tune_aggregation.py, so every configuration is
essentially free -- the eight forward passes are already done. Validation only.

    python scripts/sweep_aggregation.py --split data/val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.matching import make_template, refine_zncc, standardize  # noqa: E402
from scripts.tune_aggregation import centroid, cluster  # noqa: E402


def decide(cl, sea_n, tpl_n, alpha, top_k, gate_only_on_disagreement, n_views=8):
    """Score candidate clusters by network confidence plus, optionally, a ZNCC
    verification term. alpha=0 disables verification entirely."""
    ranked = sorted(cl, key=lambda c: -sum(m[2] for m in c))
    if alpha <= 0 or len(ranked) == 1:
        return centroid(ranked[0])
    # If every view already agrees, there is nothing to arbitrate.
    if gate_only_on_disagreement and len(ranked[0]) == n_views:
        return centroid(ranked[0])

    cands = ranked[:top_k]
    tot = sum(sum(m[2] for m in c) for c in cands) or 1.0
    best, best_s = None, -np.inf
    for c in cands:
        cx, cy = centroid(c)
        _, _, zn = refine_zncc(sea_n, tpl_n, cx, cy)
        s = zn + alpha * (sum(m[2] for m in c) / tot)
        if s > best_s:
            best, best_s = (cx, cy), s
    return best


def run(cache, imgs, alpha, top_k, cluster_px, gate):
    d = []
    for i, s in enumerate(cache):
        cl = cluster(s["props"], cluster_px)
        sea_n, tpl_n = imgs[i]
        cx, cy = decide(cl, sea_n, tpl_n, alpha, top_k, gate)
        rx, ry, _ = refine_zncc(sea_n, tpl_n, cx, cy)
        if np.hypot(rx - cx, ry - cy) <= 10.0:
            cx, cy = rx, ry
        gx, gy = s["gt"]
        d.append(np.hypot(cx - gx, cy - gy))
    return np.array(d)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="data/val")
    p.add_argument("--cache", default="")
    args = p.parse_args()

    cache = json.load(open(args.cache or f"/tmp/tta_cache_{os.path.basename(args.split)}.json"))
    imgs = []
    for s in cache:
        ref = cv2.imread(os.path.join(args.split, s["ref"]), cv2.IMREAD_GRAYSCALE)
        sea = cv2.imread(os.path.join(args.split, s["search"]), cv2.IMREAD_GRAYSCALE)
        imgs.append((standardize(sea / 255.0), standardize(make_template(ref) / 255.0)))
    print(f"{len(cache)} samples from {args.split}\n")

    base = run(cache, imgs, 0.0, 1, 6.0, False)
    print(f"  {'baseline (shipped sum_score)':<44} acc@2 {(base<=2).mean():.3f}  "
          f"acc@5 {(base<=5).mean():.3f}  acc@10 {(base<=10).mean():.3f}  mean {base.mean():6.2f}")

    print("\n  ZNCC-verified, alpha sweep (top_k=4, cluster 6px, always arbitrate):")
    for a in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.5):
        d = run(cache, imgs, a, 4, 6.0, False)
        print(f"    alpha={a:<5} {'':<32} acc@2 {(d<=2).mean():.3f}  acc@5 {(d<=5).mean():.3f}  "
              f"acc@10 {(d<=10).mean():.3f}  mean {d.mean():6.2f}  ({int((d<=5).sum())-int((base<=5).sum()):+d} samples)")

    print("\n  gated: only arbitrate when the eight views disagree (alpha sweep):")
    for a in (0.2, 0.35, 0.5, 1.0):
        d = run(cache, imgs, a, 4, 6.0, True)
        print(f"    alpha={a:<5} {'':<32} acc@2 {(d<=2).mean():.3f}  acc@5 {(d<=5).mean():.3f}  "
              f"acc@10 {(d<=10).mean():.3f}  mean {d.mean():6.2f}  ({int((d<=5).sum())-int((base<=5).sum()):+d} samples)")

    print("\n  top_k / cluster radius (alpha=0.5):")
    for k in (2, 3, 4, 6):
        for cp in (4.0, 6.0, 10.0):
            d = run(cache, imgs, 0.5, k, cp, False)
            print(f"    top_k={k} cluster={cp:<5} {'':<24} acc@2 {(d<=2).mean():.3f}  acc@5 {(d<=5).mean():.3f}  "
                  f"acc@10 {(d<=10).mean():.3f}  mean {d.mean():6.2f}  ({int((d<=5).sum())-int((base<=5).sum()):+d} samples)")


if __name__ == "__main__":
    main()

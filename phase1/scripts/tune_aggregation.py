#!/usr/bin/env python3
"""Tune the TTA aggregation rule.

Measured on the shipped weights, plain cluster-voting fixes 3 samples and
breaks 2 relative to a single view -- and an oracle that picked the better of
the two per sample would score meaningfully higher. That gap is a decode
problem, not a model problem, so it is worth attacking without retraining.

The eight view proposals are the expensive part (8 forward passes), so they are
computed once per sample and cached; every candidate rule is then evaluated
offline over the same cache. Rules are selected on validation only -- the test
split is touched once, afterwards, by evaluate.py.

    python scripts/tune_aggregation.py --split data/val --build
    python scripts/tune_aggregation.py --split data/val
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense.dataset import load_manifest  # noqa: E402
from driftsense.matching import (  # noqa: E402
    _dihedral_img, _dihedral_point_inv, locate, make_template, refine_zncc,
    standardize,
)
from driftsense.model import DriftSenseNet  # noqa: E402


def build_cache(split: str, weights: str, cache_path: str, limit=None):
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    model = DriftSenseNet()
    model.load_state_dict(ck.get("model", ck))
    model.to(device).eval()

    rows = load_manifest(split)[:limit] if limit else load_manifest(split)
    out = []
    for n, r in enumerate(rows):
        ref = cv2.imread(os.path.join(split, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        sea = cv2.imread(os.path.join(split, r["search_path"]), cv2.IMREAD_GRAYSCALE)
        size = sea.shape[0]
        props = []
        for t in range(8):
            res = locate(model, _dihedral_img(ref, t), _dihedral_img(sea, t),
                         device, refine=False)
            x, y = _dihedral_point_inv(res["x"], res["y"], size, t)
            props.append([float(x), float(y), float(res["score"])])
        out.append({"ref": r["reference_path"], "search": r["search_path"],
                    "gt": [float(r["gt_x_corr"]), float(r["gt_y_corr"])],
                    "props": props})
        if (n + 1) % 50 == 0:
            print(f"  cached {n+1}/{len(rows)}", flush=True)
    json.dump(out, open(cache_path, "w"))
    print(f"wrote {cache_path} ({len(out)} samples)")


def cluster(props, cluster_px=6.0):
    """Greedy proximity clustering, strongest proposal first."""
    ps = sorted(props, key=lambda p: -p[2])
    cl = []
    for p in ps:
        for c in cl:
            if np.hypot(p[0] - c[0][0], p[1] - c[0][1]) <= cluster_px:
                c.append(p)
                break
        else:
            cl.append([p])
    return cl


def centroid(c):
    w = np.array([m[2] for m in c], dtype=np.float64)
    return (float(np.average([m[0] for m in c], weights=w)),
            float(np.average([m[1] for m in c], weights=w)))


# ---------------------------------------------------------------------------
# Aggregation rules. Each takes the clusters (and optionally the images, for
# rules that consult an independent signal) and returns a centre.
# ---------------------------------------------------------------------------

def rule_sum_score(cl, **kw):
    """Current shipped rule: cluster with greatest total confidence."""
    return centroid(max(cl, key=lambda c: (sum(m[2] for m in c), len(c))))


def rule_top1(cl, **kw):
    """No voting -- just the single strongest view."""
    best = max((m for c in cl for m in c), key=lambda m: m[2])
    return best[0], best[1]


def rule_count(cl, **kw):
    """Majority vote; ties broken by total confidence."""
    return centroid(max(cl, key=lambda c: (len(c), sum(m[2] for m in c))))


def rule_max_score(cl, **kw):
    """Cluster containing the single strongest proposal."""
    return centroid(max(cl, key=lambda c: max(m[2] for m in c)))


def rule_count_x_mean(cl, **kw):
    return centroid(max(cl, key=lambda c: len(c) * float(np.mean([m[2] for m in c]))))


def _zncc_at(sea_n, tpl_n, x, y):
    _, _, s = refine_zncc(sea_n, tpl_n, x, y)
    return s


def rule_zncc_verify(cl, sea_n=None, tpl_n=None, top_k=4, **kw):
    """Two-stage: the network shortlists candidate regions, classical ZNCC at
    full resolution arbitrates between them.

    ZNCC searching the whole frame picks the wrong repeat ~50% of the time, but
    choosing among a handful of network-proposed regions is a far easier task,
    and it is an *independent* signal from the network's own confidence.
    """
    cands = sorted(cl, key=lambda c: -(sum(m[2] for m in c)))[:top_k]
    best, best_s = None, -np.inf
    for c in cands:
        cx, cy = centroid(c)
        s = _zncc_at(sea_n, tpl_n, cx, cy)
        if s > best_s:
            best, best_s = (cx, cy), s
    return best


def rule_zncc_plus_prior(cl, sea_n=None, tpl_n=None, top_k=4, alpha=0.35, **kw):
    """ZNCC verification, but combined with the network's confidence rather
    than replacing it -- guards against ZNCC preferring a crisper-looking decoy.
    """
    cands = sorted(cl, key=lambda c: -(sum(m[2] for m in c)))[:top_k]
    tot = sum(sum(m[2] for m in c) for c in cands) or 1.0
    best, best_s = None, -np.inf
    for c in cands:
        cx, cy = centroid(c)
        s = _zncc_at(sea_n, tpl_n, cx, cy) + alpha * (sum(m[2] for m in c) / tot)
        if s > best_s:
            best, best_s = (cx, cy), s
    return best


RULES = {
    "sum_score (shipped)": rule_sum_score,
    "top1 single view": rule_top1,
    "count (majority)": rule_count,
    "max_score": rule_max_score,
    "count x mean": rule_count_x_mean,
    "ZNCC verify top4": rule_zncc_verify,
    "ZNCC verify + prior a=0.35": rule_zncc_plus_prior,
}
NEEDS_IMAGES = {"ZNCC verify top4", "ZNCC verify + prior a=0.35"}


def evaluate(cache, split, cluster_px=6.0, rules=None):
    rules = rules or RULES
    results = {}
    # Pre-load images only for the rules that need them.
    need = any(n in NEEDS_IMAGES for n in rules)
    imgs = []
    if need:
        for s in cache:
            ref = cv2.imread(os.path.join(split, s["ref"]), cv2.IMREAD_GRAYSCALE)
            sea = cv2.imread(os.path.join(split, s["search"]), cv2.IMREAD_GRAYSCALE)
            imgs.append((standardize(sea / 255.0), standardize(make_template(ref) / 255.0)))

    for name, fn in rules.items():
        d = []
        for i, s in enumerate(cache):
            cl = cluster(s["props"], cluster_px)
            kw = {}
            if name in NEEDS_IMAGES:
                kw = {"sea_n": imgs[i][0], "tpl_n": imgs[i][1]}
            cx, cy = fn(cl, **kw)
            # Every rule gets the same sub-pixel snap, so the comparison is
            # purely about which region was chosen.
            if need:
                sea_n, tpl_n = imgs[i]
            else:
                ref = cv2.imread(os.path.join(split, s["ref"]), cv2.IMREAD_GRAYSCALE)
                sea = cv2.imread(os.path.join(split, s["search"]), cv2.IMREAD_GRAYSCALE)
                sea_n, tpl_n = standardize(sea / 255.0), standardize(make_template(ref) / 255.0)
            rx, ry, _ = refine_zncc(sea_n, tpl_n, cx, cy)
            if np.hypot(rx - cx, ry - cy) <= 10.0:
                cx, cy = rx, ry
            gx, gy = s["gt"]
            d.append(np.hypot(cx - gx, cy - gy))
        d = np.array(d)
        results[name] = d
        print(f"  {name:<30} median {np.median(d):6.3f}  acc@1 {(d<=1).mean():.3f}  "
              f"acc@2 {(d<=2).mean():.3f}  acc@5 {(d<=5).mean():.3f}  "
              f"acc@10 {(d<=10).mean():.3f}  mean {d.mean():7.2f}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="data/val")
    p.add_argument("--weights", default="weights/driftsense.pt")
    p.add_argument("--cache", default="")
    p.add_argument("--build", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    cache_path = args.cache or f"/tmp/tta_cache_{os.path.basename(args.split)}.json"
    if args.build or not os.path.exists(cache_path):
        build_cache(args.split, args.weights, cache_path, args.limit or None)

    cache = json.load(open(cache_path))
    print(f"\n{len(cache)} samples from {args.split}\n")
    res = evaluate(cache, args.split)

    print("\n  oracle over all rules (upper bound if we always chose right):")
    best = np.min(np.stack([res[k] for k in res]), axis=0)
    print(f"    acc@5 {(best<=5).mean():.3f}  acc@2 {(best<=2).mean():.3f}")


if __name__ == "__main__":
    main()

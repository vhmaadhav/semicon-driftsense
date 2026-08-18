#!/usr/bin/env python3
"""Evaluate Drift-Sense localisation on generated splits.

Reports, per split, the distance between the predicted centre and the
geometry-corrected ground truth, and accuracy at several tolerances. Runs the
classical multi-scale ZNCC matcher over the same data for comparison, so the
learned model's contribution is visible rather than asserted.

    python evaluate.py --splits data/test data/test_medium data/test_severe \
        --weights weights/driftsense.pt --out results/

Writes results.json (and per-split error arrays) into --out.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from driftsense.dataset import load_manifest  # noqa: E402
from driftsense.matching import locate, locate_tta, zncc_only  # noqa: E402
from driftsense.model import DriftSenseNet  # noqa: E402

TOLERANCES = (1.0, 2.0, 5.0, 10.0)


def summarize(dists: np.ndarray, name: str) -> dict:
    d = np.asarray(dists, dtype=np.float64)
    out = {"method": name, "n": int(d.size),
           "median_px": float(np.median(d)), "mean_px": float(d.mean()),
           "p90_px": float(np.percentile(d, 90)), "p99_px": float(np.percentile(d, 99))}
    for t in TOLERANCES:
        out[f"acc@{t:g}px"] = float((d <= t).mean())
    return out


def run_split(split_dir: str, model, device, limit=None, do_baseline=True, tta=True) -> dict:
    rows = load_manifest(split_dir)
    if limit:
        rows = rows[:limit]

    d_model, d_zncc, scores = [], [], []
    for n, r in enumerate(rows):
        ref = cv2.imread(os.path.join(split_dir, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        sea = cv2.imread(os.path.join(split_dir, r["search_path"]), cv2.IMREAD_GRAYSCALE)
        gx, gy = float(r["gt_x_corr"]), float(r["gt_y_corr"])

        res = (locate_tta(model, ref, sea, device, refine=True) if tta
               else locate(model, ref, sea, device, refine=True))
        d_model.append(np.hypot(res["x"] - gx, res["y"] - gy))
        scores.append(res["score"])

        if do_baseline:
            z = zncc_only(ref, sea)
            d_zncc.append(np.hypot(z["x"] - gx, z["y"] - gy))

        if (n + 1) % 50 == 0:
            print(f"    {n + 1}/{len(rows)}", flush=True)

    result = {"split": os.path.basename(split_dir.rstrip("/")),
              "siamese": summarize(np.array(d_model),
                                   "siamese+tta8+zncc" if tta else "siamese+zncc"),
              "errors_siamese": [float(x) for x in d_model],
              "scores": [float(s) for s in scores]}
    if do_baseline:
        result["zncc"] = summarize(np.array(d_zncc), "zncc-baseline")
        result["errors_zncc"] = [float(x) for x in d_zncc]
    return result


def fmt(s: dict) -> str:
    return (f"median {s['median_px']:7.2f}px  mean {s['mean_px']:8.2f}px  "
            f"acc@1 {s['acc@1px']:.3f}  acc@2 {s['acc@2px']:.3f}  "
            f"acc@5 {s['acc@5px']:.3f}  acc@10 {s['acc@10px']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--splits", nargs="+", default=["data/test", "data/test_medium", "data/test_severe"])
    p.add_argument("--weights", default="weights/driftsense.pt")
    p.add_argument("--out", default="results")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--no-baseline", action="store_true")
    p.add_argument("--no-tta", action="store_true")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = DriftSenseNet()
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()
    print(f"weights: {args.weights}  (trained epochs: {ckpt.get('epoch', '?')})")
    print(f"device : {device}\n")

    os.makedirs(args.out, exist_ok=True)
    all_results = []
    for split in args.splits:
        print(f"=== {split}")
        r = run_split(split, model, device, limit=args.limit or None,
                      do_baseline=not args.no_baseline, tta=not args.no_tta)
        all_results.append(r)
        print(f"  siamese : {fmt(r['siamese'])}")
        if "zncc" in r:
            print(f"  zncc    : {fmt(r['zncc'])}")
        print()

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    print("=" * 100)
    print(f"{'split':<14}{'method':<22}{'median':>9}{'acc@1px':>10}{'acc@2px':>10}"
          f"{'acc@5px':>10}{'acc@10px':>10}")
    print("-" * 100)
    for r in all_results:
        for key in ("siamese", "zncc"):
            if key not in r:
                continue
            s = r[key]
            print(f"{r['split']:<14}{s['method']:<22}{s['median_px']:>8.2f}p"
                  f"{s['acc@1px']:>10.3f}{s['acc@2px']:>10.3f}"
                  f"{s['acc@5px']:>10.3f}{s['acc@10px']:>10.3f}")
    print("=" * 100)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()

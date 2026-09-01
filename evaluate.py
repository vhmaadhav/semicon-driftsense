#!/usr/bin/env python3
"""Evaluate Drift-Sense localisation on generated splits.

Reports, per split, the distance between the predicted centre and the ground
truth, and accuracy at several tolerances. Runs the classical multi-scale ZNCC
matcher over the same data for comparison, so the learned model's contribution
is visible rather than asserted.

    python evaluate.py --splits data/test data/test_medium data/test_severe \
        --weights weights/driftsense.pt --out results/

Ground-truth frame
------------------
Every manifest carries two labels for the same scene:

* ``gt_x_corr``/``gt_y_corr`` -- **corrected**: where the reference pattern
  actually lands in the acquired search image, after the raster shear/jitter
  and barrel distortion that `sem_imaging.image_search` applies. This is what
  a correct localisation visually points at, and what training uses.
* ``gt_x``/``gt_y`` -- **upstream**: the crop origin in the undistorted fine
  canvas divided by the 10x scale factor, i.e. the pre-warp navigation
  coordinate. This is the only label the organizer's own
  `generator/generate_dataset.py` and `generator/app.py` ever emit.

The two differ by the warp displacement at that row -- ~0.8 px mean at the
organizer's default shear, ~2.4 px on `severe`. That is decisive for acc@1px
and material for acc@5px on severe splits, so **both are reported by default**
(`--gt-frame both`). Never quote a headline number without saying which frame
it is measured in.

Writes results.json into --out: a dict with a `provenance` block (checkpoint
identity, device, decode settings) and a `splits` list. Older files written by
this script were a bare list; `scripts/compare_checkpoints.py` reads both.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
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
from driftsense.model import DriftSenseNet, net_from_checkpoint  # noqa: E402

TOLERANCES = (1.0, 2.0, 5.0, 10.0)

# label frame -> (manifest x column, manifest y column)
GT_COLUMNS = {
    "corrected": ("gt_x_corr", "gt_y_corr"),
    "upstream": ("gt_x", "gt_y"),
}


def summarize(dists: np.ndarray, name: str) -> dict:
    d = np.asarray(dists, dtype=np.float64)
    out = {"method": name, "n": int(d.size),
           "median_px": float(np.median(d)), "mean_px": float(d.mean()),
           "p90_px": float(np.percentile(d, 90)), "p99_px": float(np.percentile(d, 99))}
    for t in TOLERANCES:
        out[f"acc@{t:g}px"] = float((d <= t).mean())
    return out


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_split(split_dir: str, model, device, limit=None, do_baseline=True, tta=True,
              frames=("corrected",)) -> dict:
    rows = load_manifest(split_dir)
    if limit:
        rows = rows[:limit]

    # One inference pass per scene; the frames only change what it is scored
    # against, never what the model sees.
    d_model = {f: [] for f in frames}
    d_zncc = {f: [] for f in frames}
    scores = []

    for n, r in enumerate(rows):
        ref = cv2.imread(os.path.join(split_dir, r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        sea = cv2.imread(os.path.join(split_dir, r["search_path"]), cv2.IMREAD_GRAYSCALE)

        res = (locate_tta(model, ref, sea, device, refine=True) if tta
               else locate(model, ref, sea, device, refine=True))
        scores.append(res["score"])
        z = zncc_only(ref, sea) if do_baseline else None

        for f in frames:
            xcol, ycol = GT_COLUMNS[f]
            gx, gy = float(r[xcol]), float(r[ycol])
            d_model[f].append(np.hypot(res["x"] - gx, res["y"] - gy))
            if z is not None:
                d_zncc[f].append(np.hypot(z["x"] - gx, z["y"] - gy))

        if (n + 1) % 50 == 0:
            print(f"    {n + 1}/{len(rows)}", flush=True)

    method = "siamese+tta8+zncc" if tta else "siamese+zncc"
    result = {"split": os.path.basename(split_dir.rstrip("/")),
              "gt_frames": list(frames),
              "scores": [float(s) for s in scores]}

    for f in frames:
        # `corrected` keeps the historical unsuffixed key names so older
        # tooling and the committed cmp_* artifacts stay comparable.
        suffix = "" if f == "corrected" else f"_{f}"
        result[f"siamese{suffix}"] = summarize(np.array(d_model[f]), method)
        result[f"errors_siamese{suffix}"] = [float(x) for x in d_model[f]]
        if do_baseline:
            result[f"zncc{suffix}"] = summarize(np.array(d_zncc[f]), "zncc-baseline")
            result[f"errors_zncc{suffix}"] = [float(x) for x in d_zncc[f]]
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
    p.add_argument("--gt-frame", choices=("corrected", "upstream", "both"), default="both",
                   help="label convention to score against (see module docstring). "
                        "Default 'both': the corrected frame is what the pattern "
                        "visually occupies, the upstream frame is what the "
                        "organizer's generator emits.")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    frames = ("corrected", "upstream") if args.gt_frame == "both" else (args.gt_frame,)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = net_from_checkpoint(ckpt)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()

    digest = sha256_of(args.weights)
    print(f"weights: {args.weights}  (trained epochs: {ckpt.get('epoch', '?')})")
    print(f"sha256 : {digest}")
    print(f"device : {device}")
    print(f"decode : {'TTA x8' if not args.no_tta else 'single view'} + ZNCC refine")
    print(f"gt     : {', '.join(frames)}\n")

    os.makedirs(args.out, exist_ok=True)
    all_results = []
    for split in args.splits:
        print(f"=== {split}")
        r = run_split(split, model, device, limit=args.limit or None,
                      do_baseline=not args.no_baseline, tta=not args.no_tta,
                      frames=frames)
        all_results.append(r)
        for f in frames:
            suffix = "" if f == "corrected" else f"_{f}"
            print(f"  siamese [{f:9s}]: {fmt(r['siamese' + suffix])}")
            if ("zncc" + suffix) in r:
                print(f"  zncc    [{f:9s}]: {fmt(r['zncc' + suffix])}")
        print()

    # Provenance travels with the numbers: a results file that cannot be tied
    # to a specific checkpoint is not evidence of anything.
    payload = {
        "provenance": {
            "weights": args.weights,
            "weights_sha256": digest,
            "trained_epochs": ckpt.get("epoch"),
            "decode": "tta8+zncc-refine" if not args.no_tta else "single-view+zncc-refine",
            "gt_frames": list(frames),
            "device": str(device),
            "splits": list(args.splits),
            "limit": args.limit or None,
            "generated_utc": datetime.datetime.now(datetime.timezone.utc)
                             .strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "splits": all_results,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(payload, f, indent=2)

    width = 100 + (12 if len(frames) > 1 else 0)
    print("=" * width)
    header = f"{'split':<14}{'method':<22}"
    if len(frames) > 1:
        header += f"{'gt frame':<11}"
    print(header + f"{'median':>9}{'acc@1px':>10}{'acc@2px':>10}"
                   f"{'acc@5px':>10}{'acc@10px':>10}")
    print("-" * width)
    for r in all_results:
        for f in frames:
            suffix = "" if f == "corrected" else f"_{f}"
            for key in ("siamese", "zncc"):
                if (key + suffix) not in r:
                    continue
                s = r[key + suffix]
                line = f"{r['split']:<14}{s['method']:<22}"
                if len(frames) > 1:
                    line += f"{f:<11}"
                print(line + f"{s['median_px']:>8.2f}p"
                             f"{s['acc@1px']:>10.3f}{s['acc@2px']:>10.3f}"
                             f"{s['acc@5px']:>10.3f}{s['acc@10px']:>10.3f}")
    print("=" * width)
    print(f"\nwrote {os.path.join(args.out, 'results.json')}")


if __name__ == "__main__":
    main()

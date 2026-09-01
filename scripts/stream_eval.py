#!/usr/bin/env python3
"""Evaluate a checkpoint on freshly generated scenes -- no disk, any size.

Why this exists
---------------
`data/val` has 300 scenes. At acc@5px ~= 0.95 the binomial standard error there
is +/-1.3 points, but the differences between candidate checkpoints are 0.5-0.7
points (two samples). The split physically cannot resolve them: tonight it tied
two streaming epochs at exactly 0.953 and called a rejected experiment 0.937,
and only the *rejected* one was outside the noise.

Selecting a checkpoint on a metric that cannot see the effect being selected for
is just picking noise, and picking noise on validation buys nothing on test. The
fix is more validation scenes. Storing them is not an option -- 1 200 pairs is
~1.7 GB and the disk has ~20 GB free -- so they are generated in the dataloader
workers and consumed immediately, exactly like the streaming trainer.

Seeding is `SeedSequence([999983, i])`, a namespace disjoint from the training
stream (`[seed, epoch, worker]`) and from every on-disk split, so no scene here
has been trained on and none collides with `test*`.

Ground truth comes from `make_pairs`, which applies `correct_gt` -- the same
geometry-corrected convention as `gt_x_corr`/`gt_y_corr`.

    python scripts/stream_eval.py --weights weights/driftsense.pt -n 1200
    python scripts/stream_eval.py --weights weights/driftsense_v4_e7.pt -n 1200 --tta
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from driftsense.generate import PRESETS, make_pairs  # noqa: E402
from driftsense.matching import locate, locate_tta  # noqa: E402
from driftsense.model import DriftSenseNet, net_from_checkpoint  # noqa: E402

VAL_NAMESPACE = 999983  # disjoint from the training stream and every split seed


class StreamingScenes(IterableDataset):
    """Fresh (reference, search, gt) triples, uint8, geometry-corrected gt."""

    def __init__(self, n: int, crops_per_canvas: int = 4, noise: str = "randomized"):
        self.n = n
        self.crops = crops_per_canvas
        self.noise = noise
        self.arch = list(PRESETS.keys())

    def __len__(self):
        return self.n

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        # Canvas c is seeded from c alone and owns samples [c*crops, (c+1)*crops).
        # Workers stride over canvases, so the *set* of scenes is fixed by --num
        # and --crops-per-canvas and does not depend on --workers. That keeps
        # the comparison between checkpoints paired even if the worker count
        # changes between runs.
        total_canvases = (self.n + self.crops - 1) // self.crops
        for c in range(wid, total_canvases, nw):
            ss = np.random.SeedSequence([VAL_NAMESPACE, c])
            entropy = int(np.random.default_rng(ss).integers(0, 2 ** 63 - 1))
            room = min(self.crops, self.n - c * self.crops)
            for p in make_pairs(entropy, self.arch, self.noise, crops=self.crops)[:room]:
                yield {"reference": torch.from_numpy(p["reference"]),
                       "search": torch.from_numpy(p["search"]),
                       "gt": torch.tensor([p["gt_x"], p["gt_y"]], dtype=torch.float64)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/driftsense.pt")
    p.add_argument("-n", "--num", type=int, default=1200)
    p.add_argument("--crops-per-canvas", type=int, default=4)
    p.add_argument("--noise", default="randomized")
    p.add_argument("--tta", action="store_true", help="8-way TTA (matches infer.py; ~3x slower)")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = net_from_checkpoint(ckpt)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device).eval()

    ds = StreamingScenes(args.num, args.crops_per_canvas, args.noise)
    loader = DataLoader(ds, batch_size=None, num_workers=args.workers)

    print(f"weights : {args.weights} (epoch {ckpt.get('epoch', '?')})")
    print(f"device  : {device}   decode: {'tta8+zncc' if args.tta else 'single+zncc'}")
    print(f"scenes  : {args.num} generated fresh, {args.crops_per_canvas}/canvas, "
          f"noise={args.noise}\n")

    dists, scores = [], []
    t0 = time.time()
    for k, s in enumerate(loader):
        ref = s["reference"].numpy()
        sea = s["search"].numpy()
        gx, gy = float(s["gt"][0]), float(s["gt"][1])
        res = (locate_tta(model, ref, sea, device, refine=True) if args.tta
               else locate(model, ref, sea, device, refine=True))
        dists.append(float(np.hypot(res["x"] - gx, res["y"] - gy)))
        scores.append(float(res["score"]))
        if (k + 1) % 100 == 0:
            d = np.array(dists)
            print(f"  {k+1}/{args.num}  acc@5 {(d <= 5).mean():.4f}  "
                  f"med {np.median(d):.2f}px  {(k+1)/(time.time()-t0):.1f} scene/s",
                  flush=True)

    d = np.array(dists)
    n = d.size
    acc5 = float((d <= 5).mean())
    se = float(np.sqrt(acc5 * (1 - acc5) / n))
    summary = {"weights": args.weights, "n": n, "tta": args.tta,
               "median_px": float(np.median(d)), "mean_px": float(d.mean()),
               "acc@1px": float((d <= 1).mean()), "acc@2px": float((d <= 2).mean()),
               "acc@5px": acc5, "acc@10px": float((d <= 10).mean()),
               "se_acc5": se, "errors": [float(x) for x in d],
               "scores": scores}

    print(f"\n{'-' * 66}")
    print(f"n {n}   median {summary['median_px']:.2f}px   mean {summary['mean_px']:.2f}px")
    print(f"acc@1 {summary['acc@1px']:.4f}  acc@2 {summary['acc@2px']:.4f}  "
          f"acc@5 {acc5:.4f} +/- {se:.4f}  acc@10 {summary['acc@10px']:.4f}")
    print(f"{'-' * 66}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

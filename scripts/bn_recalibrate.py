#!/usr/bin/env python3
"""Recalibrate BatchNorm running statistics at the *inference* frame size.

Why
---
Training crops a 512 px window out of the search frame; inference runs on the
full 1000 px frame. The network is fully convolutional so the weights transfer,
but BatchNorm's running mean/variance do not: they were estimated on 128x128
feature maps and are applied to 250x250 ones. Two things shift with the frame
size:

* Padding fraction. The context branch stacks dilations 2/4/8/16, so its taps
  reach +/-31 feature cells. On a 128x128 map a large share of positions have
  part of that reach in zero padding; on a 250x250 map far fewer do. Zeros
  drag the activation mean and variance down by an amount that depends on the
  map size.
* Decoy population. The response map grows 104x104 -> 226x226, so the head's
  BN layers see a different distribution of correlation peaks.

This is the train/test resolution discrepancy of Touvron et al., "Fixing the
train-test resolution discrepancy" (arXiv:1906.06423), whose cheapest remedy is
exactly this: leave the weights alone and re-estimate the normalisation
statistics at test resolution.

The recalibration is forward-only -- no gradients, no weight updates, nothing
fitted to labels. It just replaces one set of moment estimates with a set
measured under the conditions the model is actually deployed in, so it cannot
overfit and it is not a training run. Scenes come from the existing training
split (their labels are never read).

    python scripts/bn_recalibrate.py --weights weights/driftsense.pt \
        --out weights/driftsense_bnrecal.pt --batches 250
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from driftsense.dataset import DriftSenseDataset  # noqa: E402
from driftsense.model import DriftSenseNet, net_from_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/driftsense.pt")
    p.add_argument("--out", default="weights/driftsense_bnrecal.pt")
    p.add_argument("--dirs", nargs="+", default=["data/train"])
    p.add_argument("--crop", type=int, default=1000,
                   help="frame size to estimate statistics at (1000 = inference)")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batches", type=int, default=250)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("mps" if torch.backends.mps.is_available()
                              else ("cuda" if torch.cuda.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = net_from_checkpoint(ckpt)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device)

    # train=False means no dihedral and no photometric jitter, and at crop 1000
    # no window crop either -- so the forward pass sees exactly what inference
    # sees. Only the images are used; build_sample's labels are ignored.
    ds = DriftSenseDataset(args.dirs, crop=args.crop, train=False,
                           limit=args.batches * args.batch_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=True)

    bns = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    print(f"weights   : {args.weights}")
    print(f"device    : {device}")
    print(f"BN layers : {len(bns)}")
    print(f"frames    : {len(ds)} at {args.crop}px, batch {args.batch_size}")

    # momentum=None makes each layer accumulate a cumulative moving average,
    # i.e. the exact mean over every batch seen, rather than an exponentially
    # decaying one that would over-weight the last few batches.
    for m in bns:
        m.reset_running_stats()
        m.momentum = None

    model.train()
    done = 0
    with torch.no_grad():
        for batch in loader:
            model(batch["template"].to(device), batch["search"].to(device))
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{len(loader)} batches", flush=True)
            if done >= args.batches:
                break

    # Restore the default momentum so the checkpoint behaves normally if it is
    # ever resumed for further training.
    for m in bns:
        m.momentum = 0.1
    model.eval()

    out = dict(ckpt) if isinstance(ckpt, dict) and "model" in ckpt else {}
    out["model"] = model.state_dict()
    out["bn_recalibrated_at"] = args.crop
    out["bn_recalibration_frames"] = done * args.batch_size
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"\nwrote {args.out}  ({done} batches, {done * args.batch_size} frames)")


if __name__ == "__main__":
    main()

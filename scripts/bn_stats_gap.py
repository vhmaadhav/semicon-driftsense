#!/usr/bin/env python3
"""Measure how far BatchNorm's stored statistics are from the truth at 1000 px.

The model trains on 512 px crops and infers on 1000 px frames. BN applies the
running mean/variance it accumulated at 512 to activations produced at 1000. If
those two distributions differ, every BN layer is mis-normalising at inference.

This script forward-hooks each BN layer, records the *actual* per-channel batch
statistics at both frame sizes, and reports the discrepancy against the stored
running stats. Output is a per-layer table of

    d_mean = mean_ch | batch_mean - running_mean | / sqrt(running_var)
    r_var  = mean_ch   batch_var  / running_var

both expressed in units where 0 / 1.0 means "the stored statistics are correct".
A layer whose r_var at 1000 sits well away from 1.0 while its r_var at 512 sits
near it is one the resolution change is actively breaking.

    python scripts/bn_stats_gap.py --weights weights/driftsense.pt --n 6
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from driftsense.dataset import DriftSenseDataset  # noqa: E402
from driftsense.model import DriftSenseNet, net_from_checkpoint  # noqa: E402


def collect(model, ds, idxs, device, frame: int):
    """Return {layer_name: (d_mean, r_var)} averaged over the given samples.

    `frame` slices the search image down to that size. The dataset itself only
    crops when train=True (which would also switch on photometric jitter, and
    that inflates activation variance independently of frame size), so the
    slice is done here instead -- leaving frame size as the single variable.
    """
    acc: dict[str, list] = {}
    handles = []

    def hook(name):
        def fn(mod, inp, _out):
            x = inp[0].detach().float()
            # BN normalises over (N, H, W) per channel.
            dims = [0] + list(range(2, x.dim()))
            bm = x.mean(dim=dims)
            bv = x.var(dim=dims, unbiased=False)
            rm, rv = mod.running_mean, mod.running_var
            d_mean = float((((bm - rm).abs()) / rv.clamp_min(1e-8).sqrt()).mean())
            r_var = float((bv / rv.clamp_min(1e-8)).mean())
            acc.setdefault(name, []).append((d_mean, r_var))
        return fn

    for name, m in model.named_modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            handles.append(m.register_forward_hook(hook(name)))

    model.eval()  # eval mode: BN uses running stats, hooks read the true ones
    with torch.no_grad():
        for i in idxs:
            s = ds[i]
            sea = s["search"][None][:, :, :frame, :frame]
            model(s["template"][None].to(device), sea.to(device))

    for h in handles:
        h.remove()
    return {k: (float(np.mean([a for a, _ in v])), float(np.mean([b for _, b in v])))
            for k, v in acc.items()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="weights/driftsense.pt")
    p.add_argument("--dir", default="data/val")
    p.add_argument("--n", type=int, default=6)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    ckpt = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = net_from_checkpoint(ckpt)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.to(device)

    idxs = list(range(args.n))
    # train=False so photometric jitter never enters; the search image is then
    # sliced in collect(), leaving frame size as the only variable.
    ds = DriftSenseDataset([args.dir], crop=1000, train=False)
    at512 = collect(model, ds, idxs, device, frame=512)
    at1000 = collect(model, ds, idxs, device, frame=1000)

    print(f"weights: {args.weights}   samples: {args.n}   (train=False, no augmentation)\n")
    print(f"{'BN layer':<34}{'d_mean@512':>12}{'d_mean@1000':>13}"
          f"{'var@512':>10}{'var@1000':>11}")
    print("-" * 80)
    worst = []
    for name in at1000:
        # crop 512 does not exercise a full-frame-only path, so keys match.
        dm5, rv5 = at512.get(name, (float("nan"), float("nan")))
        dm10, rv10 = at1000[name]
        print(f"{name:<34}{dm5:>12.3f}{dm10:>13.3f}{rv5:>10.3f}{rv10:>11.3f}")
        worst.append((abs(np.log(max(rv10, 1e-8))) - abs(np.log(max(rv5, 1e-8))), name, rv5, rv10))
    print("-" * 80)
    print("d_mean: |batch mean - running mean| / running std   (0 = stored stats correct)")
    print("var   : batch var / running var                     (1 = stored stats correct)\n")

    worst.sort(reverse=True)
    print("layers most degraded by the 512 -> 1000 change:")
    for delta, name, rv5, rv10 in worst[:5]:
        print(f"  {name:<32} var ratio {rv5:.3f} -> {rv10:.3f}")


if __name__ == "__main__":
    main()

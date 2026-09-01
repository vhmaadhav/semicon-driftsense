#!/usr/bin/env python3
"""Is the model data-limited, or capacity-limited?

The two have opposite fixes and we have three days, so guessing is expensive.
Training loss on p9 fell steadily (0.3548 -> 0.3371 over 30 epochs) while
held-out set B credit moved +0.006. That is consistent with overfitting, and
also with a model that is simply near its ceiling. The distinguishing evidence
is the training objective evaluated on shards the model never saw.

`data/ext_holdout/` was kept out of every training run, and it is in the same
100 px template format as the training pool, so the *training* loss can be
computed on it directly -- unlike locate_phase2, which needs full references.

  seen loss ~= unseen loss   -> not overfitting; more data will not help much,
                                the lever is capacity or the recipe
  unseen loss >> seen loss   -> overfitting; more data is the lever, and we hold
                                167 of 991 available shards
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def mean_loss(model, ds, device, n, bs=16):
    from driftsense.engine import compute_loss
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)
    tot, parts, seen = 0.0, {"focal": 0.0, "offset": 0.0}, 0
    model.eval()
    with torch.no_grad():
        for batch in dl:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out = model(batch["template"], batch["search"])
            loss, p = compute_loss(out, batch)
            b = batch["template"].shape[0]
            tot += float(loss) * b
            for k in parts:
                parts[k] += p[k] * b
            seen += b
            if seen >= n:
                break
    return tot / seen, {k: v / seen for k, v in parts.items()}, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/driftsense.pt")
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import infer as I
    from driftsense.dataset import DriftSenseDataset

    model, device = I.load_model(a.weights)
    seen_dirs = sorted(__import__("glob").glob(os.path.join(HERE, "data/ext_train/B_00*/")))[:4]
    hold_dirs = sorted(__import__("glob").glob(os.path.join(HERE, "data/ext_holdout/*/")))
    if not hold_dirs:
        sys.exit("no data/ext_holdout shards -- nothing was held out")

    print(f"weights: {a.weights}")
    print(f"seen   : {len(seen_dirs)} shards from the training pool")
    print(f"unseen : {len(hold_dirs)} shards never trained on\n")

    # train=True: with train=False the search frame is left at full size and
    # batches cannot collate. Both sets get identical treatment and a fixed
    # seed, so augmentation adds variance but no bias to the comparison.
    out = {}
    for label, dirs in (("seen (trained on)", seen_dirs), ("unseen (holdout)", hold_dirs)):
        ds = DriftSenseDataset(dirs, crop=512, train=True, seed=0)
        L, parts, n = mean_loss(model, ds, device, a.n)
        out[label] = L
        print(f"  {label:<20} loss {L:.4f}   focal {parts['focal']:.4f}   "
              f"offset {parts['offset']:.4f}   ({n} samples)")

    gap = out["unseen (holdout)"] - out["seen (trained on)"]
    rel = 100 * gap / out["seen (trained on)"]
    print(f"\n  generalisation gap: {gap:+.4f}  ({rel:+.1f}%)")
    if rel > 8:
        print("  => OVERFITTING. More data is the lever; we hold 167 of 991 shards.")
    elif rel > 3:
        print("  => mild overfitting. More data should help; regularisation may too.")
    else:
        print("  => NOT overfitting. More data will not help much -- the lever is")
        print("     capacity or the training recipe (EMA, LR schedule, weight decay).")


if __name__ == "__main__":
    main()

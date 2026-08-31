#!/usr/bin/env python3
"""Average same-init fine-tunes into one checkpoint (issue #10).

`p6_last`, `p8_last` and `p9_last` are fine-tunes from the same initialisation;
p8 and p9 differ only in `--jitter-power`. That is the setup model soups
(Wortsman et al. 2022, arXiv:2203.05482) are built for: averaging the weights of
same-init fine-tunes often beats the best individual member, and costs nothing
at inference because the result is just another checkpoint.

Averaged in float64 to avoid accumulating rounding across members, then cast
back. Integer buffers -- BatchNorm's `num_batches_tracked` -- are counters, not
quantities to average, so the first member's value is kept.

  ./venv-train/bin/python scripts/checkpoint_soup.py \
      weights/driftsense_p6_last.pt weights/driftsense_p8_last.pt \
      weights/driftsense_p9_last.pt --out weights/soup_all.pt
"""
from __future__ import annotations

import argparse
import os

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("members", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    states, base = [], None
    for m in a.members:
        ck = torch.load(m, map_location="cpu", weights_only=False)
        sd = ck["model"]
        if base is None:
            base = ck
        states.append(sd)
        print(f"  loaded {os.path.basename(m)}  ({len(sd)} tensors)")

    keys = set(states[0])
    for sd in states[1:]:
        if set(sd) != keys:
            raise SystemExit("checkpoints have different parameter sets -- "
                             "these are not the same architecture")

    soup, skipped = {}, 0
    for k in states[0]:
        vals = [sd[k] for sd in states]
        if not vals[0].is_floating_point():
            soup[k] = vals[0].clone()      # counters, not quantities
            skipped += 1
            continue
        acc = torch.zeros_like(vals[0], dtype=torch.float64)
        for v in vals:
            acc += v.to(torch.float64)
        soup[k] = (acc / len(vals)).to(vals[0].dtype)

    base["model"] = soup
    base.pop("raw_model", None)
    base["soup_members"] = [os.path.basename(m) for m in a.members]
    torch.save(base, a.out)
    print(f"\nwrote {a.out}  ({len(soup)} tensors averaged over {len(states)} "
          f"members, {skipped} integer buffers copied)")


if __name__ == "__main__":
    main()

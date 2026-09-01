#!/usr/bin/env python3
"""Average the weights of several checkpoints into one model (SWA / "model soup").

Why this is worth trying here
-----------------------------
A one-cycle run ends by annealing the LR to ~0, so the last few epochs are
different points in the same flat basin rather than genuinely different models.
Averaging them lands nearer the centre of that basin, which usually generalises
slightly better than any single point on its rim. It costs one pass over some
state dicts -- no training, no labels, no gradients.

Two things this is NOT:

* It is not the 2-checkpoint *ensemble* that NIGHT_LOG.md rejected. That ran
  two models at inference and arbitrated between their proposals; this produces
  a single model with the same parameter count and the same inference cost.
* It is not free of risk. Averaging weights from different basins produces
  garbage, so only average checkpoints from one continuous run, and always
  measure -- `scripts/stream_eval.py` then `scripts/compare_checkpoints.py`.

BatchNorm running statistics are averaged along with everything else. That is
the right call here rather than re-estimating them: NIGHT_LOG.md measured BN
recalibration on clean frames and it cost a point, because the stored stats are
the average of *augmented* batch statistics and reproducing the training-time
normaliser is the consistent thing to do.

    python scripts/average_checkpoints.py weights/driftsense_v5f_e1[6-9].pt \
        weights/driftsense_v5f_e2*.pt --out weights/driftsense_v5f_swa.pt
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from driftsense.model import DriftSenseNet, net_from_checkpoint  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoints", nargs="+", help="paths or globs")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    paths = []
    for c in args.checkpoints:
        hits = sorted(glob.glob(c))
        paths.extend(hits if hits else [c])
    paths = [q for q in paths if os.path.exists(q)]
    if len(paths) < 2:
        raise SystemExit(f"need at least 2 checkpoints, got {len(paths)}: {paths}")

    if os.path.abspath(args.out) in {os.path.abspath(q) for q in paths}:
        raise SystemExit("--out would overwrite one of the inputs")

    print(f"averaging {len(paths)} checkpoints:")
    acc, meta = None, None
    for q in paths:
        ck = torch.load(q, map_location=args.device, weights_only=False)
        sd = ck.get("model", ck)
        val = ck.get("val", {})
        print(f"  {os.path.basename(q):<34} epoch {str(ck.get('epoch', '?')):>3}  "
              f"acc@5 {val.get('acc@5px', float('nan')):.3f}  "
              f"median {val.get('median_px', float('nan')):.2f}px")
        if acc is None:
            acc = {k: v.detach().clone().to(torch.float64) if v.is_floating_point()
                   else v.detach().clone() for k, v in sd.items()}
            meta = ck
        else:
            if sd.keys() != acc.keys():
                raise SystemExit(f"{q} has a different parameter set -- different arch?")
            for k, v in sd.items():
                if acc[k].is_floating_point():
                    acc[k] += v.to(torch.float64)
                else:
                    # num_batches_tracked and friends: integer counters, not
                    # parameters. Keep the first rather than averaging.
                    pass

    n = len(paths)
    averaged = {}
    for k, v in acc.items():
        ref = meta["model"][k] if "model" in meta else meta[k]
        averaged[k] = (v / n).to(ref.dtype) if v.is_floating_point() else ref.clone()

    # Load into the real module so a shape or name mismatch fails here rather
    # than at evaluation time.
    model = DriftSenseNet()
    model.load_state_dict(averaged)

    out = {"model": averaged, "arch": "DriftSenseNet",
           "crop": meta.get("crop"), "epoch": meta.get("epoch"),
           "averaged_from": [os.path.basename(q) for q in paths]}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(out, args.out)
    print(f"\nwrote {args.out}  ({n} checkpoints averaged)")
    print("Now measure it -- averaging is not guaranteed to help:")
    print(f"  python scripts/stream_eval.py --weights {args.out} -n 1000 "
          f"--workers 6 --out results/stream/swa.json")


if __name__ == "__main__":
    main()

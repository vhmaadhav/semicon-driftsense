#!/usr/bin/env python3
"""Assert that consecutive epochs really do produce different training tensors.

This exists because the project has already been bitten once by exactly this
bug and it is silent: `driftsense/stream_dataset.py` relied on `set_epoch()` to
advance its seed stream while `train.py` built the DataLoader with
`persistent_workers=True`, so the workers kept a frozen copy of the dataset,
`set_epoch()` never reached them, and "unlimited fresh data" was a fixed
16 000-scene pool replayed every epoch (NIGHT_LOG.md). It only surfaced days
later, from a loss discrepancy on a resumed run.

`DriftSenseDataset.set_epoch()` now re-rolls the augmentation for a second pass
over a disk pool, and it is reachable by the same trap. So this checks it the
way the streaming fix was checked -- by hashing the tensors that actually come
out of the loader -- rather than by reading the code.

    python scripts/check_epoch_variety.py --dir data/val
    python scripts/check_epoch_variety.py --dir data/train_pool --workers 4
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import cv2
import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from driftsense.dataset import DriftSenseDataset  # noqa: E402
from driftsense.stream_dataset import StreamingDriftSense  # noqa: E402


def _init(_):
    cv2.setNumThreads(0)


def digest(batch) -> str:
    h = hashlib.sha1()
    for k in ("template", "search", "peak", "offset"):
        h.update(batch[k].numpy().tobytes())
    return h.hexdigest()[:12]


def epoch_digests(ds, loader, epochs: int, batches: int) -> list[list[str]]:
    out = []
    for e in range(epochs):
        ds.set_epoch(e)
        got = []
        for i, b in enumerate(loader):
            if i >= batches:
                break
            got.append(digest(b))
        out.append(got)
    return out


def report(title: str, digs: list[list[str]]) -> bool:
    print(f"\n{title}")
    for e, d in enumerate(digs):
        print(f"  epoch {e}: {' '.join(d)}")
    ok = True
    for a in range(len(digs)):
        for b in range(a + 1, len(digs)):
            same = sum(x == y for x, y in zip(digs[a], digs[b]))
            verdict = "FROZEN -- epochs are identical" if same == len(digs[a]) else "fresh"
            if same == len(digs[a]):
                ok = False
            print(f"  epoch {a} vs {b}: {same}/{len(digs[a])} batches identical  -> {verdict}")
    return ok


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default="data/val")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batches", type=int, default=3)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--skip-stream", action="store_true")
    args = p.parse_args()

    ok = True

    # --- on-disk pool: shuffle off, so any difference is the augmentation ----
    ds = DriftSenseDataset([args.dir], crop=args.crop, train=True, seed=0)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, drop_last=True,
                        worker_init_fn=_init, persistent_workers=False)
    ok &= report(f"DriftSenseDataset({args.dir}) -- persistent_workers=False "
                 f"(what train.py uses)", epoch_digests(ds, loader, args.epochs, args.batches))

    # The failure mode itself, demonstrated rather than described. Expected to
    # report FROZEN; that is the point, and it is why train.py must not use it.
    if args.workers > 0:
        ds2 = DriftSenseDataset([args.dir], crop=args.crop, train=True, seed=0)
        loader2 = DataLoader(ds2, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.workers, drop_last=True,
                             worker_init_fn=_init, persistent_workers=True)
        digs = epoch_digests(ds2, loader2, args.epochs, args.batches)
        frozen = all(d == digs[0] for d in digs)
        print(f"\nDriftSenseDataset -- persistent_workers=True (the trap)")
        print(f"  epochs identical: {frozen}  "
              f"-> {'as expected; train.py avoids this' if frozen else 'unexpected'}")
        del loader2

    if not args.skip_stream:
        sds = StreamingDriftSense(length=args.batch_size * args.batches * max(args.workers, 1),
                                  crop=args.crop, seed=0)
        sloader = DataLoader(sds, batch_size=args.batch_size, num_workers=args.workers,
                             drop_last=True, worker_init_fn=_init,
                             persistent_workers=False)
        ok &= report("StreamingDriftSense (regression check on the original bug)",
                     epoch_digests(sds, sloader, args.epochs, args.batches))

    print("\nPASS" if ok else "\nFAIL -- an epoch replayed identical samples")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train the Drift-Sense Siamese localiser.

Example (what produced the shipped weights):

    python train.py --train-dirs data/train_mc data/train --val-dir data/val \
        --crop 512 --batch-size 8 --epochs 24 --lr 1e-3 --out weights/driftsense.pt

Training runs on a 512 px window of the search frame; the network is fully
convolutional, so inference runs on the full 1000x1000 frame unchanged. A
short fine-tune at full frame size (--crop 1000) at the end closes the small
train/test gap in how much decoy context the head sees.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler

from driftsense.dataset import DriftSenseDataset, load_manifest
from driftsense.stream_dataset import StreamingDriftSense
from driftsense.engine import compute_loss, decode_batch, evaluate
from driftsense.model import DriftSenseNet


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def worker_init(_):
    # Each worker doing its own OpenCV threading oversubscribes the machine.
    cv2.setNumThreads(0)


def scan_pool(dirs: list[str]) -> list[str]:
    """Expand --train-dirs into readable split directories.

    A directory holding a manifest.csv is used as-is, which is how every
    existing split (data/train, data/val, ...) behaves. A directory holding
    *shards* instead contributes each subdirectory that has a COMPLETE marker,
    so a shard still being written by scripts/build_pool.py is never half-read.
    """
    out = []
    for d in dirs:
        if os.path.exists(os.path.join(d, "manifest.csv")):
            out.append(d)
            continue
        for sub in sorted(glob.glob(os.path.join(d, "*"))):
            if os.path.exists(os.path.join(sub, "COMPLETE")):
                out.append(sub)
    return out


def tune_cuda(device: torch.device) -> None:
    """CUDA-only throughput switches. No effect on MPS/CPU runs.

    Shapes are fixed for a whole run, so cudnn's autotuner pays for itself in
    the first few steps. TF32 and bf16 affect numerics, so the evaluation path
    deliberately stays in fp32 (see the autocast scope in main): accuracy
    numbers have to stay comparable with the Mac's, and the parity gate in
    PORT.md checks exactly that.
    """
    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dirs", nargs="+", default=["data/train_mc", "data/train"])
    p.add_argument("--val-dir", default="data/val")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="weights/driftsense.pt")
    p.add_argument("--resume", default="")
    p.add_argument("--finetune", action="store_true",
                   help="with --resume, load the model weights only: fresh "
                        "optimizer, fresh one-cycle at --lr, epoch counter from 0. "
                        "Use this to train on top of a finished run; plain --resume "
                        "is for continuing an interrupted one and would restore a "
                        "schedule that has already annealed to zero.")
    p.add_argument("--val-limit", type=int, default=60, help="full-frame val samples per epoch")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--limit", type=int, default=0, help="cap training pairs (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stream", action="store_true",
                   help="generate fresh scenes on the fly instead of reading a fixed "
                        "dataset -- unlimited data, no disk, no scene ever repeated")
    p.add_argument("--stream-length", type=int, default=14000,
                   help="samples per nominal epoch when streaming")
    p.add_argument("--keep-epochs", action="store_true",
                   help="also write an unoverwritten checkpoint per epoch as "
                        "{out}_e{N}.pt. The in-loop val metric runs on a subset "
                        "without TTA, which is too coarse to separate epochs that "
                        "differ by a few samples; keeping them all allows the "
                        "final pick to be made on the full split afterwards.")
    p.add_argument("--amp", action="store_true",
                   help="bf16 autocast for the training step (CUDA only). bf16 has "
                        "fp32's exponent range so no GradScaler is needed. The "
                        "evaluation path stays fp32.")
    p.add_argument("--samples-per-epoch", type=int, default=0,
                   help="draw this many pairs from the pool per epoch instead of "
                        "walking all of it (0 = the whole pool). A large pool makes "
                        "one full pass long enough that per-epoch validation and "
                        "checkpointing get too coarse; this keeps the epoch a fixed "
                        "unit while a fresh random subset is drawn each time, so the "
                        "run still covers the pool.")
    p.add_argument("--prefetch-factor", type=int, default=4,
                   help="batches each worker runs ahead (CUDA only)")
    p.add_argument("--refresh-pool", action="store_true",
                   help="re-scan --train-dirs for newly COMPLETE shards at every "
                        "epoch boundary, so training can start before generation "
                        "has finished. Requires --samples-per-epoch, so the step "
                        "count per epoch stays fixed as the pool grows.")
    p.add_argument("--vram-fraction", type=float, default=0.92,
                   help="hard cap on VRAM as a fraction of the card (CUDA only). "
                        "Windows WDDM does not fail cleanly when VRAM runs out -- it "
                        "spills into system RAM and the step slows ~20x while still "
                        "reporting 100%% GPU utilisation. Measured here at crop 768 / "
                        "batch 32: free RAM fell to 0.9 GB and the run never "
                        "finished. Capping turns that into an honest OOM. Set 0 to "
                        "disable.")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    tune_cuda(device)
    if device.type == "cuda" and args.vram_fraction:
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
    amp = args.amp and device.type == "cuda"
    print(f"device: {device}   amp: {'bf16' if amp else 'off'}")
    if args.refresh_pool and not args.samples_per_epoch:
        raise SystemExit("--refresh-pool needs --samples-per-epoch: otherwise the "
                         "epoch length changes as shards land and the one-cycle "
                         "schedule's total_steps is wrong from the first refresh.")

    if args.stream:
        train_ds = StreamingDriftSense(length=args.stream_length, crop=args.crop,
                                       seed=args.seed)
        print(f"STREAMING: {args.stream_length} freshly generated samples/epoch "
              f"(no scene reused)   crop: {args.crop}")
    else:
        pool_dirs = scan_pool(args.train_dirs)
        if not pool_dirs:
            raise SystemExit(f"no readable splits under {args.train_dirs} "
                             f"(a shard needs a COMPLETE marker)")
        train_ds = DriftSenseDataset(pool_dirs, crop=args.crop, train=True,
                                     seed=args.seed, limit=args.limit or None)
        print(f"train pairs: {len(train_ds)} from {len(pool_dirs)} split(s)   "
              f"crop: {args.crop}")
    val_rows = load_manifest(args.val_dir)
    print(f"val scenes: {len(val_rows)}")

    # A fresh random subset per epoch when the pool is bigger than one epoch.
    sampler = None
    if not args.stream and args.samples_per_epoch:
        n = min(args.samples_per_epoch, len(train_ds))
        sampler = RandomSampler(train_ds, replacement=False, num_samples=n)
        print(f"sampling {n} of {len(train_ds)} pairs per epoch "
              f"({args.epochs * n / max(len(train_ds), 1):.1f} passes over the pool)")

    loader_kw = {}
    if device.type == "cuda" and args.workers > 0:
        loader_kw = {"pin_memory": True, "prefetch_factor": args.prefetch_factor}

    loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=(sampler is None and not args.stream), sampler=sampler,
        num_workers=args.workers, drop_last=True,
        worker_init_fn=worker_init,
        # Never persistent. Both dataset classes advance their randomness via
        # set_epoch(), and persistent workers keep a frozen copy of the dataset,
        # so set_epoch() never reaches them -- that is what silently turned
        # "unlimited fresh data" into a fixed 16 000-scene pool on repeat
        # (NIGHT_LOG.md), and the same trap now applies to the on-disk pool's
        # per-epoch re-augmentation. Respawning workers costs seconds against a
        # multi-minute epoch. scripts/check_epoch_variety.py verifies it holds.
        persistent_workers=False,
        **loader_kw,
    )

    model = DriftSenseNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params/1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = max(len(loader) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=steps, pct_start=0.15)

    start_epoch, best = 0, float("inf")
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if args.finetune:
            # Weights only. --resume is for continuing an interrupted run: it
            # restores the optimizer and fast-forwards the scheduler to the
            # checkpoint's global_step, which is right for that case and wrong
            # here. The shipped model's one-cycle already ran to completion with
            # the LR annealed to ~0, so restoring it would resume a schedule
            # that is over, and its epoch counter would eat most of --epochs.
            # A fine-tune wants a fresh optimizer and a fresh one-cycle at a
            # lower max_lr, which is what phases 2 and 3 did (NIGHT_LOG.md).
            # `best` resets too, so best-checkpoint selection is scored on this
            # run's own scale rather than against a number from another split.
            print(f"fine-tuning from {args.resume}: weights only "
                  f"(fresh optimizer, fresh one-cycle at max_lr {args.lr:g}, "
                  f"epoch counter from 0)")
        elif "optimizer" in ck and ck.get("crop") == args.crop:
            opt.load_state_dict(ck["optimizer"])
            for _ in range(ck.get("global_step", 0)):
                sched.step()
            start_epoch = ck.get("epoch", 0)
        best = ck.get("best", float("inf"))
        print(f"resumed from {args.resume} (epoch {start_epoch}, best {best:.3f}px)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    history = []
    global_step = start_epoch * len(loader)

    for epoch in range(start_epoch, args.epochs):
        # Streaming: a fresh seed stream, so no scene is ever repeated.
        # On-disk pool: a fresh roll of the dihedral / photometric / window-crop
        # augmentation, so a second pass over the pool is a new view of it
        # rather than a literal replay.
        train_ds.set_epoch(epoch)
        if args.refresh_pool:
            now = scan_pool(args.train_dirs)
            if len(now) != len(pool_dirs):
                before = len(train_ds)
                pool_dirs = now
                n = train_ds.reload(pool_dirs, args.limit or None)
                print(f"  pool grew: {len(pool_dirs)} shards, {before} -> {n} pairs "
                      f"({args.epochs * (args.samples_per_epoch) / max(n, 1):.1f} "
                      f"passes at the current size)", flush=True)
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "focal": 0.0, "offset": 0.0}
        seen = 0

        for it, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                out = model(batch["template"], batch["search"])
            # Loss in fp32, deliberately. The focal term evaluates
            # log(1 - sigmoid(logit)) on ~10^4 near-zero negatives per sample;
            # in bf16's 8-bit mantissa `1 - p` collapses to 0 for confident
            # negatives and the log goes to -inf. The convolutions are where
            # the bf16 speedup is, and they keep it.
            if amp:
                out = {k: v.float() for k, v in out.items()}
            loss, parts = compute_loss(out, batch)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if global_step < steps - 1:
                sched.step()
            global_step += 1

            agg["loss"] += float(loss.detach())
            agg["focal"] += parts["focal"]
            agg["offset"] += parts["offset"]
            seen += 1

            if args.log_every and (it + 1) % args.log_every == 0:
                pred = decode_batch(out, (args.crop, args.crop))
                gt = batch["gt"].float().cpu().numpy()
                err = np.hypot(pred[:, 0] - gt[:, 0], pred[:, 1] - gt[:, 1])
                rate = (it + 1) * args.batch_size / (time.time() - t0)
                print(f"  e{epoch} [{it+1}/{len(loader)}] loss {agg['loss']/seen:.3f} "
                      f"focal {agg['focal']/seen:.3f} off {agg['offset']/seen:.4f} "
                      f"| batch median err {np.median(err):.1f}px | {rate:.1f} img/s "
                      f"| lr {sched.get_last_lr()[0]:.2e}", flush=True)

        val = evaluate(model, val_rows, device, limit=args.val_limit, refine=True)
        dt = time.time() - t0
        print(f"epoch {epoch}: loss {agg['loss']/max(seen,1):.4f} | "
              f"val median {val['median_px']:.2f}px  acc@1 {val['acc@1px']:.3f}  "
              f"acc@2 {val['acc@2px']:.3f}  acc@5 {val['acc@5px']:.3f}  "
              f"acc@10 {val['acc@10px']:.3f} | {dt/60:.1f} min", flush=True)

        history.append({"epoch": epoch, "loss": agg["loss"] / max(seen, 1),
                        **{k: v for k, v in val.items() if k not in ("dists", "scores")}})

        # Select on accuracy at the operating tolerance, breaking ties by
        # median error -- median alone would happily trade a few catastrophic
        # decoy lock-ons for slightly tighter sub-pixel placement.
        score = (1.0 - val["acc@5px"]) * 1000.0 + val["median_px"]
        ckpt = {
            "model": model.state_dict(), "optimizer": opt.state_dict(),
            "epoch": epoch + 1, "global_step": global_step, "best": min(best, score),
            "crop": args.crop, "val": {k: v for k, v in val.items() if k not in ("dists", "scores")},
            "arch": "DriftSenseNet",
        }
        torch.save(ckpt, args.out.replace(".pt", "_last.pt"))
        if args.keep_epochs:
            torch.save(ckpt, args.out.replace(".pt", f"_e{epoch}.pt"))
        if score < best:
            best = score
            ckpt["best"] = best
            torch.save(ckpt, args.out)
            print(f"  saved {args.out} (acc@5 {val['acc@5px']:.3f}, median {val['median_px']:.2f}px)")

        with open(args.out.replace(".pt", "_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    print("done. best score:", best)


if __name__ == "__main__":
    main()

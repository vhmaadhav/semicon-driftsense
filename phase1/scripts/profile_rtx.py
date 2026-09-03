#!/usr/bin/env python3
"""Measure the three numbers that decide --batch-size, --workers and the data strategy.

PORT.md step 4. Nothing here trains or writes weights.

    python scripts/profile_rtx.py gpu       # step throughput + peak VRAM, synthetic tensors
    python scripts/profile_rtx.py gen       # scene generation pairs/s vs worker count, with RSS
    python scripts/profile_rtx.py loader    # on-disk DriftSenseDataset throughput

The `gpu` mode also A/B tests a vectorised grouped cross-correlation against the
per-sample Python loop in driftsense.model, and asserts the two agree, because
that loop issues one conv2d launch per batch element and is the obvious
candidate for the step being launch-bound rather than compute-bound.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

from driftsense.model import (  # noqa: E402
    CORR_GROUPS, STRIDE, TEMPLATE_FEAT, TEMPLATE_SIZE, DriftSenseNet, grouped_xcorr,
)


def xcorr_vectorised(search_feat: torch.Tensor, template_feat: torch.Tensor,
                     groups: int = CORR_GROUPS) -> torch.Tensor:
    """Same maths as grouped_xcorr, as a single grouped conv2d.

    Folds the batch into the group dimension: B*groups groups over a (1, B*C)
    input. Mathematically identical -- each output group still sees exactly its
    own sample's channels -- but one kernel launch instead of B.
    """
    b, c, hs, ws = search_feat.shape
    ht, wt = template_feat.shape[-2:]
    x = search_feat.reshape(1, b * c, hs, ws)
    w = template_feat.reshape(b * groups, c // groups, ht, wt)
    out = F.conv2d(x, w, groups=b * groups)
    return out.reshape(b, groups, out.shape[-2], out.shape[-1])


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def make_batch(bs: int, crop: int, dev):
    resp = crop // STRIDE - TEMPLATE_FEAT + 1
    return {
        "template": torch.randn(bs, 1, TEMPLATE_SIZE, TEMPLATE_SIZE, device=dev),
        "search": torch.randn(bs, 1, crop, crop, device=dev),
        "heat": torch.rand(bs, 1, resp, resp, device=dev),
        "offset": torch.randn(bs, 2, device=dev),
        "peak": torch.randint(0, resp, (bs, 2), device=dev),
    }


def profile_gpu(args):
    from driftsense.engine import compute_loss

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")
    if dev.type == "cuda":
        print(f"gpu   : {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Windows WDDM does NOT hard-fail when VRAM runs out: the driver spills
        # allocations into system RAM and the step slows by ~20x while staying
        # at "100% utilisation". Measured here at crop 768 / batch 32 -- the
        # machine dropped to 0.9 GB free RAM and the profile never finished.
        # A hard cap turns that silent cliff into a normal OOM we can catch.
        torch.cuda.set_per_process_memory_fraction(args.vram_fraction)
        print(f"vram cap: {args.vram_fraction:.2f} of "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB "
              f"(prevents WDDM spilling into system RAM)")
    print(f"torch : {torch.__version__}\n", flush=True)

    # --- correctness + speed of the vectorised cross-correlation -------------
    torch.manual_seed(0)
    b, c, hs, ws = 8, 64, 128, 128
    sf = torch.randn(b, c, hs, ws, device=dev)
    tf = torch.randn(b, c, TEMPLATE_FEAT, TEMPLATE_FEAT, device=dev)
    a = grouped_xcorr(sf, tf)
    v = xcorr_vectorised(sf, tf)
    err = (a - v).abs().max().item()
    scale = a.abs().max().item()
    print(f"xcorr vectorised vs loop: max abs diff {err:.3e} "
          f"(values up to {scale:.2f}, rel {err/max(scale,1e-9):.2e})", flush=True)
    for name, fn in (("loop", grouped_xcorr), ("vectorised", xcorr_vectorised)):
        for _ in range(3):
            fn(sf, tf)
        _sync(dev)
        t0 = time.perf_counter()
        for _ in range(30):
            fn(sf, tf)
        _sync(dev)
        print(f"  {name:<11} {(time.perf_counter()-t0)/30*1000:.2f} ms/call", flush=True)
    print(flush=True)

    # --- full training step --------------------------------------------------
    print(f"{'crop':>6}{'batch':>7}{'amp':>6}{'img/s':>10}{'ms/step':>10}{'peakVRAM':>11}")
    print("-" * 50, flush=True)
    for crop in args.crops:
        oom_at = None  # once a batch size OOMs, larger ones will too
        for bs in args.batches:
            if oom_at is not None and bs >= oom_at:
                print(f"{crop:>6}{bs:>7}{'-':>6}{'skipped':>10}  (>= OOM at {oom_at})",
                      flush=True)
                continue
            for amp in ([False, True] if args.amp else [False]):
                model = DriftSenseNet().to(dev)
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                try:
                    batch = make_batch(bs, crop, dev)
                    for it in range(args.iters + args.warmup):
                        if it == args.warmup:
                            _sync(dev)
                            t0 = time.perf_counter()
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
                            out = model(batch["template"], batch["search"])
                            loss, _ = compute_loss(out, batch)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                        opt.step()
                    _sync(dev)
                    dt = (time.perf_counter() - t0) / args.iters
                    peak = (torch.cuda.max_memory_allocated() / 2**30) if dev.type == "cuda" else 0.0
                    print(f"{crop:>6}{bs:>7}{str(amp):>6}{bs/dt:>10.1f}{dt*1000:>10.1f}"
                          f"{peak:>10.2f}G", flush=True)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    print(f"{crop:>6}{bs:>7}{str(amp):>6}{'OOM':>10}", flush=True)
                    oom_at = min(oom_at, bs) if oom_at else bs
                finally:
                    del model, opt
                    if dev.type == "cuda":
                        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------


def _cv2_single_thread(_):
    # Module level, not a closure: Windows spawns dataloader workers and pickles
    # worker_init_fn by reference, so a local function fails to start the worker.
    import cv2
    cv2.setNumThreads(0)


def _gen_task(job):
    seed, n, crops = job
    import numpy as _np
    from driftsense.generate import PRESETS as _P, make_pairs as _mp
    rng = _np.random.default_rng(seed)
    arch = list(_P.keys())
    t0 = time.perf_counter()
    total = 0
    for _ in range(n):
        total += len(_mp(int(rng.integers(0, 2**63 - 1)), arch, "randomized", crops=crops))
    return total, time.perf_counter() - t0


def profile_gen(args):
    """Scene-generation throughput vs process count, with peak system RAM used."""
    import subprocess
    from concurrent.futures import ProcessPoolExecutor

    import cv2
    cv2.setNumThreads(0)

    def mem_used_gb():
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "$o=Get-CimInstance Win32_OperatingSystem;"
             "[math]::Round(($o.TotalVisibleMemorySize-$o.FreePhysicalMemory)/1MB,2)"],
            capture_output=True, text=True)
        try:
            return float(out.stdout.strip())
        except ValueError:
            return float("nan")

    print(f"cpu count: {os.cpu_count()}   crops/canvas: {args.crops_per_canvas}")
    print(f"baseline RAM in use: {mem_used_gb():.2f} GB\n")
    print(f"{'workers':>8}{'canvases':>10}{'pairs/s':>10}{'steady':>9}"
          f"{'canvas/s':>10}{'core-s/pair':>13}{'peakRAM':>10}")
    print("-" * 71, flush=True)

    for nw in args.workers:
        per = max(args.canvases // nw, 1)
        jobs = [(1000 + w, per, args.crops_per_canvas) for w in range(nw)]
        t0 = time.perf_counter()
        peak = 0.0
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = [ex.submit(_gen_task, j) for j in jobs]
            while not all(f.done() for f in futs):
                peak = max(peak, mem_used_gb())
                time.sleep(2.0)
            res = [f.result() for f in futs]
        wall = time.perf_counter() - t0
        pairs = sum(r[0] for r in res)
        canv = per * nw
        # Windows spawns (not forks) workers, so each one re-imports
        # numpy/cv2/the generator -- seconds of startup that would understate a
        # long run's real rate. `steady` divides by the slowest worker's own
        # measured time instead, which excludes spawn: that is the rate a
        # multi-hour generation job actually sustains.
        busiest = max(r[1] for r in res)
        print(f"{nw:>8}{canv:>10}{pairs/wall:>10.1f}{pairs/busiest:>9.1f}"
              f"{canv/wall:>10.2f}{sum(r[1] for r in res)/max(pairs,1):>13.2f}"
              f"{peak:>9.2f}G", flush=True)


def profile_loader(args):
    """On-disk dataset throughput -- the ceiling if training reads a pool."""
    from torch.utils.data import DataLoader

    from driftsense.dataset import DriftSenseDataset

    ds = DriftSenseDataset([args.dir], crop=args.crop, train=True, seed=0)
    print(f"pool: {args.dir}  {len(ds)} pairs  crop {args.crop}\n")
    print(f"{'workers':>8}{'batch':>7}{'img/s':>10}")
    print("-" * 25)
    for nw in args.workers:
        dl = DataLoader(ds, batch_size=args.batch, num_workers=nw, shuffle=True,
                        drop_last=True, worker_init_fn=_cv2_single_thread,
                        persistent_workers=False,
                        pin_memory=torch.cuda.is_available())
        it = iter(dl)
        n = 0
        for _ in range(args.warmup):
            next(it)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            next(it)
            n += args.batch
        print(f"{nw:>8}{args.batch:>7}{n/(time.perf_counter()-t0):>10.1f}", flush=True)
        del it, dl


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("gpu")
    g.add_argument("--crops", type=int, nargs="+", default=[512, 768, 1000])
    g.add_argument("--batches", type=int, nargs="+", default=[8, 16, 32])
    g.add_argument("--iters", type=int, default=20)
    g.add_argument("--warmup", type=int, default=8)
    g.add_argument("--amp", action="store_true", default=True)
    g.add_argument("--vram-fraction", type=float, default=0.90,
                   help="hard cap on VRAM, as a fraction of the card. Below 1.0 so "
                        "an oversized config raises OOM instead of being silently "
                        "spilled into system RAM by the WDDM driver.")
    g.set_defaults(fn=profile_gpu)

    n = sub.add_parser("gen")
    n.add_argument("--workers", type=int, nargs="+", default=[4, 8, 12, 16])
    n.add_argument("--canvases", type=int, default=48)
    n.add_argument("--crops-per-canvas", type=int, default=8)
    n.set_defaults(fn=profile_gen)

    l = sub.add_parser("loader")
    l.add_argument("--dir", default="data/val")
    l.add_argument("--crop", type=int, default=512)
    l.add_argument("--batch", type=int, default=16)
    l.add_argument("--workers", type=int, nargs="+", default=[4, 6, 8])
    l.add_argument("--iters", type=int, default=20)
    l.add_argument("--warmup", type=int, default=3)
    l.set_defaults(fn=profile_loader)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()

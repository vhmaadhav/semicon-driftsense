# Porting Drift-Sense to the RTX laptop

Written 2026-08-17 for a move from the M-series Mac (16GB unified, MPS) to an
RTX 4060/4070 laptop (8GB VRAM, 16GB system RAM).

The submission in `weights/driftsense.pt` (acc@5px 0.9757) is **never
overwritten** by anything below. New runs write `driftsense_v5*`.

---

## Why the port needs a plan at all

Measured on the Mac, 2026-08-17:

| | measured |
| --- | --- |
| streaming training, 4 workers | 6.1 img/s → 44 min/epoch |
| pure scene generation, 4 workers | 7.6 pairs/s |
| cost per pair | ~0.6 CPU-core-seconds |
| disk per pair | 0.84 MB |
| model | 0.46M parameters |

Streaming training runs at 80% of the speed of generation alone, so the loop is
**data-generation-bound, not GPU-bound**. The laptop has the same 16GB of system
RAM, so a straight port that keeps `--stream` gains ~20-30%, not 8x — the GPU
would idle.

The fix is to take generation off the critical path: generate a large pool to
disk once, then train from disk, which is GPU-bound. `scripts/gen_data.py` and
`train.py --train-dirs` already support this, so **no dataloader code changes
are needed**.

---

## Step 0 — pack and copy (~1.8GB)

Excluded: `venv/` (macOS/MPS build), `data/train` and `data/train_mc` (13GB, and
the remaining headroom is all in fresh data), caches. Kept: code, `weights/`,
`data/val`, the three test splits, `results/` (including
`results/stream/driftsense.json`, the baseline for paired comparisons), and the
docs.

On the Mac — archive (no gzip; the payload is PNG):

```bash
cd /Users/sachin/Development
tar -cf ~/Desktop/semicon-port.tar \
  --exclude='semicon/venv' \
  --exclude='semicon/data/train' \
  --exclude='semicon/data/train_mc' \
  --exclude='__pycache__' --exclude='.pytest_cache' --exclude='.DS_Store' \
  semicon
```

Or straight over the network (WSL2 counts):

```bash
rsync -avh --progress \
  --exclude venv --exclude data/train --exclude data/train_mc \
  --exclude '__pycache__' --exclude '.pytest_cache' \
  /Users/sachin/Development/semicon/ USER@LAPTOP_IP:~/semicon/
```

## Step 1 — OS choice

**Use Ubuntu or WSL2, not native Windows.** Windows DataLoader workers use
`spawn`, so every worker re-imports numpy/cv2 and costs ~200MB more RAM — the
exact resource that is already short at 16GB.

## Step 2 — environment

`requirements.txt` pins `torch==2.13.0`, which resolves to the CPU/MPS wheel.
Install everything else first, then torch from the CUDA index:

```bash
cd ~/semicon
python3 -m venv venv
./venv/bin/pip install -U pip
grep -v '^torch' requirements.txt > requirements-cuda.txt
./venv/bin/pip install -r requirements-cuda.txt
./venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

(`cu128` for RTX 40/50-series. If the wheel is missing for this Python, check
what the CUDA index offers before downgrading Python.)

Verify, and record the machine's limits:

```bash
./venv/bin/python -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi; nproc; free -g; df -h .
```

`pick_device()` in `train.py` already falls through MPS → CUDA. No edit needed
to run.

## Step 3 — parity gate (do this before trusting any new number)

Eval scenes are generated on CPU from fixed seeds, so they are byte-identical
across machines: a paired Mac-vs-laptop comparison is a real test of whether the
CUDA port changed numerics.

```bash
./venv/bin/python scripts/stream_eval.py --weights weights/driftsense.pt \
    -n 1000 --out results/stream/driftsense_rtx.json
./venv/bin/python scripts/compare_checkpoints.py \
    --baseline results/stream/driftsense.json results/stream/driftsense_rtx.json
```

The bootstrap CI must contain zero. If it does not, stop and find out why before
training anything.

> **Correction (2026-08-17, made on the laptop).** The baseline named above was
> **`results/stream/driftsense.json`, acc@5px 0.933** — and that is the wrong
> file. Its `weights` field does say `weights/driftsense.pt`, but it was written
> *before* the overnight promotion, when that path still held the phase-2 model.
> NIGHT_LOG.md records 0.933 as the phase-2 score and 0.962 as v4 epoch 11, the
> model that was actually promoted.
>
> Verified rather than assumed: `weights/driftsense.pt`'s model tensors are
> **bit-identical** to `weights/driftsense_v4_e11.pt` (the container bytes differ,
> the weights do not). So the correct Mac baseline for the shipped weights is
> **`results/stream/driftsense_v4_e11.json`, acc@5px 0.962**.
>
> Compared against the stale 0.933 the CUDA port would have looked like a
> spurious **+2.7 point** win, and every later comparison drawn against it would
> have inherited the error.
>
> A results json records the *path* it evaluated, not the *contents*. Any file
> promoted in place invalidates every earlier json naming it. Prefer the
> `_e{N}.pt` jsons, which name a checkpoint that is never overwritten.

The like-for-like caution still stands: `stream_eval` scenes are freshly
generated and harder than the on-disk test splits, and these baselines run
without TTA. Compare same `-n`, same TTA setting, and never compare a streamed
number to a disk-split number.

**Result on this machine.** acc@5px **0.9600** vs the Mac's 0.9620, paired
bootstrap Δ **−0.0020, 95% CI [−0.0060, +0.0020]** — contains zero, gate passed.
The pairing is real: 537 of 1000 per-scene errors are bit-identical across macOS
and Windows and almost all the rest differ by <1e-4 px, so only 4 scenes select
a different peak. Confirmed independently on the stored 300-scene `data/val`
with the full 8-way TTA decode, which involves no generation at all and
reproduced the Mac's numbers exactly: median 0.65 px, acc@1 0.653, acc@2 0.843,
acc@5 0.960.

From here, use **`results/stream/driftsense_rtx.json` as the baseline** for every
comparison made on this machine.

## Step 4 — profile, ~20 minutes

`scripts/profile_rtx.py gpu | gen | loader`. **Measured on the RTX 4060 laptop,
2026-08-17** — full detail and the negative results are in RTX_LOG.md:

| | measured |
| --- | --- |
| GPU step, crop 512, batch 8, fp32 | **57.4 img/s**, 2.88 GB VRAM |
| GPU step, crop 768 / crop 1000, batch 8 | 25.1 / 14.7 img/s |
| disk loader, 6 workers (under generation load) | 65 img/s |
| **scene generation, 6 workers** | **8.8–9.5 pairs/s — the bottleneck** |

Three things this overturned:

- **Batch 8 is fastest per image** (57.4 vs 47.3 at batch 32). The net is
  bandwidth-bound, not launch-bound, so bigger batches do not help. Convenient:
  the shipped `--batch-size 8 --lr 5e-4` carries over with no LR rescaling.
- **`--amp` is 18% *slower* at every crop size.** Keep it off; it is a VRAM
  tool, not a speed tool. See RTX_LOG.md.
- **Generation saturates at 6 workers** and is memory-bandwidth bound, not
  core-bound — 24 threads, ~6 usable. The GPU consumes data 6.5x faster than
  this machine makes it, which is the inverse of the Mac and is what step 5
  is built around.

**Windows-specific trap.** Exceeding VRAM under WDDM does not raise OOM — the
driver spills into system RAM, the step slows ~20x, and `nvidia-smi` still
reports 100% utilisation. Measured at crop 768 / batch 32: free RAM fell to
0.9 GB and the run never finished. `train.py` now defaults to
`--vram-fraction 0.92` so this fails honestly instead.

## Step 5 — build the training pool

Two changes to this step, both from measurements above. Details in RTX_LOG.md.

**Store templates, not references — 7.2x less disk per pair.** Training uses the
reference for exactly one thing: an exact 10x `INTER_AREA` downsample to a
100×100 template. That downsample commutes exactly with the 8 dihedral
symmetries (verified: max difference 0 over 20 images × 8 symmetries), so
storing the result is byte-identical for training. Measured end-to-end:
**119 KB/pair** against 859 KB/pair before.

**Generate in shards and overlap with training.** Generation is 6.5x slower than
the GPU, so a serial "generate then train" phase would idle the GPU for most of
the night. Each shard is a self-contained split with a `COMPLETE` marker written
last; `train.py --refresh-pool` picks up finished shards between epochs.

```bash
./venv/bin/python scripts/build_pool.py --out data/pool \
    --shards 16 --canvases 3000 --workers 6
```

16 shards × 24 000 pairs = 384 000 pairs ≈ 45 GB, ~42 min per shard at
9.5 pairs/s. All shards draw from one continuous seed stream (31337, advanced by
`--start-index`), disjoint from every split seed (42, 555, 1234, 7777, 20001,
20002) and from `stream_eval`'s 999983, so no canvas is ever generated twice and
nothing reported is contaminated.

Sanity check (`--sample` avoids opening all 384 000 images; row-level geometry
checks stay exhaustive):

```bash
./venv/bin/python scripts/verify_dataset.py --sample 2000 data/pool/s000
```

**Keep `--crops-per-canvas 8`.** Measured: 16 buys only +17% pairs/s and 32 only
+32%, while halving and quartering the number of distinct canvases behind each
pair. Canvas diversity is what the streaming experiment showed matters.

**Reuse caveat, restated.** Pool reuse is now handled two ways: the pool grows
during the run, and `DriftSenseDataset.set_epoch()` re-rolls the augmentation
each pass — which it did **not** do before. The rng was keyed on `(seed, idx)`
alone, so every epoch replayed identical dihedral, photometric and crop choices
and a second pass over a pool was the same tensors again. Verify with
`scripts/check_epoch_variety.py` after touching either loader path.

## Step 6 — train.py additions *(done)*

Everything else is identical to the shipped run so results stay comparable.

- `cudnn.benchmark`, TF32 on matmul and cudnn — unconditional on CUDA.
- `pin_memory=True`, `prefetch_factor` (default 4), non-blocking transfers.
- `--amp` bf16 autocast **around the forward only**, with the loss forced back
  to fp32. The focal term evaluates `log(1 - sigmoid(logit))` over ~10^4
  near-zero negatives per sample; in bf16's 8-bit mantissa `1 - p` collapses to
  0 for confident negatives and the log goes to −inf. The convolutions keep the
  bf16. **Default off** — measured 18% slower at every crop size (RTX_LOG.md).
- `--vram-fraction` (default 0.92) — see the WDDM trap in step 4.
- `--samples-per-epoch` — draw a fresh random subset of the pool per epoch, so
  the epoch stays a fixed unit (and `total_steps` stays right) while the pool
  grows.
- `--refresh-pool` — rescan `--train-dirs` for newly `COMPLETE` shards between
  epochs. Requires `--samples-per-epoch`; `train.py` refuses otherwise.
- `persistent_workers=False` on **both** loader paths, and
  `DriftSenseDataset.set_epoch()` to re-roll augmentation per pass.

`torch.compile` not attempted; profiling says the step is bandwidth-bound, and
vectorising the cross-correlation — the obvious win — measured at ~1%
end-to-end, so the model is unchanged.

## Step 7 — experiments, in priority order

From RESUME.md's headroom list. An epoch of 16 000 samples is ~5 min at
57 img/s, against 44 min on the Mac, so the schedule is no longer the expensive
part — the data is.

1. **24 epochs, crop 512, from scratch** — held-out accuracy was still rising
   when the 12-epoch one-cycle ran out, with no overfitting signature. Batch 8
   and lr 5e-4 carry over from the shipped run unchanged (batch 8 profiled
   fastest per image, so there is no LR rescaling to argue about).
   ```bash
   ./venv/bin/python train.py --train-dirs data/pool --val-dir data/val \
       --crop 512 --batch-size 8 --epochs 24 --lr 5e-4 --workers 6 \
       --samples-per-epoch 16000 --refresh-pool --keep-epochs \
       --out weights/driftsense_v5a.pt
   ```
   Run this as soon as ~2 shards exist rather than waiting for the full pool:
   it costs ~2.5 h, de-risks the whole pipeline before a longer run, and the
   pool keeps growing underneath it.
2. **A longer schedule on the grown pool** — the actual attempt at the best
   model, once the pool is large enough that reuse stays low.
3. **Crop 768** — aimed at the wrong-repeat residue: the head argmaxes over a
   226x226 lattice at test having trained on 104x104. Costs 2.3x per image
   (25.1 vs 57.4 img/s), which is now affordable.
4. **Width 64 → 96** in `driftsense/model.py` — worth trying now that the
   finite-data constraint is gone.

Not worth more effort: TTA aggregation, BatchNorm recalibration (both written up
as negative results in NIGHT_LOG.md).

## Step 8 — judging results

Never on `data/val` alone — at acc@5px ~0.97 its 300 scenes carry a ±1.3-point
standard error, which once tied three epochs a 1000-scene run separated cleanly.

```bash
./venv/bin/python scripts/stream_eval.py --weights weights/driftsense_v5a_e17.pt \
    -n 1000 --workers 6 --out results/stream/v5a_e17.json
./venv/bin/python scripts/compare_checkpoints.py \
    --baseline results/stream/driftsense_rtx.json results/stream/v5a_*.json
```

Baseline is `driftsense_rtx.json` — the shipped weights measured *on this
machine*, not the Mac's json (see the correction in step 3).

Train with `--keep-epochs` so every epoch survives for this comparison. Touch
the test splits once, at the end.

**Sizing the final measurement.** The three test splits total 700 scenes, so at
acc@5px ≈ 0.976 their standard error is ±0.6 points. A claimed move from 0.976
to 0.980 is +0.4 points — about **3 scenes in 700** — which that set cannot
resolve. Any such claim needs a larger paired `stream_eval` (n ≥ 3000, both
sides re-run at the same `-n`) to mean anything. Generated eval scenes cost
nothing but CPU time, and canvas *c* owns samples `[c*crops, (c+1)*crops)`, so a
larger run's leading scenes coincide with a smaller one's. Do this after
generation stops, so it gets the full machine.

## Standing rules that still apply

- No git commands in this project.
- Train and evaluate on `gt_x_corr`/`gt_y_corr`, never `gt_x`/`gt_y`.
- Tune on `data/val`; the test splits are for the final number only.
- Report negative results as-is.

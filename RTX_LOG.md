# RTX 4060 port and long-run log — 2026-08-17

Continues NIGHT_LOG.md on new hardware. Same standard: negative results are
written up as-is, and nothing is claimed that was not measured.

**`weights/driftsense.pt` has not been touched.** Every run here writes
`weights/driftsense_v5*`.

---

## Machine

| | |
| --- | --- |
| GPU | RTX 4060 Laptop, 8 GB, sm_89, driver 581.57 / CUDA 13.0 |
| CPU | i7-13700HX, 16 cores / 24 threads |
| RAM | 15.7 GB |
| Disk | 250 GB free |
| OS | Windows 11, native (not WSL2 — see below) |
| torch | 2.11.0+cu128, Python 3.12.3 |

Two deviations from PORT.md's setup plan, both deliberate:

- **Native Windows, not WSL2.** PORT.md step 1 recommends WSL2 because Windows
  `spawn`s dataloader workers instead of forking. That reasoning is sound but is
  outweighed here: the data lives on `C:`, which WSL2 reaches over the 9p
  `/mnt/c` bridge, and the pool is hundreds of thousands of small PNGs — exactly
  the access pattern 9p is worst at. Copying the pool into the WSL ext4 VHDX
  instead would duplicate tens of GB. Spawn costs ~0.5 GB per worker once;
  9p would cost throughput on every read for the whole run.
- **torch 2.11.0, not the pinned 2.13.0.** `requirements.txt` pins
  `torch==2.13.0`; the cu128 index does not carry it. 2.11.0+cu128 is what
  installs. Recorded because the pin is now inaccurate for CUDA installs.

---

## Parity gate — passed, after correcting the baseline

The headline finding is not the gate result. It is that **the baseline PORT.md
tells you to use is the wrong file.**

`PORT.md` step 3 and `AGENT_HANDOFF.md` both name
`results/stream/driftsense.json` (acc@5px **0.933**) as the Mac's number for
`weights/driftsense.pt`. That json's `weights` field does record
`weights/driftsense.pt` — but it was written *before* the overnight promotion,
when that path still held the phase-2 model. NIGHT_LOG.md is unambiguous: 0.933
is phase 2, and the promoted model (v4 epoch 11) scored **0.962**.

Verified rather than argued: `weights/driftsense.pt`'s model tensors are
**bit-identical** to `weights/driftsense_v4_e11.pt`. The container bytes differ
(re-saved on promotion), the weights do not.

| n=1000 fresh scenes, TTA off | acc@5px | median | acc@1px | acc@2px |
| --- | ---: | ---: | ---: | ---: |
| Mac, `driftsense_v4_e11.json` (= shipped weights) | 0.9620 | 0.609 | 0.658 | 0.860 |
| **RTX 4060, same weights** | **0.9600** | 0.620 | 0.655 | 0.858 |
| ~~`driftsense.json`~~ (phase-2, stale) | ~~0.9330~~ | 0.642 | 0.639 | 0.837 |

Paired bootstrap vs the correct baseline: **Δ acc@5px −0.0020, 95% CI
[−0.0060, +0.0020]**. Contains zero. **Gate passed.**

Against the stale 0.933 the port would have looked like **+2.7 points** of free
accuracy, and every downstream comparison would have inherited that error.

**The pairing is real.** 537 of 1000 per-scene errors are bit-identical between
macOS and Windows; almost all the rest differ by <1e-4 px (float-level jitter in
the ZNCC refine), and only 4 scenes select a different peak.

**Confirmed without any generation involved.** On the stored 300-scene
`data/val` with the real 8-way TTA decode — same PNG files the Mac used, so this
isolates model and decode:

| | median | acc@1px | acc@2px | acc@5px |
| --- | ---: | ---: | ---: | ---: |
| Mac (NIGHT_LOG step 3) | 0.65 px | 0.653 | 0.843 | 0.960 |
| **RTX 4060** | **0.65 px** | **0.653** | **0.843** | **0.960** |

All four identical.

**Lesson worth keeping:** a results json records the *path* it evaluated, not
the contents. Promoting a checkpoint in place silently invalidates every earlier
json naming it. Prefer the `_e{N}.pt` jsons — those paths are never overwritten.

Baseline for everything on this machine: `results/stream/driftsense_rtx.json`.

---

## Profile — the machine is not what the port plan assumed

`scripts/profile_rtx.py`, added for this.

### GPU step (synthetic tensors, fp32 unless noted)

| crop | batch | img/s | ms/step | peak VRAM |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 8 | **57.4** | 139 | 2.88 G |
| 512 | 16 | 53.4 | 300 | 3.70 G |
| 512 | 32 | 47.3 | 676 | 5.35 G |
| 768 | 8 | 25.1 | 319 | 5.78 G |
| 1000 | 8 | 14.7 | 546 | 5.20 G |
| 1000 | 16 | OOM | | |

Batch 8 is the fastest *per image*; throughput falls as batch grows, which is
the signature of a bandwidth-bound network rather than a launch-bound one. That
is convenient — it means the shipped run's `--batch-size 8 --lr 5e-4` carries
over unchanged, with no LR rescaling to argue about.

The GPU is not power-limited: 80 W sustained (cap 112 W), 2490 MHz boost, 66 °C.

### Negative result: bf16 AMP is slower

| crop | fp32 | bf16 | |
| ---: | ---: | ---: | --- |
| 512 / bs 8 | 57.4 | 47.1 | −18% |
| 768 / bs 8 | 25.1 | 20.7 | −18% |
| 1000 / bs 8 | 14.7 | 12.1 | −18% |

PORT.md step 6 lists `--amp` as a throughput win. It is not, consistently, at
every size. The network is 0.46M parameters over large spatial maps, so it is
memory-bandwidth bound; tensor cores have nothing to bite on and the per-op cast
overhead dominates. AMP *is* worth ~3.5x on VRAM (2.88 G → 0.75 G at crop 512),
and crop 1000 at batch 16 only fits with it. Kept as an option, **default off**.

### Negative result: vectorising the cross-correlation is not worth it

`grouped_xcorr` issues one `conv2d` per batch element. Folding the batch into
the group dimension makes it a single launch, and the two agree exactly
(max abs diff 0.000e+00 on one run; 1.1e-3 on another, which is cudnn algorithm
selection noise at 3e-6 relative, not a difference between the implementations).

It is 14% faster on that op — 8.7 ms vs 10.1 ms — but the op is ~7% of the step,
so the end-to-end gain is ~1%. **Not applied.** `driftsense/model.py` is
unchanged.

### Scene generation is the bottleneck, and it is bandwidth-bound

Timed on the real `scripts/gen_data.py`, 8 crops per canvas:

| workers | 1 | 2 | 4 | 6 | 10 | 14 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| pairs/s | 3.6 | 5.6 | 7.0 | **8.8** | 8.3 | 8.6 |

24 threads, and roughly 6 are usable: 1→6 workers buys 2.4x, and 6→14 buys
nothing. Per-worker efficiency falls from 2.0 s/canvas to 5.5 s/canvas. The
imaging pipeline works on 10000² float64 intermediates (~800 MB per canvas), so
the workers saturate memory bandwidth long before they saturate cores.

RAM is *not* the limit for the real generator (5.8 GB in use at 14 workers) —
though it was for an earlier measurement that mistakenly loaded torch into every
worker. Freeing ~2.8 GB of desktop apps did raise 6-worker throughput from
6.6 to 9.6 pairs/s in that flawed setup, so headroom still helps.

**The GPU consumes data 6.5x faster than this machine can create it.** That is
the inverse of the Mac, where generation was only slightly ahead of the GPU, and
it is what the data strategy below is built around.

### Negative result: more crops per canvas is a bad trade

| crops/canvas | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: |
| pairs/s (6 workers) | 7.4 | 9.6 | 11.2 | 12.7 |
| distinct canvases/s | 1.85 | 1.20 | 0.70 | 0.40 |

Fitting `C + k·c`: a canvas costs ~1.6 core-s and each additional reference crop
~0.43 core-s, so the crops already dominate at k=8. Going 8→16 buys +17% pairs
while halving canvas diversity per pair, and 8→32 buys +32% while quartering it.
Canvas diversity is what the streaming experiment showed actually matters.
**Kept at 8**, the validated setting.

### Disk loader

65 img/s at 6 workers on `data/val` *while generation was running* — above the
GPU's 57.4 img/s, so the loader is not the constraint. `data/val` holds
full-resolution references; the compact pool below decodes 100×100 templates
instead, so this is a conservative figure.

---

## Trap: Windows does not OOM, it silently degrades

Profiling crop 768 at batch 32 exceeded 8 GB of VRAM. On Linux that raises OOM.
Under the Windows WDDM driver it does not — the driver spills allocations into
system RAM and the step slows by roughly 20x **while still reporting 100% GPU
utilisation**. Free system RAM fell to 0.9 GB and the run never finished; it had
to be killed.

For an unattended overnight run that failure mode is worse than a crash, because
it looks like progress. `train.py` now defaults to
`--vram-fraction 0.92`, which caps the process and turns the cliff into an
honest OOM. `scripts/profile_rtx.py` does the same.

---

## Data strategy

The constraint chain is: generation 8.8 pairs/s ≪ GPU 57 img/s ≪ disk 250 GB.
Generation is the critical path, so the plan maximises unique scenes per hour
and never lets the GPU wait on a serial generation phase.

### Compact pool format — 7.2x more data per GB

Training uses the reference for exactly one thing: an exact 10x `INTER_AREA`
downsample to a 100×100 template (`driftsense/dataset.py:build_sample`, and
`matching.make_template` at inference — even the ZNCC refinement correlates
against the template, never the full-resolution reference).

That downsample **commutes exactly with the 8 dihedral symmetries** used for
augmentation: a 1000 px frame partitions into 10×10 blocks, and that partition
is invariant under the square symmetries. Verified over 20 images × 8
symmetries — **max difference 0**.

So a training pool can store the template and skip the reference. Training
samples are byte-identical; disk collapses:

| | per pair | measured |
| --- | ---: | --- |
| reference 1000×1000 (old) | 859 KB | 748 KB ref + 890 KB search / 8 crops |
| **template 100×100 (new)** | **119 KB** | measured end-to-end on a real shard |

`scripts/gen_data.py --store-templates`. **Training pools only** —
`evaluate.py` and `infer.py` are specified on full-resolution references, and
`scripts/verify_dataset.py` now checks the stored size against a new
`reference_px` manifest column so the two cannot be mixed up silently.

### Bug found: every pass over a disk pool was a literal replay

`DriftSenseDataset.__getitem__` seeded its augmentation rng from `(seed, idx)`
alone. The dihedral transform, the photometric jitter and the random search
window were therefore **identical on every epoch**. A second pass over a pool
was not a new view of it, it was the same tensors again.

That was invisible while pools were read roughly once, and it is exactly the
wrong property for the plan here, which makes several passes. `set_epoch()` now
re-rolls it, keyed so epoch 0 reproduces the old behaviour exactly.

This is the same shape as the `persistent_workers` bug NIGHT_LOG.md records, and
reachable by the same trap — a persistent worker keeps its own frozen copy of
the dataset and never sees `set_epoch()`. So `train.py` uses
`persistent_workers=False` on both paths, and
`scripts/check_epoch_variety.py` verifies it by hashing the tensors the loader
actually emits rather than by trusting the comment:

```
DriftSenseDataset -- persistent_workers=False (what train.py uses)
  epoch 0 vs 1: 0/3 batches identical  -> fresh
DriftSenseDataset -- persistent_workers=True (the trap)
  epochs identical: True  -> as expected; train.py avoids this
```

### Overlapping generation with training

Generation runs as independent shards (`scripts/build_pool.py`), each a
self-contained split directory with a `COMPLETE` marker written last.
`train.py --refresh-pool` rescans between epochs and picks up finished shards,
so the pool grows underneath a run in progress and a half-written shard is never
read. `--samples-per-epoch` keeps the step count per epoch fixed as the pool
grows, so the one-cycle schedule's `total_steps` stays correct.

All shards draw from one continuous seed stream (31337, advanced by
`--start-index`), disjoint from every split seed (42, 555, 1234, 7777, 20001,
20002) and from `stream_eval`'s 999983 namespace. No canvas is generated twice.

Serialising instead — generate everything, then train — would have idled the GPU
for most of the night, since generation of a 384k-pair pool takes ~12 h and
training consumes it in ~4 h.

---

## Run v5f — fine-tune from the shipped weights

### Why fine-tune rather than train from scratch

PORT.md, RESUME.md and AGENT_HANDOFF.md all name "24 epochs from scratch" as the
next experiment. A from-scratch run was started and then **abandoned in favour of
fine-tuning**, on the user's call, and the call was right:

- **Risk floor.** From scratch on a pool that starts at 24k pairs may not reach
  0.976 at all — the shipped model accumulated ~200k+ samples over three phases.
  Fine-tuning starts at a known-good point and, with best-checkpoint selection,
  is very unlikely to end worse. Under a deadline that asymmetry decides it.
- **It is this project's own precedent.** Phase 2 resumed from phase 1 at
  lr 4e-4; phase 3 resumed from v3_last at 5e-4 and gained +2.9 points. Resuming
  is what has actually worked here. "From scratch" was the handoff's
  recommendation, not a result.

The difference was visible immediately: from scratch, step 250 showed loss 4.12
and batch median error 147 px; fine-tuning showed **0.317 and 1.5 px**.

`--finetune` was added for this: `--resume` alone restores the optimizer and
fast-forwards the scheduler to the checkpoint's `global_step`, which is correct
for continuing an interrupted run and wrong here — the shipped model's one-cycle
had already annealed to zero, and its epoch counter would have eaten most of
`--epochs`. `--finetune` loads weights only, with a fresh optimizer and a fresh
one-cycle at a lower `max_lr`.

```bash
python train.py --train-dirs data/pool --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 24 --lr 3e-4 --workers 6 \
    --samples-per-epoch 16000 --refresh-pool --keep-epochs --val-limit 100 \
    --resume weights/driftsense.pt --finetune --out weights/driftsense_v5f.pt
```

Training loss fell **0.3215 → 0.2528** — below the phase-3 run's 0.308 endpoint —
with held-out accuracy rising throughout and no overfitting signature. The
one-cycle warmup bump appeared on schedule (epochs 2–4 rising to 0.341) and
annealed away.

### Result — 1 000 freshly generated scenes, single-view decode

Baseline is `driftsense_rtx.json`: the **shipped weights measured on this
machine**, not the Mac's json.

| checkpoint | acc@1px | acc@2px | **acc@5px** | median | Δ vs shipped | 95% CI | P(worse) |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **v5f epoch 23** | **0.665** | **0.873** | **0.975** | 0.60 px | **+1.5 pts** | [+0.001, +0.029] | **0.018** |
| v5f epoch 20 | 0.666 | 0.872 | 0.974 | 0.60 px | +1.4 pts | [+0.000, +0.028] | 0.029 |
| v5f SWA (e19–23) | 0.666 | 0.872 | 0.974 | 0.60 px | +1.4 pts | [+0.000, +0.028] | 0.027 |
| v5f epoch 16 | 0.664 | 0.869 | 0.972 | 0.60 px | +1.2 pts | [−0.002, +0.026] | 0.051 |
| v5f epoch 15 | 0.665 | 0.869 | 0.971 | 0.60 px | +1.1 pts | [−0.003, +0.026] | 0.074 |
| v5f epoch 18 | 0.663 | 0.868 | 0.969 | 0.60 px | +0.9 pts | [−0.006, +0.024] | 0.130 |
| shipped | 0.655 | 0.858 | 0.960 | 0.62 px | — | — | — |

Epoch 23's interval excludes zero, and it is also the principled pick — the
fully annealed end of the one-cycle. Every metric improves together (acc@1,
acc@2, acc@5 and median), which matters because sub-pixel precision and
wrong-repeat rejection are different failure modes; an intervention that moved
only one would be suspicious.

### Negative result: checkpoint averaging (SWA) did not help

Averaging epochs 19–23 scored **0.974**, tying epoch 20 and losing to epoch 23's
0.975. The reasoning was sound — a one-cycle tail is several points in one flat
basin, so their centroid usually generalises slightly better — but the tail here
was already flat enough that there was nothing to smooth.

Recorded rather than dropped. `scripts/average_checkpoints.py` is kept: it costs
nothing to try on a future run, and it is *not* the 2-checkpoint ensemble
NIGHT_LOG.md rejected (it produces one model at unchanged inference cost).

### Not attempted, and why: reinforcement learning

Raised as an option. It does not fit this problem. The generator supplies the
exact ground-truth coordinate for every sample and the loss is differentiable
end to end, so supervised learning strictly dominates — a policy gradient would
be a higher-variance estimate of a gradient already available in closed form.

The natural place to apply it would be the *selection* policy over candidate
peaks, and that is the half this project has now measured as a dead end three
times: TTA aggregation swept flat, pure ZNCC arbitration rejected, a
2-checkpoint ensemble rejected. **Better proposals beat better arbitration.**

### The test splits, measured once

| split | acc@5px before | **after** | acc@2px before | **after** | failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test` | 0.9700 | **0.9800** | 0.8767 | **0.8833** | 9 → **6** |
| `test_medium` | 1.0000 | **1.0000** | 1.0000 | **1.0000** | 0 → **0** |
| `test_severe` | 0.9600 | **0.9600** | 0.6850 | 0.6800 | 8 → **8** |
| **all 700** | **0.9757** | **0.9800** | | | 17 → **14** |

**The test set alone does not establish this.** Paired bootstrap over the 700
scenes: **+0.43 points, 95% CI [−0.29, +1.14]** — spans zero. Three scenes in
700 is below what that set can resolve, which is exactly why the 1 000-scene
fresh evaluation exists and why it was run first. The larger set supplies the
significance; the test splits supply the headline number and confirm direction.

Two things not to overstate:

- **`test_severe` did not move.** 8 failures before and after, and its acc@2
  slipped 0.685 → 0.680 (one scene). The whole gain came from `test`. The severe
  residue is the low-dose shot-noise floor described in AGENT_HANDOFF.md, and
  nothing here addressed it.
- **The data lever was cut short.** The pool was planned at 384 000 pairs and
  stopped at **48 000** so the fine-tune would run faster under deadline. The run
  made ~8 passes over it rather than 1–2. The gain above was therefore bought
  mostly by the longer schedule and the fine-tune, *not* by the extra data — the
  "more data" hypothesis is still largely untested on this hardware.

### Promotion — deliberately not done

`weights/driftsense.pt` is **untouched** (SHA256 `615C9CA2F47A…`), per the
standing rule in AGENT_HANDOFF.md. The candidate is
`weights/driftsense_v5f_e23.pt`. Promoting it is a one-line copy and a human
decision:

```bash
cp weights/driftsense_v5f_e23.pt weights/driftsense.pt   # then re-run infer.py end to end
```

If promoted, README.md's results tables need updating to 0.9800 / 14 failures,
and `results/results.json` regenerating.

### What to do next

1. **Let the pool actually grow.** This is the untested lever. `build_pool.py`
   on an idle machine reaches ~9.5 pairs/s; a few hours gets 100–200k pairs and
   drops reuse from 8 passes to 1–2. That is the experiment the user originally
   asked for and it has not really been run.
2. **Crop 768.** Still untested, still aimed at the wrong-repeat residue, and now
   affordable at 25 img/s.
3. **Hard-negative mining** (DaSiamRPN-style distractor-aware loss) — the one
   genuinely new idea NIGHT_LOG.md flagged, and the only listed lever aimed at
   `test_severe`, which is where the remaining failures now live.

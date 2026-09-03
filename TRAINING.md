# How the Drift-Sense model was trained

A complete, reproducible account: what the data is, where it comes from, what
the network is, every training phase that was run, how checkpoints were chosen,
and what the numbers actually mean.

Written to be read by someone who has never seen this repository.

---

## 1. The problem

Given two SEM images of a repeating semiconductor layout:

| | resolution | field of view |
| --- | --- | --- |
| **Reference** | 1000 x 1000 px @ **1 nm/px** | 1 um |
| **Search** | 1000 x 1000 px @ **10 nm/px** | 10 um |

Find where the Reference pattern sits inside the Search frame, as a single
`(x, y)` centre in Search-image pixels.

The 10x pixel-size ratio means the Reference occupies exactly a **100 x 100 px
box** inside the Search image. That geometry is fixed by the problem statement
and is hard-coded rather than learned.

**Why it is hard.** The layout is periodic. Dozens of positions in the Search
frame are locally identical, so patch appearance alone cannot identify the site.
The metric is accuracy at a 5 px tolerance, so locking onto the wrong repeat is
a total failure, not a small error.

---

## 2. The data

**There is no downloadable dataset.** The hackathon provides a *generator*, not
data - its README states this directly. Every image used here was produced
locally from a random seed, so the training set is limited only by CPU time and
disk, not by supply.

### Generating data

```bash
python generate_dataset.py --architecture mixed --num-pairs 100 --output-dir ./out
```

| flag | meaning |
| --- | --- |
| `--architecture` | `dram`, `finfet` or `mixed` (6 node presets each) |
| `--num-pairs` | number of Reference/Search pairs |
| `--output-dir` | destination for `reference/`, `search/`, `manifest.csv` |
| `--noise` | `randomized` (default) or a fixed `low`/`medium`/`high`/`severe` |
| `--seed` | reproducibility seed |

Each sample builds a 10000 x 10000 px "fine canvas" of a repeating layout
composed into array mats separated by flatter routing strips. A 1000 x 1000
window becomes the Reference; the whole canvas is blurred, downsampled 10x and
imaged as the Search frame. Both pass through an SEM acquisition model: beam
PSF, shot noise, detector noise, raster drift, and optionally astigmatism,
vignetting, gamma, barrel distortion, charging streaks, speckle and impulse
noise.

### Ground truth: use `gt_x_corr` / `gt_y_corr`, never `gt_x` / `gt_y`

This is the single most important detail in the repository.

The upstream generator computes ground truth from the crop origin on the
*pre-imaging* canvas (`gt_x = x0 / 10 + 50`), then warps the Search frame with
raster drift and barrel distortion **without updating the label**. Those warps
move the pattern relative to the frame the label was written in.

On the shipped splits the two conventions differ by a **mean of 3.1 px and up to
22 px** - well past the 5 px tolerance. Training on the uncorrected label
teaches a systematic error larger than the thing being measured.

Because the warps are generated here, they are invertible exactly.
`driftsense/generate.py:correct_gt()` maps the label through them, and both
conventions are written to every manifest:

| column | meaning |
| --- | --- |
| `gt_x`, `gt_y` | upstream convention - kept only for comparability |
| `gt_x_corr`, `gt_y_corr` | where the pattern actually is - **train and evaluate on these** |
| `label_shift_px` | distance between the two |

Verified empirically by `scripts/verify_gt_correction.py`: under clean
photometrics with only the geometric warp varying, the residual between a ZNCC
match and the label falls from **2.5-4.8 px to 0.4-0.9 px**.

**Residual noise floor.** The per-row drift jitter is i.i.d. and cannot be
predicted from the images, so it sets a hard limit on achievable accuracy.
`scripts/label_noise_floor.py` quantifies it.

### Data splits

| split | scenes | purpose |
| --- | ---: | --- |
| `data/val` | 300 | tuning - the only split touched during development |
| `data/test` | 300 | held out |
| `data/test_medium` | 200 | held out, fixed medium noise |
| `data/test_severe` | 200 | held out, fixed severe noise |
| **held-out total** | **700** | measured once, at the end |

Split seeds (42, 555, 1234, 7777, 20001, 20002) are disjoint from the training
pool seed (31337) and from the streaming-evaluation namespace (999983), so no
scene reported on has ever been trained on.

**The 700 is a choice, not a supply limit.** More evaluation scenes can be
generated at any time; the set is frozen so model-to-model comparisons stay
like-for-like.

---

## 3. The model

A Siamese fully-convolutional localiser, **0.46 M parameters**
(`driftsense/model.py`).

The layout is periodic, so *local* appearance cannot identify a site. Three
things break the tie, and the architecture is built around each:

1. **Large-scale zone structure** (array mats separated by routing strips) -
   visible only with a wide receptive field, so a **context branch** of stacked
   dilated convolutions runs over the whole Search frame and conditions the
   response map.
2. **The arrangement of the decoy peaks themselves** - the **head** is
   deliberately deep and dilated *over the response map*, so it judges a peak
   against the surrounding lattice of candidates rather than scoring each in
   isolation.
3. **Per-line CD/placement fingerprints**, which survive the 10x downsample as
   low-amplitude intensity modulation - this is what the **correlation branch**
   picks up.

Layout is a SiamRPN++-style cross-correlation: a shared encoder embeds both the
template and the Search frame, a grouped cross-correlation produces a
multi-channel match-evidence volume, and the head turns that plus context into a
centre heatmap and a sub-cell offset field.

**Decoding is two-stage, deliberately split:**

* the network decides *which* of the many identical-looking candidates is right
  - the hard, learned part;
* a classical **ZNCC snap** at full resolution decides *exactly where* - the
  easy, precise part, which a 4 px-stride heatmap cannot do alone.

Handing the sub-pixel job to correlation is what keeps the final error well
under one pixel.

**Losses** (`driftsense/engine.py`): a CenterNet-style penalty-reduced focal
loss on the heatmap (exactly one positive cell against ~10^4 negatives, so plain
BCE is swamped), plus smooth-L1 on the sub-cell offset, supervised only at the
true cell.

**Augmentation** (`driftsense/dataset.py`): the 8 dihedral symmetries applied
jointly to Reference and Search, independent photometric jitter per frame
(brightness, contrast, gamma, additive noise, multiplicative speckle, impulse
noise), and random 512 px Search-window crops.

---

## 4. Training phases

Training runs on a 512 px window of the Search frame. The network is fully
convolutional, so inference runs on the full 1000 x 1000 frame unchanged.

### Phase 1 - base training

```bash
python train.py --train-dirs data/train_mc data/train --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 9 --lr 1e-3 --out weights/driftsense_p1.pt
```

14 000 pairs (12 000 multi-crop + 2 000 single-crop). Best checkpoint was
**epoch 6**; epochs 7-9 kept reducing training loss without improving held-out
accuracy.

### Phase 2 - speckle fine-tune

Seeded from the phase-1 best, after failure analysis identified multiplicative
noise as the dominant driver of wrong-repeat lock-ons (standardised effect
+0.59): because speckle scales with signal it is not removed by the per-image
standardisation that absorbs additive noise, and the generator only ever emitted
it at four discrete levels.

```bash
python train.py --train-dirs data/train_mc data/train --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 6 --lr 4e-4 \
    --resume weights/driftsense_p1.pt --out weights/driftsense.pt
```

Best checkpoint was **epoch 2** - again, later epochs overfit.

**Both phases showed the same signature: training loss kept improving long after
held-out accuracy stopped.** Both drew from a fixed pool of 3 500 scenes, so
that is the pool being memorised, not the schedule running out.

### Phase 3 - streamed training on unlimited fresh data

```bash
python train.py --stream --stream-length 16000 --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 12 --lr 5e-4 --workers 4 --keep-epochs \
    --out weights/driftsense_v4.pt
```

`--stream` (`driftsense/stream_dataset.py`) generates a brand-new scene for
every sample inside the dataloader workers rather than reading from disk: no
scene is ever reused. Worth **+1.2 points** over all 700 held-out scenes. The
overfitting signature disappeared with it - training loss fell 0.410 -> 0.308
while held-out accuracy *rose* throughout.

> **A bug worth recording.** The first version of this silently did nothing. The
> dataset shifts its seed stream via `set_epoch()`, but the DataLoader was built
> with `persistent_workers=True` - and persistent workers are forked once and
> keep their own copy of the dataset, so `set_epoch()` never reached them. Every
> epoch regenerated **identical scenes**: "unlimited fresh data" was a fixed
> 16 000-scene pool on repeat, reproducing the exact failure it was written to
> fix. It surfaced only on a resumed run, which forks its workers *after*
> `set_epoch` and so reported a much higher loss (0.43 vs 0.27) at the same
> epoch with the same weights.

**Epoch 11 was promoted**, giving acc@5px **0.9757** over the 700 held-out
scenes.

### Phase 4 - fine-tune on a large on-disk pool

Run on an RTX 4060 laptop. Full detail in `RTX_LOG.md`.

```bash
# 1. build a training pool (shards, each with a COMPLETE marker written last)
python scripts/build_pool.py --out data/pool --shards 16 --canvases 3000 --workers 6

# 2. fine-tune from the phase-3 weights
python train.py --train-dirs data/pool --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 24 --lr 3e-4 --workers 6 \
    --samples-per-epoch 16000 --refresh-pool --keep-epochs --val-limit 100 \
    --resume weights/driftsense_prev_0.9757.pt --finetune \
    --out weights/driftsense_v5f.pt
```

**Why fine-tune rather than train from scratch.** Starting fresh risks landing
*below* the existing model - it accumulated ~200 k+ samples across three phases -
and fine-tuning from a known-good point with best-checkpoint selection is very
unlikely to end worse. It is also this project's own precedent: phases 2 and 3
both resumed. The difference was immediate - at step 250, from scratch showed
loss 4.12 and batch median error 147 px; fine-tuning showed **0.317 and 1.5 px**.

`--finetune` loads **weights only**. Plain `--resume` restores the optimizer and
fast-forwards the scheduler to the checkpoint's `global_step`, which is correct
for continuing an *interrupted* run and wrong here: the previous one-cycle had
already annealed to zero, and its epoch counter would have consumed most of
`--epochs`.

Training loss fell **0.3215 -> 0.2528** with held-out accuracy rising throughout
and no overfitting signature. **Epoch 23** - the fully annealed end of the
one-cycle - was selected.

#### Two supporting changes made in this phase

**Compact pool format - 7.2x less disk per pair.** Training uses the Reference
for exactly one thing: an exact 10x `INTER_AREA` downsample to a 100 x 100
template. That downsample **commutes exactly** with the 8 dihedral symmetries
(verified over 20 images x 8 symmetries, max difference **0**), so storing the
template instead of the full Reference is byte-identical for training. Measured:
**859 KB/pair -> 119 KB/pair**. Enabled by `scripts/gen_data.py
--store-templates`; training pools only, since `infer.py` and `evaluate.py` are
specified on full-resolution references.

**Per-epoch augmentation was frozen, and is now fixed.** `DriftSenseDataset`
seeded its augmentation rng from `(seed, idx)` alone, so the dihedral transform,
photometric jitter and window crop were **identical on every epoch** - a second
pass over a pool was the same tensors again, not a new view. `set_epoch()` now
re-rolls it. Because this is reachable by the same `persistent_workers` trap
described above, both loader paths use `persistent_workers=False`, and
`scripts/check_epoch_variety.py` verifies it by hashing the tensors the loader
actually emits.

---

## 5. How checkpoints are chosen

**Not on `data/val` alone, and never on the in-loop metric.**

At acc@5px ~ 0.97 the 300-scene validation split has a **+/-1.3 point standard
error**, larger than any difference worth chasing. The in-loop metric is worse
still - 100 scenes, single-view - and it once scored four epochs identically and
picked the **worst** of three candidates that a larger run separated cleanly.

The procedure actually used:

1. Train with `--keep-epochs`, so every epoch survives.
2. Shortlist candidates from the in-loop history (free, already computed).
3. Separate them with `scripts/stream_eval.py` on **1 000 freshly generated
   scenes**, generated in dataloader workers and consumed immediately - no disk.
   Seeded in namespace 999983, disjoint from training and from every split. The
   scene set is fixed by `--num` and `--crops-per-canvas`, so the comparison is
   **paired** across checkpoints.
4. Compare with `scripts/compare_checkpoints.py`, a **paired bootstrap**:
   resample scenes, recompute both accuracies on the same resample, take the
   difference. Pairing cancels scene difficulty, which dominates the variance.
5. Promote only on a consistent win - ahead on acc@5px *and* not worse on
   median, acc@1px or acc@2px.
6. Only then, once, measure the test splits.

`scripts/judge_run.py` automates steps 1-6.

---

## 6. Results

### Held-out test splits (700 scenes, 8-way TTA decode)

| split | scenes | phase 3 | **phase 4** | failures |
| --- | ---: | ---: | ---: | ---: |
| `test` | 300 | 0.9700 | **0.9800** | 9 -> **6** |
| `test_medium` | 200 | 1.0000 | **1.0000** | 0 -> **0** |
| `test_severe` | 200 | 0.9600 | **0.9600** | 8 -> **8** |
| **all** | **700** | **0.9757** | **0.9800** | **17 -> 14** |

Classical ZNCC baseline over the same data: **0.573**.

### Significance

The 700-scene test set **cannot establish this on its own**: the paired
bootstrap gives +0.43 points with a 95% CI of **[-0.29, +1.14]**, spanning zero.
Three scenes out of 700 is below its resolution.

The significance comes from the larger fresh-scene evaluation:

| checkpoint | acc@1px | acc@2px | **acc@5px** | median | delta | 95% CI | P(worse) |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **phase 4, epoch 23** | **0.665** | **0.873** | **0.975** | 0.60 px | **+1.5 pts** | [+0.001, +0.029] | **0.018** |
| phase 3 (previous) | 0.655 | 0.858 | 0.960 | 0.62 px | - | - | - |

n = 1 000 freshly generated scenes, single-view decode, paired. The interval
excludes zero, and every metric improves together - which matters, because
sub-pixel precision and wrong-repeat rejection are different failure modes and
an intervention that moved only one would be suspicious.


### Independent check on two unseen noise regimes

The evaluations above use the `randomized` noise envelope. As a separate
confirmation, both models were run on **fixed-noise regimes with a different
scene set entirely** (`--crops-per-canvas 1`, so the canvases and crops differ
from every randomized run). Neither regime was used for any selection decision.

| regime | n | phase 3 | **phase 4** | delta | 95% CI | P(worse) |
| --- | ---: | ---: | ---: | ---: | :---: | ---: |
| `high` | 600 | 0.9967 | **1.0000** | +0.33 pts | [+0.00, +0.83] | 0.136 |
| `severe` | 600 | 0.9650 | **0.9717** | +0.67 pts | [-0.33, +1.67] | 0.129 |
| **pooled** | **1 200** | **0.9808** | **0.9858** | **+0.50 pts** | **[+0.00, +1.08]** | **0.049** |

Phase 4 is ahead on both, solves the `high` regime outright (600/600), and cuts
pooled failures 23 -> 17. Neither regime alone is conclusive; pooled, the
interval's lower bound sits at zero with P(worse) = 0.049. Read this as
consistent supporting evidence for the 1 000-scene result, not as a second
independent proof.

### Honest limitations

- **`test_severe` did not improve.** 8 failures before and after; its acc@2px
  slipped 0.685 -> 0.680 (one scene). The entire gain came from `test`.
- **The "more data" lever is still largely untested.** The pool was planned at
  384 000 pairs and stopped at **48 000** for time, so the run made ~8 passes
  over it rather than 1-2. The gain came mostly from the longer schedule and the
  fine-tune, not from more unique data.
- The residual failures are references landing deep inside uniform mats at low
  dose, where the per-line CD fingerprint that distinguishes one repeat from the
  next is destroyed by shot noise rather than merely obscured.

---

## 7. Negative results

Recorded because "we tried the obvious thing and it did not work" is the useful
form of these.

| tried | outcome |
| --- | --- |
| **BatchNorm recalibration at test resolution** (FixRes) | **Cost a point** (0.947 -> 0.937). The stored running stats are the average of *augmented* batch statistics, so using them at inference reproduces the training-time normaliser - the consistent thing to do. Re-estimating on clean frames substitutes a normaliser the network was never trained under. |
| **Checkpoint averaging (SWA)** over the last 5 epochs | **0.974 vs epoch 23's 0.975** - no help. The one-cycle tail was already flat enough that there was nothing to smooth. |
| **TTA aggregation sweeps** | Flat across alpha 0.35-0.5 and any top_k >= 3; worth exactly +1 sample in 300. |
| **Pure ZNCC arbitration** | Searching the whole frame it picks the wrong repeat ~50% of the time; even among shortlisted candidates it scored 0.920 vs 0.943, because it prefers crisper-looking decoys. |
| **2-checkpoint ensemble** | No gain. |
| **Denoising TTA views** | No gain. |
| **bf16 AMP** | **18% *slower*** at every crop size on an RTX 4060. The network is small and bandwidth-bound, so tensor cores have nothing to bite on and cast overhead dominates. Useful for VRAM, not speed. |
| **Vectorising the grouped cross-correlation** | Numerically identical and 14% faster on that op, but the op is ~7% of the step - about 1% end-to-end. Not applied. |
| **More crops per canvas** (8 -> 16 -> 32) | +17% / +32% pairs per second while halving and quartering canvas diversity per pair. Bad trade; kept at 8. |
| **Reinforcement learning** | Does not fit. The generator supplies the exact ground-truth coordinate and the loss is differentiable end to end, so supervised learning strictly dominates - a policy gradient would be a higher-variance estimate of a gradient available in closed form. The natural place to apply it, the peak-*selection* policy, is the half measured as a dead end three times over. |

The recurring conclusion: **better proposals beat better arbitration.**

---

## 8. Reproducing this

```bash
python -m venv venv
venv/Scripts/pip install -r requirements.txt          # Linux/macOS: venv/bin/pip
# for an NVIDIA GPU, install CUDA torch afterwards:
venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# generate data
python scripts/gen_data.py --split val --num-samples 300 --seed 1234 --workers 6
python scripts/build_pool.py --out data/pool --shards 16 --canvases 3000 --workers 6

# train
python train.py --train-dirs data/pool --val-dir data/val --crop 512 \
    --batch-size 8 --epochs 24 --lr 3e-4 --workers 6 --samples-per-epoch 16000 \
    --refresh-pool --keep-epochs --out weights/driftsense_v5.pt

# judge, then measure the test splits once
python scripts/judge_run.py --run weights/driftsense_v5.pt --epochs 24
```

Every sample is reproducible from its seed: sample *i* is drawn from its own
`SeedSequence` child, so a given `(seed, i)` yields the same sample regardless of
how many workers run or how the run was chunked.


---

## Phase 2 retraining lineage (current shipped weights)

The narrative above stops at the Phase 4/v5 story. The shipped Phase 2 model
(`weights/driftsense.pt`) descends from the Phase 2 fine-tune chain; the
authoritative session log is `.agents/PHASE2_STATE.md`. Summary of the lineage
and how the shipped checkpoint was selected:

- **Phase 2 base** — fine-tuned from the Phase 1 best on the Phase 2
  distribution (unknown pose, absent pairs, severity ladder).
- **p6 / p6_last** — continuation fine-tunes; `p6_last` measured +2.72 on the
  external hold-out (2.2σ on the 200-pair grade scale) and was shipped as
  `cand_driftsense_p6_last`.
- **p8 vs p9** — two arms differing (p8 vs p9) only in `--jitter-power`
  label-noise weighting; p8's inverted weighting (`--jitter-power -1`) measured
  +0.05 (0.04σ on the grade scale — inside noise) and was recorded as a
  controlled negative.
- **p9_last (shipped)** — the epoch-39 checkpoint; measured 72.55 → 75.27 on
  the 85-point scale across the held-out re-measurement, and is the state in
  `weights/driftsense.pt` (epoch metadata in the checkpoint file). Promoted on
  the paired full-set comparison even though one CI crossed zero, because the
  component breakdown was uniformly non-inferior (decision documented in
  `.agents/PHASE2_STATE.md`).
- **Negative experiments kept on record:** spectral pose estimation (measured,
  negative, `7ee5473`), Set D bonus unreachable at the then-current score
  (`3185778`), second-checkpoint ensemble (hurt, dropped), epoch-12/24/30
  trajectory soups (−1.7/−2.0 on a paired draw, `.agents/INFERENCE_TWEAKS.md`).

Which checkpoint ships: whatever `weights/driftsense.pt` contains — verify with
`python -c "import torch; ck=torch.load('weights/driftsense.pt', weights_only=True); print(ck.get('epoch'), ck.get('arch'))"`
and the SHA-256 recorded in the submission notes.

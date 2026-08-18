# Drift-Sense — Navigation-Error Recovery

Locate a high-resolution **Reference** patch inside a low-resolution **Search**
frame of a repeating semiconductor layout, and return the match centre in
Search-image pixels.

|            | Reference        | Search            |
| ---------- | ---------------- | ----------------- |
| size       | 1000 × 1000 px   | 1000 × 1000 px    |
| pixel size | 1 nm/px          | 10 nm/px          |
| field      | 1 µm             | 10 µm             |

The 10× pixel-size ratio means the Reference occupies exactly a **100 × 100 px**
box in the Search frame. The difficulty is that the layout is periodic —
hundreds of near-identical candidates sit in the same frame — so local
appearance cannot identify a site. This is Applied Materials'
*Navigation-Error Recovery* problem.

**Approach:** a Siamese correlation network picks *which* candidate is correct
(the hard, learned part), eight-way dihedral test-time augmentation votes away
the wrong-repeat lock-ons, and a classical ZNCC snap at full resolution places
the answer sub-pixel (the precise part). Details in [Method](#method).

**Result:** **1.000** accuracy at the 5 px tolerance on the standard `medium`
operating point, **0.980** on the hardest randomized split, and **0.980**
across all 700 held-out scenes — versus 0.705, 0.477 and 0.573 for the
classical ZNCC baseline.

---

## Quick start

Requires **Python 3.10–3.13** (PyTorch does not yet support 3.14).

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

Generate a sample Reference/Search pair with ground truth:

```bash
./venv/bin/python generate_dataset.py --architecture mixed --num-pairs 4 --output-dir ./sample
```

Run the localiser on the first pair:

```bash
./venv/bin/python infer.py --reference sample/reference/00000.png --search sample/search/00000.png
```

This prints a single line — the predicted centre in Search-image pixels:

```
418.73,265.10
```

Compare against the ground truth in `sample/manifest.csv` (columns
`gt_x_corr`, `gt_y_corr` — see [Ground truth](#ground-truth-use-gt_x_corr--gt_y_corr)).

---

## The inference script

`infer.py` is the submission entry point. It loads `weights/driftsense.pt`
automatically and needs no edits.

```bash
python infer.py --reference REF.png --search SEARCH.png     # prints "x,y"
python infer.py REF.png SEARCH.png                          # positional form
python infer.py -r REF.png -s SEARCH.png --json             # adds confidence
python infer.py -r REF.png -s SEARCH.png --save-heatmap h.png
```

| flag | meaning |
| --- | --- |
| `--reference`, `-r` | reference image path (1 nm/px) |
| `--search`, `-s` | search image path (10 nm/px) |
| `--weights`, `-w` | override the weights path (default `weights/driftsense.pt`) |
| `--json` | print a JSON object instead of `x,y` (adds confidence and vote count) |
| `--save-heatmap` | write the response map as a colour image (implies `--no-tta`) |
| `--no-tta` | single view instead of 8-way voting: ~7× faster, ~0.3–1.6 points less accurate at 5 px |

Only the coordinate goes to stdout; warnings go to stderr. Device is picked
automatically (CUDA → MPS → CPU). If the weights or PyTorch cannot be loaded,
it falls back to classical multi-scale ZNCC and still prints a coordinate, so
the script always produces a scoreable answer.

**Runtime** (0.46 M parameters, no GPU required; results identical on CPU and
GPU): ~3.8 s per pair on CPU with the default 8-way TTA, or ~0.5 s with
`--no-tta`. Verified end to end in a clean virtualenv built only from
`requirements.txt`, invoked from a directory outside the repo.

**Tie-breaking.** When several candidates score within 4 % of the best, the one
nearest the centre of the Search image is returned, as the problem statement
requires. It is applied as a tie-break, not as a general centre prior — the
ground-truth location is uniform over the frame, so biasing every prediction
toward the centre would cost accuracy.

---

## Repository contents

| path | what it is |
| --- | --- |
| [`README.md`](README.md) | this file |
| [`generate_dataset.py`](generate_dataset.py) | **dataset generator** — architecture / pair count / output dir, records ground truth |
| [`infer.py`](infer.py) | **localisation inference** — reference + search → `x,y` |
| [`train.py`](train.py) | training script that reproduces the shipped weights |
| [`evaluate.py`](evaluate.py) | batch evaluation vs. the classical ZNCC baseline |
| [`weights/driftsense.pt`](weights/) | trained model weights, loaded automatically |
| [`requirements.txt`](requirements.txt) | full `pip freeze` of the dev environment |
| [`CITATIONS.md`](CITATIONS.md) | references behind the physics, noise and design choices |
| [`driftsense/`](driftsense/) | the package: model, dataset, matching, losses, generation core |
| [`generator/`](generator/) | vendored upstream synthetic-data generator (unmodified) |
| [`scripts/`](scripts/) | development tooling: parallel generation, verification, analysis |

---

## Method

### Why plain template matching fails

The layout repeats, so a 100 × 100 template correlates almost equally well at
dozens of positions. Measured on clean images with only geometry varying, the
classical ZNCC baseline locks onto the right region in only **38–83 %** of
samples depending on distortion — when it fails it is typically ~50 px away,
one full repeat over.

Three things actually break the periodicity, and the architecture is built
around them:

1. **Mat/strip composition.** Arrays are discrete blocks separated by flat
   routing strips. Globally unique, but only visible with a wide receptive
   field. 35 % of reference crops are deliberately placed to straddle a
   boundary.
2. **Grid-phase random walk.** Lines are placed with ~1–1.5 nm cumulative
   placement jitter, so over ~100 lines the grid phase drifts by ≈1.5 Search
   pixels. The lattice is genuinely non-periodic at long range.
3. **Per-line CD fingerprint.** ~10 % line-width variation survives the 10×
   downsample as low-amplitude intensity modulation. Real signal, but the
   first thing shot noise destroys at low dose.

### Architecture

```
reference (1000²) ──area-downsample 10×──► template (100²) ──┐
                                                              ├─► shared encoder (stride 4)
search (1000²) ───────────────────────────────────────────────┘        │
                                                                       ├─► grouped cross-correlation ─┐
                                                                       │                              ├─► dilated head ─► heatmap (226²)
                                                                       └─► context branch ────────────┘                └─► sub-cell offsets
```

- **Shared encoder**, stride 4, deliberately *local*. It runs on both the 25×25
  template and the full search frame; a large receptive field here would make
  the template embedding mostly padding.
- **Grouped cross-correlation** (8 groups) over L2-normalised features, so the
  match score reflects pattern agreement rather than local contrast — the two
  frames are captured at very different dose and gamma.
- **Context branch**: dilations 2/4/8/16 over the search embedding, reaching
  several hundred search pixels — the scale of the mat/strip composition.
- **Dilated head** over the *response map*, so a peak is judged against the
  surrounding lattice of decoy peaks rather than in isolation.
- **Heatmap + offset heads**, trained with a penalty-reduced focal loss (one
  positive against ~10⁴ negatives) and smooth-L1 on the sub-cell offset.
- **ZNCC refinement** at inference: a ±4 px search at full resolution with
  parabolic sub-pixel interpolation of the correlation peak. The window is
  deliberately narrow — wide enough to absorb the 4 px response-grid stride,
  too narrow to reach a neighbouring repeat. The network chooses the region;
  correlation places it. (Ablated: ±8 px costs 2 points of acc@5px.)
- **Dihedral test-time augmentation**, 8 views, cluster-voted. All eight square
  symmetries were seen in training, so every view is in-distribution; a decoy
  that wins under one view rarely wins under all eight — see
  [Test-time augmentation](#test-time-augmentation).
- **ZNCC-verified arbitration** between the surviving candidate clusters, using
  full-resolution correlation as a signal independent of the network's own
  confidence — see [Aggregation](#aggregation-which-cluster-wins).

0.46 M parameters.

### Training

14 000 pairs (12 000 multi-crop + 2 000 single-crop), 512 px search windows,
batch 8, AdamW with a one-cycle schedule. Augmentation: the 8 dihedral
symmetries applied jointly to reference and search, independent photometric
jitter per frame, random search-window crops.

Training runs in two phases.

**Phase 1 — base training (9 epochs, ~3.7 h):**

```bash
./venv/bin/python train.py --train-dirs data/train_mc data/train --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 9 --lr 1e-3 --out weights/driftsense_p1.pt
```

Best checkpoint was **epoch 6**, selected on validation accuracy at the 5 px
tolerance; epochs 7–9 kept reducing training loss without improving held-out
accuracy, and epoch 9 was measurably worse on the full validation set
(acc@5px 0.870 vs 0.883).

**Phase 2 — speckle fine-tune (~2 h)**, seeded from the phase-1 best after the
failure analysis above identified multiplicative noise as the dominant failure
driver:

```bash
./venv/bin/python train.py --train-dirs data/train_mc data/train --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 6 --lr 4e-4 \
    --resume weights/driftsense_p1.pt --out weights/driftsense.pt
```

Best checkpoint was **epoch 2** — again, later epochs overfit (loss 0.32 → 0.21
while held-out accuracy fell). ~25 min/epoch on an Apple M-series GPU (MPS).
Per-epoch history is in `weights/*_history.json`.

The same lesson appeared in both phases: **training loss kept improving long
after held-out accuracy stopped**. Both phases drew from a fixed pool of 3 500
scenes, so that signature is the pool being memorised, not the schedule running
out. Phase 3 removes the pool.

**Phase 3 — streamed training on unlimited fresh data (~6 h)**, which produced
the shipped weights:

```bash
./venv/bin/python train.py --stream --stream-length 16000 --val-dir data/val \
    --crop 512 --batch-size 8 --epochs 12 --lr 5e-4 --workers 4 --keep-epochs \
    --out weights/driftsense.pt
```

`--stream` ([`driftsense/stream_dataset.py`](driftsense/stream_dataset.py))
generates a brand-new scene for every sample inside the dataloader workers
rather than reading from disk: no scene is ever reused, and 24 000 more pairs
would otherwise have cost ~17 GB. Sample construction goes through the same
`build_sample` the on-disk dataset uses, so the two paths differ only in where
the images come from.

This is worth **+1.7 points** on `test` (0.953 → 0.970) and **+1.2 points**
over all 700 scenes, the largest single gain after the speckle fine-tune. The
overfitting signature disappears with it: training loss falls 0.410 → 0.308
across the run while held-out accuracy *rises* throughout, which is what the
earlier phases could not do.

> **A bug worth recording.** The first version of this silently did nothing.
> The dataset shifts its seed stream via `set_epoch()`, but the DataLoader was
> built with `persistent_workers=True` — and persistent workers are forked once
> and keep their own copy of the dataset, so `set_epoch()` never reaches them.
> Every epoch re-derived an identical seed and regenerated **identical scenes**:
> "unlimited fresh data" was a fixed 16 000-scene pool on repeat, reproducing
> the exact failure it was written to fix. It surfaced only on a resumed run,
> which forks its workers after `set_epoch` and so reported a much higher loss
> (0.43 vs 0.27) at the same epoch with the same weights. Fixed both by not
> using persistent workers on this path and by keying the seed on a counter
> held in the worker's own copy, so the dataset is correct under either loader
> configuration.

Checkpoint selection needed the same care. The in-loop validation runs
single-view on a 100-scene subset, which cannot separate epochs that differ by
two or three samples — it scored epochs 7 through 10 identically. `--keep-epochs`
retains every epoch so the choice can be made afterwards.

**Even the 300-scene validation split is too small to choose on.** At acc@5px
≈ 0.95 its standard error is ±1.3 points, while the differences between
candidate epochs are under 1. So selection used
[`scripts/stream_eval.py`](scripts/stream_eval.py), which evaluates on **1 000
freshly generated scenes** — generated in dataloader workers and consumed
immediately, so it costs no disk — seeded in a namespace disjoint from training
and from every on-disk split, and held fixed across checkpoints so the
comparison is paired. That halves the standard error and made the ranking
unambiguous:

| checkpoint | acc@5px (1 000 fresh scenes) | Δ vs phase 2 | 95% CI |
| --- | ---: | ---: | :---: |
| **epoch 11** (shipped) | **0.962** | **+2.9 pts** | [+1.8, +4.0] |
| epoch 9 | 0.961 | +2.8 pts | [+1.7, +3.9] |
| epoch 7 | 0.959 | +2.6 pts | [+1.5, +3.8] |
| phase-2 weights | 0.933 | — | — |

Intervals are from a paired bootstrap over scenes
([`scripts/compare_checkpoints.py`](scripts/compare_checkpoints.py)); pairing
cancels scene difficulty, which is the dominant variance term. On the 300-scene
validation split the same comparison gives +1.3 points with a 95 % interval of
[−0.3, +3.3] — right direction, no significance. The larger set is what
establishes the result.

The epoch the 100-scene in-loop metric would have picked was epoch 7, the worst
of the three candidates.

Evaluate:

```bash
./venv/bin/python evaluate.py --splits data/test data/test_medium data/test_severe
```

---

## Ground truth: use `gt_x_corr` / `gt_y_corr`

The upstream generator computes ground truth from the crop origin on the
*pre-imaging* canvas:

```
gt_x = x0 / 10 + 50
```

but the search frame is afterwards warped by `apply_raster_drift` (row shear +
per-row jitter) and `apply_barrel_distortion`. Those warps move the pattern
relative to the frame the label was written in, and **the label is never
corrected for them**. On the shipped splits the two conventions differ by a
mean of 3.1 px and up to 22 px — well past the 5 px tolerance.

Because the warps are generated here, they are invertible exactly. Both
conventions are written to every manifest:

| column | meaning |
| --- | --- |
| `gt_x`, `gt_y` | upstream convention — kept only for comparability |
| `gt_x_corr`, `gt_y_corr` | where the pattern actually is — **train and evaluate on these** |
| `label_shift_px` | distance between the two |

Verified empirically ([`scripts/verify_gt_correction.py`](scripts/verify_gt_correction.py)):
under clean photometrics with only the geometric warp varying, the residual
between a ZNCC match and the label drops from **2.5–4.8 px to 0.4–0.9 px**, and
accuracy at 5 px rises from 0.65–0.78 to **0.96–1.00** on samples where the
matcher locks onto the right region.

```
condition                    residual → gt    → corrected   acc@5px
shear = 4.0                       2.48 px        0.63 px    0.98 → 0.98
barrel k = +0.02                  3.88 px        0.44 px    0.69 → 1.00
barrel k = −0.02                  3.28 px        0.70 px    0.78 → 0.96
shear + barrel                    4.84 px        0.43 px    0.65 → 1.00
shear + jitter + barrel           4.16 px        0.92 px    0.77 → 1.00
```

**Residual noise floor.** The per-row drift jitter is i.i.d. and cannot be
predicted from the images, so it sets a hard limit on achievable accuracy.
[`scripts/label_noise_floor.py`](scripts/label_noise_floor.py) quantifies it
for a given split.

---

## Generating data

There is no downloadable dataset — the Hugging Face Space is a *generator*, and
its README states that no dataset is provided by the hackathon. All data is
produced locally and is reproducible from its seed.

```bash
./venv/bin/python generate_dataset.py --architecture dram   --num-pairs 100 --output-dir ./out/dram
./venv/bin/python generate_dataset.py --architecture finfet --num-pairs 100 --output-dir ./out/finfet
./venv/bin/python generate_dataset.py --num-pairs 100 --noise severe --output-dir ./out/hard
```

| flag | meaning |
| --- | --- |
| `--architecture` | `dram`, `finfet` or `mixed` (6 node presets each) |
| `--num-pairs` | number of Reference/Search pairs |
| `--output-dir` | destination for `reference/`, `search/`, `manifest.csv` |
| `--noise` | `randomized` (default) or fixed `low`/`medium`/`high`/`severe` |
| `--seed` | reproducibility seed |
| `--crops-per-canvas` | >1 yields many references per search frame — training only |

The splits used here (regenerate with [`scripts/gen_data.py`](scripts/gen_data.py)):

| split | pairs | conditions | seed |
| --- | --- | --- | --- |
| `train` | 2 000 | randomized | 42 |
| `train_mc` | 12 000 (1 500 canvases × 8) | randomized | 555 |
| `val` | 300 | randomized | 1234 |
| `test` | 300 | randomized | 7777 |
| `test_medium` | 200 | fixed `medium` | 20001 |
| `test_severe` | 200 | fixed `severe` | 20002 |

Seeds are disjoint, so no structure is shared between splits. `medium` and
`severe` match the fixed operating points in
`generator/baseline_solution/evaluate.py`, so numbers stay comparable to the
published ZNCC baseline.

Sample *i* is seeded from its own `SeedSequence` child, so a given `(seed, i)`
reproduces regardless of worker count — verified byte-for-byte. Note that
`train` and `val` were generated before a refactor that reordered RNG draws;
re-running reproduces `test*` and `train_mc` byte-identically, and `train`/`val`
statistically but not byte-for-byte.

Verify a generated split:

```bash
./venv/bin/python scripts/verify_dataset.py data/test
```

---

## Results

All numbers are on **held-out test splits** generated from seeds disjoint from
training and validation, measured against the geometry-corrected ground truth.
Decode settings were tuned on validation only.

| split | method | median | acc@1px | acc@2px | **acc@5px** | acc@10px |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `test` | **Siamese + TTA + ZNCC** | **0.64 px** | 0.680 | 0.883 | **0.980** | 0.980 |
| | ZNCC baseline | 26.20 px | 0.320 | 0.430 | 0.477 | 0.477 |
| `test_medium` | **Siamese + TTA + ZNCC** | **0.35 px** | 0.965 | 1.000 | **1.000** | 1.000 |
| | ZNCC baseline | 0.69 px | 0.640 | 0.705 | 0.705 | 0.720 |
| `test_severe` | **Siamese + TTA + ZNCC** | **1.38 px** | 0.360 | 0.680 | **0.960** | 0.960 |
| | ZNCC baseline | 2.49 px | 0.250 | 0.430 | 0.585 | 0.590 |

Across all **700 held-out scenes**: median **0.63 px**, acc@5px **0.980**
(14 failures), versus 0.573 for the classical baseline.

These are the **phase 4** numbers (see [TRAINING.md](TRAINING.md)). The 700-scene
set cannot establish the phase-3 → phase-4 step on its own — a paired bootstrap
gives +0.43 points with a 95 % CI of [−0.29, +1.14], spanning zero. The
significance comes from a 1 000-scene freshly generated evaluation: **+1.5
points, 95 % CI [+0.1, +2.9], P(worse) = 0.018**.

`test_medium` — the standard operating point, and the one matching the upstream
baseline's `medium` noise level — is solved **perfectly: 1.000 at both the 2 px and
5 px tolerances**, median 0.35 px. `test` draws acquisition conditions per sample
and additionally varies geometric distortion, making it the hardest of the three.

### Where the gain comes from

The failure that matters is latching onto the **wrong repeat** — an error of one
full period, tens to hundreds of pixels. That is what the network fixes:

| split | wrong-repeat rate (>10 px) — ZNCC | — ours |
| --- | ---: | ---: |
| `test` | 52.3% | **3.0%** |
| `test_medium` | 28.0% | **0.0%** |
| `test_severe` | 41.0% | **4.0%** |

Over all 700 scenes the wrong-repeat rate falls from **42.1% to 2.4%**. On
**48%** of `test`, the model is within 5 px while ZNCC is more than 10 px off.
Median error where both succeed is comparable — correlation was never the
imprecise part, it was the part that could not tell identical candidates apart.

### Failure analysis drove the final gain

Rather than train longer, the residual failures were profiled against the
generation parameters that produced them. **Speckle noise** was by far the
strongest predictor of a wrong-repeat lock-on (standardised effect **+0.59**;
failures averaged sigma 0.183 vs 0.110 for successes), followed by larger mats
(+0.35 — bigger uniform regions offer fewer boundary cues). Shear, drift jitter
and detector noise showed essentially no effect.

That exposed a genuine gap: the photometric augmentation applied gamma, gain and
*additive* noise but **no multiplicative speckle at all** — and because speckle
scales with signal, it survives the per-image standardisation that removes
additive noise. A fine-tune with continuous speckle sigma in [0.05, 0.40] plus
impulse noise closed it:

| | `test` acc@5px | `test_severe` acc@5px | all-700 acc@5px | wrong-repeat |
| --- | ---: | ---: | ---: | ---: |
| before (phase 1) | 0.940 | 0.925 | 0.953 | 4.6% |
| **after speckle fine-tune** | 0.950 | 0.945 | 0.963 | 3.6% |
| **+ ZNCC-verified aggregation** | 0.953 | 0.945 | 0.964 | 3.4% |
| **+ phase 3, streamed fresh data** | 0.970 | 0.960 | 0.976 | 2.4% |
| **+ phase 4, pooled fine-tune** | **0.980** | **0.960** | **0.980** | **2.0%** |

Inference-side fixes were tried first and rejected on measurement: median
filtering *hurt* (0.947 vs 0.953) and a homomorphic log transform — the textbook
multiplicative-to-additive trick — changed nothing. The information was not
recoverable by preprocessing; it had to be learned.

### Test-time augmentation

The eight square symmetries were all seen during training, so all eight views are
in-distribution. Each proposes a centre; proposals are mapped back to the original
frame, clustered, and the group with the greatest total confidence wins.
*Voting on agreement* rather than averaging heatmaps is the point: a wrong-repeat
proposal is an outlier one period away, so clustering discards it instead of
letting it drag the answer off.

| decode | val median | val acc@2px | val acc@5px | cost |
| --- | ---: | ---: | ---: | ---: |
| no ZNCC refine | 1.084 px | 0.787 | 0.913 | 1× |
| + ZNCC refine ±8 px | 0.720 px | 0.820 | 0.893 | 1× |
| + ZNCC refine ±4 px | 0.708 px | 0.833 | 0.913 | 1× |
| **+ dihedral TTA ×8** (shipped) | **0.631 px** | **0.867** | **0.953** | 8× |
| + 2-checkpoint ensemble (×16) | 0.638 px | 0.860 | 0.947 | 16× |

Ensembling a second checkpoint *hurt* and was dropped. Disable TTA with
`infer.py --no-tta` (~7× faster — see the measured trade-off below).

(These decode figures were measured on validation with the phase-1 checkpoint,
which is what they were used to tune. They were not re-tuned afterwards: the
shipped phase-3 weights use the identical decode settings and score higher in
absolute terms — see the tables above. Treat this table as the record of how
the decode was chosen, not as a description of the current model.)

**TTA is worth much less than it used to be.** Re-measured on the shipped
phase-3 weights, the gain is small: the model is now robust enough on its own
that voting has little left to fix.

| split | single view | + TTA ×8 | Δ acc@5px | mean error |
| --- | ---: | ---: | ---: | --- |
| `val` (300) | 0.957 | **0.960** | +0.3 pts | 12.01 → **9.90** px |
| `test` (300) | 0.963 | **0.970** | +0.7 pts | 8.62 → **5.16** px |

Per-sample on `test`, cluster-voting **fixes 3** and **breaks 1** — a net gain
of two samples. It is kept as the default because it never lost on any split
and it cuts mean error by 18–40 % (it suppresses catastrophic outliers even
when it doesn't change the accuracy count), but it costs 8× compute for a small
margin. `--no-tta` is a reasonable choice if throughput matters.

That voting can *break* a correct prediction is visible in the third row of the
figure below: single view lands 1.36 px from truth, voting pulls it 121 px away.

### Aggregation: which cluster wins

The gap above is a *decode* problem, not a model problem, so it was attacked
without retraining. The eight view proposals were cached once per scene
(`scripts/tune_aggregation.py`) and candidate rules compared offline on
validation — the test split was touched only once, afterwards.

| rule (validation, 300 scenes) | acc@2px | acc@5px | mean |
| --- | ---: | ---: | ---: |
| single strongest view | 0.837 | 0.940 | 9.19 px |
| cluster by count (majority) | 0.837 | 0.943 | 9.11 px |
| cluster by total confidence | 0.837 | 0.943 | 9.11 px |
| **ZNCC-verified + confidence prior** (shipped) | **0.840** | **0.947** | **9.02 px** |
| ZNCC verification alone | 0.820 | 0.920 | 13.04 px |

The shipped rule scores each candidate region as
`ZNCC(region) + 0.5 × (normalised network confidence)`. The network shortlists;
a full-resolution ZNCC check at each candidate arbitrates.

**ZNCC alone is a bad arbiter** — 0.920 vs 0.943 — even when it only has to
choose between three or four regions the network already proposed. It prefers
whichever candidate looks crisper, which in a periodic layout is often a decoy.
Keeping the network's confidence as a prior is what turns it from a regression
into a small win. A sweep over `alpha ∈ [0.1, 2.5]`, `top_k ∈ {2,3,4,6}` and
cluster radius `{4, 6, 10}` px is flat: every setting with `alpha` 0.35–0.5 and
`top_k ≥ 3` lands on the same result.

**Be clear about the size of this.** It is worth exactly **+1 sample in 300**
on validation *and* +1 on test — consistent in direction, never negative on any
of the four splits, and it cuts `test` mean error 9.69 → 8.38 px. That is a
real but marginal gain, well inside what a 300-sample split can resolve. It
ships because it never lost and costs ~10 ms, not because it is significant.

The oracle that always picked the best available rule would score 0.967 on
validation, so most of the gap is *not* reachable by re-ranking these
proposals: different rules fail on different scenes, and on the phase-1 weights
13 of the `test` failures were wrong under every rule and under the single view
too. Closing those needs better proposals, not better arbitration.

**That conclusion held, and phase 3 is what acted on it.** Better proposals were
exactly what the streamed fresh-data training produced, and it removed 5 of the
14 `test` failures — where every re-ranking rule swept over `alpha`, `top_k` and
cluster radius had moved at most one sample. The residual is now 9, of which 8
are wrong under the single view as well.

### Confidence is a usable reject signal

The peak confidence separates correct from incorrect predictions (ROC AUC
**0.904** on `test`; mean score 0.788 on correct predictions against 0.467 on
incorrect), so a wafer tool can flag low-confidence sites for re-acquisition
rather than reporting a wrong location:

| threshold | samples kept (`test`) | acc@5px on kept |
| --- | ---: | ---: |
| none | 100% | 0.970 |
| score ≥ 0.3 | 99.0% | **0.976** |
| score ≥ 0.5 | 95.0% | **0.989** |
| score ≥ 0.6 | 90.3% | **0.989** |

The AUC is *lower* than the 0.936 measured on the phase-2 weights even though
the model is better. That is expected and not a regression: there are now only
9 failures on `test` instead of 14, and the ones that survive are the ones the
network is confidently wrong about, so the remaining errors are harder to
separate by confidence. The reject path still works — it removes two thirds of
the residual error at a 5 % reject rate — but it is filtering a harder residue.

### Remaining failures

9 of 300 on `test`, 8 of 200 on `test_severe`, 0 of 200 on `test_medium`.
These are references landing deep inside a uniform mat, where the
disambiguating cues (mat/strip boundaries, grid-phase drift) are weakest and the
per-line CD fingerprint is buried by shot noise. At low dose that information is
physically destroyed, not merely hard to extract, so a perfect score on the
randomized split is not attainable — which is why the confidence-based reject
path above matters more than the last point of raw accuracy.

A second, smaller floor is the per-row drift jitter: it is i.i.d. noise applied
to the search frame after the ground truth is fixed, so no method can recover it
(`scripts/label_noise_floor.py` quantifies it for a given split).

### Examples

![examples](results/examples.png)

Green dashed = ground truth, red = prediction; each panel reports the shipped
TTA error, the single-view error, and the ZNCC baseline. Rows 1–2 are cases
where ZNCC lands 57 px and 95 px away — one repeat over — and the model is
within 1.7 px.

Row 3 is the honest one, and it is the single case on `test` where TTA breaks a
prediction the single view got right: ZNCC is 72 px off, the single view is
**correct at 1.36 px**, and voting lands 121 px away. It is also *not* caught by
the reject path — its confidence is 0.755, comfortably above every threshold in
the table above. That is the uncomfortable version of this failure mode: the
network is confidently wrong, so neither aggregation nor a confidence cut
recovers it. The phase-2 model had an analogous failure that scored 0.389 and
would have been rejected; as the easy failures were trained away, the ones left
are the ones confidence cannot flag.

Regenerate with
`python scripts/visualize.py --split data/test --ids 3 5 153 --out results/examples.png`.

---

## Attribution

The synthetic-data generator in [`generator/`](generator/) is
[`aayushraina21/drift-sense-synthetic-data`](https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data),
vendored unmodified. This project adds the geometry-corrected ground truth,
the reproducible generation wrapper, the learned localiser, and the evaluation
harness. Full references in [`CITATIONS.md`](CITATIONS.md).

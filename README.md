# Drift-Sense — Navigation-Error Recovery, Phase 2

Locate a high-resolution **Reference** patch inside a low-resolution **Search**
frame of a repeating semiconductor layout, and report where it is — position,
rotation, scale, and whether it's there at all. This is Applied Materials'
*Navigation-Error Recovery* problem, Phase 2: unknown pose, possible absence.

|            | Reference        | Search             |
| ---------- | ---------------- | ------------------ |
| size       | 1000 × 1000 px   | 1000 × 1000 px     |
| pixel size | 1 nm/px          | `z` nm/px, `z ∈ [8, 12]`, unknown per pair |
| field      | 1 µm             | ~8–12 µm            |

Phase 1 fixed the zoom at exactly 10× and guaranteed the reference was always
present. Phase 2 removes both assumptions — the zoom is unknown in `[8, 12]`,
the rotation is unknown in `±5°` (CCW positive) and must be reported, and
about 20% of pairs contain **no true instance** at all. The underlying
localiser is the Phase 1 network, extended rather than replaced: see
[How it works](#how-it-works).

## Entry point & output contract

```bash
python register.py --input pairs.csv --output predictions.csv
```

One row per input pair, in input order: `pair_id, x, y, theta, scale, found,
score`. `x, y` is the match centre in Search-image pixels; `theta` is degrees,
CCW positive, about the match centre; `scale` is the recovered down-scaling
factor `z` (not `1/z`). When `found=0`, every pose column is written `0`. A
pair that fails for any reason — bad image, exception, timeout — still emits a
row: **a missing row scores zero, so declining beats disappearing.**

Runs CPU-only, no network access, weights load from `weights/driftsense.pt`
automatically. Reference machine: 4-core x86, 8 GB RAM, no GPU, Python 3.11;
median ≤5 s/pair, 20 s hard timeout.

## Quick start

Requires **Python 3.11**, matching the reference machine. `requirements.txt`
is a frozen `pip freeze` from a 3.11, CPU-only PyTorch environment.

```bash
python3 -m venv venv && ./venv/bin/pip install -r requirements.txt
```

Generate a Phase-2-style sample and run the graded entry point on it:

```bash
./venv/bin/python generate_dataset.py --phase2 --num-pairs 4 --output-dir ./sample
./venv/bin/python register.py --input sample/manifest.csv --output sample/predictions.csv
```

`sample/manifest.csv` already has the columns `register.py` looks for
(`id`/`reference_path`/`search_path`), so it doubles as a `pairs.csv`. Compare
`sample/predictions.csv` against the manifest's `gt_x_corr`/`gt_y_corr` (see
[Ground truth](#ground-truth-convention)).

For a single pair without building a CSV, `infer.py` is a thin, Phase-1-era
compatibility CLI (`python infer.py -r REF.png -s SEARCH.png`, prints `x,y`) —
useful for a quick manual check, but it is **not** what's graded; `register.py`
is the one entry point the reference machine runs.

## Results

**Current shipped model** (`weights/driftsense.pt`, the 1.02M "wide" checkpoint
— see [Model & training](#model--training)) against the 0.456M model it
replaced, no-band, at each model's locally-optimal threshold:

| model | threshold | total /85 | loc A | loc B | rejection F1 | calibration AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.456M (previous) | 0.2007 | 75.96 | 0.9705 | 0.7847 | 0.8893 | 0.9871 |
| **1.02M, shipped** | 0.1587 | **76.97** | 0.9737 | 0.8183 | 0.9061 | 0.9883 |

Source: `.agents/WIDE.txt`. The wide model cleared the promotion gate against
the previous one on every component. Note the table's locally-optimal
threshold (0.1587) differs from the currently configured
`driftsense/config.py:SHIPPED_THRESHOLD` (**0.18**, tuned against the full
rubric rather than this component table alone) — the two weren't re-measured
together at exactly 0.18.

**75.92–76.97 of 85 self-measurable points**, depending on snapshot (see
above). Efficiency (5 pts) is a relative ranking we can't self-assess, and the
generator/citations/failure-analysis component (10 pts) is judged separately.

Several further validated gains landed after both snapshots above (sub-pixel
row correction, **+0.59 localisation** on the canonical 2,500-pair paired A/B
— `driftsense/config.py`; a Set-C rejection fine-tune, +0.33; an
uncontested-hypothesis early exit; a fail-closed fix for a model-load
degradation bug) without yet being folded into one fresh full rescore against
the current checkpoint. Treat the table above as a conservative floor;
`FAILURE_ANALYSIS.md` carries the itemized, dated findings for everything
since.

### Spec-composition campaign (5 × 200 pairs, local x86 emulation of the published reference constraints)

A second, independent measurement under the published reference constraints,
emulated on local x86 hardware: five 200-pair sets
(seeds 1–5) composed **A70 / B70 / C40 / D20** per slide 4, built with the
spec recipe on top of the vendored generator, reusing the reviewed Issue 45
audit fixture for pose construction, seeding and label verification.
It is **not** organizer-issued data and these
1,000 pairs are **not** the official blind benchmark.

| | loc /40 | scale /10 | rot /10 | reject /15 | calib /10 | **/85** | Set D | bonus |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mean of 5 sets | 36.65 | 9.39 | 8.91 | 14.63 | 9.96 | **79.54** | 0.984 | **+10** |
| sd | 0.98 | 0.12 | 0.29 | 0.22 | 0.02 | **1.08** | 0.015 | 0 |
| worst set | 35.38 | 9.23 | 8.51 | 14.29 | 9.93 | **77.76** | 0.970 | +10 |
| best set | 37.96 | 9.57 | 9.25 | 14.81 | 9.98 | **80.67** | 1.000 | +10 |

Per-set results (seeds 1–5):

| set | loc /40 | scale /10 | rot /10 | reject /15 | calib /10 | **/85** | F1(rej) | Set D | median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 36.17 | 9.57 | 9.25 | 14.63 | 9.93 | **79.54** | 0.9750 | 1.000 | 1.164 s |
| S2 | 35.38 | 9.38 | 8.74 | 14.29 | 9.97 | **77.76** | 0.9524 | 1.000 | 1.099 s |
| S3 | 37.96 | 9.40 | 8.51 | 14.81 | 9.98 | **80.67** | 0.9877 | 0.970 | 1.099 s |
| S4 | 36.58 | 9.39 | 9.08 | 14.81 | 9.98 | **79.84** | 0.9877 | 0.980 | 1.202 s |
| S5 | 37.17 | 9.23 | 8.95 | 14.63 | 9.94 | **79.92** | 0.9756 | 0.970 | 1.117 s |

Context: the earlier campaign (PR #51) measured on an Apple M4, arm64, while
the task material names a 4-core x86 CPU with 8 GB RAM, no GPU, no network and
Python 3.11; this campaign emulates those constraints on local x86 hardware —
each cap read back and verified from inside the running process — on five sets
instead of three so the
between-set spread is visible. Latency pooled over all 1,000 pairs, idle
machine, 4-thread cap: median 1.147 s, mean 1.221 s, p90 1.883 s, p99 2.083 s,
max 2.191 s — zero pairs over the 5 s median budget, none within 9 s of the
20 s hard timeout.

Both bonus gates are met on all five sets under both readings of the +6
condition: Set D credit **0.984** against a 0.40 gate, and the Sets A–C
localisation credit **0.916** against a 0.50 gate (per-set: A 0.992, B 0.854).
So **+10 of the +10**.
The sd is the number to read next to the mean: at 1.08 points across five sets,
a difference under about a point is not a result at this sample size.

**This is not a gain over the 75.92–76.97 above — it is a different dataset.**
Those figures are 2,250 pairs of `data/ext_p2`, which is the harder and more
conservative of the two and **remains the planning number of record**.

Within this internal spec-composition campaign, Set B accounts for essentially
all of the observed gap, and inside it the loss is monotone in
severity: credit 0.964 / 0.911 / 0.833 / **0.699** at severity levels 1–4
(350 B pairs). Severity 4 alone costs ~1.5 of the 5.5 missing points. Of the 9
present pairs wrongly declined across 1,000, 8 were severity 3–4 and every one
had a true error of 291–1004 px — genuine lock-on failures the rejector
correctly declined, not calibration slop. Set C gave up exactly **1** false
accept in 200.

#### Retired: the 81.45 / 81.93 figures

PR #51 reported 81.45 mean / 81.93 best on three 200-pair sets. **Those are
withdrawn**, and the cause is a generator defect, not a regression.

`driftsense.generate.build_one` only draws the coherent per-knob degradation
when its severity range satisfies `_shi > _slo`. A severity pinned as a single
point (`lo == hi`) fails that strictly-greater test, so the pair renders as
**generic per-knob draws at severity 0.0** while still carrying the label
"Set-B severity N" — issue #31, fixed in the audit fixture with a 1e-6 epsilon
that `gen_200.py` inherits. PR #51's generator script was never committed, so
what it did cannot be read; the defect can, however, be reproduced and priced.
Identical code, identical seed, identical box and pinned cores — the only
difference being `severity=(t, t)` versus `severity=(t, t + 1e-6)`:

| | realised Set B severity | Set B credit | **/85** | median |
| --- | --- | ---: | ---: | ---: |
| ladder fires | 0.25 / 0.50 / 0.75 / 1.00 | 0.8257 | **79.54** | 1.164 s |
| degenerate pin | **0.0 on all 70 pairs** | 0.9343 | **81.57** | 0.897 s |

81.57 reproduces the old headline on demand by switching the degradation off,
and PR #51's reported Set B credit (0.9086 / 0.9143) sits in the legacy arm's
range, not the degraded one. The easier data also ran *faster* — 0.897 s vs
1.164 s, because undegraded pairs trip the early-exit gates more often — so
both of that PR's leading numbers moved for the same reason.

One caveat bounds this the other way: mapping level 4 to `severity_continuous
= 1.0` puts a quarter of Set B at the generator's **ceiling**. Slide 4 discloses
that four levels exist, not how hard they are — the organizer severity
parameters are undisclosed, and their blind data need not vary only along this
generator severity scalar. Under our two internal severity constructions, this
sensitivity experiment brackets the **synthetic-campaign** score at
**79.54–81.57**; the organizer blind score is not implied to lie in this
interval. We plan on 79.54.

**Runtime**, measured end-to-end with `register.py`, shipped 4-thread cap:

| hardware | pairs | median | p90 | max | constraint verified |
| --- | ---: | ---: | ---: | ---: | --- |
| **x86, 4 pinned P-cores, 8 GB cap** | **1,000** | **1.147 s** | **1.883 s** | **2.191 s** | yes — see below |
| x86, 4 pinned E-cores, 8 GB cap | 200 | 2.859 s | 4.538 s | 4.972 s | yes |
| Apple M4, arm64 (development machine) | 600 | 0.960 s | 1.343 s | 1.637 s | n/a — not x86 |
| x86 (AMD Ryzen, 4-thread cap) | — | 2.66 s | 6.16 s | 7.16 s | thread cap only, cores not pinned |

The first two rows are the reference-machine-constrained local x86 campaign: a
4-core x86 box with 8 GB and no GPU, emulated with the cap **read back from
inside the running process** rather than assumed; this section carries the
summary of that campaign.

| constraint | mechanism | realised |
| --- | --- | --- |
| 4 cores | `taskset`, 4 **distinct physical** cores (no HT siblings) | `sched_getaffinity` → 4 CPUs |
| 8 GB RAM, no swap | systemd scope, `MemoryMax=8G`, `MemorySwapMax=0` | cgroup `memory.max` 8589934592, `swap.max` 0 |
| no GPU | CPU-only venv | `cuda.is_available()` False, torch 2.13.0+cpu |
| no network | `unshare -n` spot-check | 0 interfaces, all rows still written |
| peak memory | `/usr/bin/time` | 0.93–1.07 GB — 13% of the cap |

Every row clears the 5 s median budget, and **no pair in 1,000 came within 9 s
of the 20 s hard timeout** — the slowest single pair was 2.191 s. Two caveats
the older rows understate:

* The **sub-2 s median is a property of the faster core, not of the code.** The
  same 200 pairs on 4 E-cores gave a 2.859 s median with a **bit-identical
  decode** (subtotal equal to 4 decimal places, every component identical). Any
  "1.15 s/pair" claim has to name the core it was measured on.
* The AMD Ryzen row's **p90 and max exceed 5 s**, so "comfortably clears the
  budget" was only ever true of its median. Its spread (p90 = 2.3× median)
  is the signature of a machine that was not idle; the pinned rows above hold
  p90 at 1.6× median. For calibration, a 5-pair spot check taken here *while
  the generator held cores* measured a 5.90 s median on the same binary and the
  same pinned cores that gave 1.15 s idle — a 5× inflation from background load
  alone. Only idle-machine, serial readings are quoted in the first two rows.

## The `score` column: what our confidence means

`score` lies in `[0, 1]`, higher means more confident the reference is
present at the reported `(x, y)`. It's monotonic, not a calibrated
probability — only its ordering is claimed.

It's formed from two signals the pipeline already computes: the **network
confidence** (sigmoid of the winning cell in the response map — *which*
repeat is correct) and the **native ZNCC** at the recovered pose, full
resolution (*does* the reference actually sit here). They fail differently —
the network can be confident on a plausible wrong repeat; ZNCC can be
respectable on a degraded frame with no true instance — so requiring both to
be high is what separates present from absent.

The threshold (`0.18`) is deliberately biased low: declining a present pair
forfeits its localisation (40 pts) and pose (20 pts) credit as well as hurting
rejection F1, while accepting an absent pair only costs F1. It's chosen
against the total rubric, not F1 alone (`scripts/optimize_threshold.py`).

## Verification: which hypothesis wins

The pose search returns up to three candidates; native-resolution ZNCC
(`verification="zncc"`) picks the winner by default. A wrong scale/rotation
basin correlates near zero at full resolution while the right one sits around
0.9, so this decision is easy even when the coarse probe couldn't make it.

Verification only reaches about a quarter of the pairs that miss the 5px
tier — most misses never had a correct candidate generated in the first
place, which is a search problem, not a ranking one (see
`FAILURE_ANALYSIS.md`). A rank-transform selector (the textbook defence
against impulse noise, our second-strongest failure discriminator) was
measured and rescues as many failures as it breaks — a net loss against plain
ZNCC, so it stays available for study but unused for selection.

## How it works

The layout repeats, so a 100×100 template correlates almost equally well at
dozens of positions — local appearance alone can't identify a site. The
network does the one hard thing it was trained for (deciding *which* repeat
is correct on matched-scale input); pose search is built around it:

```
reference (1000²) ──area-downsample──► template ──┐
                                                    ├─► shared encoder ─► cross-correlation ─► response map
search (1000²) ─────────────────────────────────────┘
```

1. **Pose hypotheses.** Correlation against a periodic layout is multi-peaked
   in scale — a wrong magnification can outscore the true one on a
   low-resolution probe. The top few local maxima of the coarse scale sweep
   are kept, not just the best.
2. **Canonicalisation.** Each hypothesis un-rotates and un-scales the search
   frame to the nominal 10×, so the network sees the distribution it was
   trained on.
3. **Native-resolution verification.** Each candidate is ZNCC-verified at
   full resolution; the best one wins.
4. **Pose polish.** Scale and rotation are re-fit against the known location,
   in a window around it, in the **native** frame — never the canonical one —
   so the reported centre never inherits resampling blur.

Two engineering bugs were worth fixing along the way, both measured:

- **A convention mismatch** between the nominal-path crop label (`x0/10 + 50`)
  and the posed-path affine (pixel-centre convention) biased every posed
  training target by a constant `(m-1)/2m` — 0.45 px at 10×, invisible at 5px
  tolerance but most of the budget at 1px. Fixing it moved ≤1px accuracy from
  57% to 68%.
- **The scale template was quantised.** `cv2.resize` to an integer pixel count
  meant only 43 attainable magnifications across `[8,12]`, in steps
  0.81–1.22% wide — as wide as the entire full-credit scale tier, so any
  search over `m` was optimising a piecewise-constant objective. Folding the
  residual sub-integer scale into the existing rotation affine fixed it at no
  extra cost: realisation error fell from a median 0.26% to 0.012%.

Measured against a true-pose oracle, the unchanged localiser reaches 99% at
5px; with one estimated pose, 83%. Nearly every localisation failure was a
pose failure, not a network failure — which is why searching multiple
hypotheses recovers most of them.

## Generating Phase 2 data

```bash
python generate_dataset.py --phase2 --num-pairs 200 --output-dir data/val_p2
```

`--phase2` samples magnification and rotation **per pair** over the disclosed
bounds and emits absent pairs (reference cropped from an independently
generated die region of the same architecture — periodically similar, no true
instance). The manifest gains a `found` column; absent rows carry a `-1`
sentinel in every geometry column so scoring code that forgets to filter on
`found` fails loudly rather than quietly.

```bash
python scripts/eval_phase2.py data/val_p2
```

## Requirement compliance

The problem statement requires the generator to model *"independent sensor
noise per image, edge-brightening, blur, rotation, and scaling variations"*
(Phase 1 prompt), plus, for Phase 2 Set B, *"charging, scan distortion,
defocus, elevated shot noise, and polygon scaling ±20%"* across four
undisclosed severity levels.

| required | where | ref |
| --- | --- | --- |
| independent sensor noise per image | shot, detector, speckle, impulse noise, drawn per frame | [CITATIONS §2](CITATIONS.md) |
| blur | `gaussian_psf_blur`, with astigmatism | [CITATIONS §2](CITATIONS.md) |
| edge-brightening | `apply_edge_brightening`, `--edge-brightening` | [CITATIONS §10](CITATIONS.md) |
| rotation | `--rotation-deg` / `--rotation-range`, affine sampling | [CITATIONS §10](CITATIONS.md) |
| scaling variations | `--magnification` / `--magnification-range` | [CITATIONS §10](CITATIONS.md) |
| charging | `charging_streak_prob`, `charging_streak_intensity` | [CITATIONS §2](CITATIONS.md) |
| scan distortion | `shear_amplitude_px`, `drift_jitter_px`, `barrel_distortion_k` | [CITATIONS §3](CITATIONS.md) |
| defocus | `beam_spot_size_nm`, `astigmatism_ratio` | [CITATIONS §2](CITATIONS.md) |
| elevated shot noise | `dose_search`, `detector_noise_sigma_search`, `speckle_sigma` | [CITATIONS §2](CITATIONS.md) |
| polygon scaling ±20% | `polygon_scale_fraction` (multiplicative CD change, pitch fixed) | [CITATIONS §10](CITATIONS.md) |

All pose/edge/severity knobs default to the nominal no-op, so a plain
`generate_dataset.py` run with no flags reproduces byte-for-byte. **The
shipped weights were trained without them** — pose is handled entirely by the
inference-side search in [How it works](#how-it-works), not by training on
rotated/rescaled data.

**Attribution.** The synthetic-data generator in [`generator/`](generator/) is
based on / vendored from
[`aayushraina21/drift-sense-synthetic-data`](https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data).
Phase 2 extensions (`polygon_scale_fraction`, `severity_continuous`, variable
canvas size, pitch-factor support) and the audit/deliverable tooling
(`generate_phase2.py`, `baseline.py`, `score.py`, `contact_sheet.py`,
`REPORT.md`) are added in this repository. This project also adds the
geometry-corrected ground truth, the learned localiser, and the evaluation
harness. Full references: [`CITATIONS.md`](CITATIONS.md).

## Examples

![examples](results/examples.png)

Green dashed = ground truth, red = prediction. Rows 1–2: cases where the
classical ZNCC baseline lands one full repeat away (57–95 px) and the model is
within 1.7 px. Row 3 is the honest one — a confidently-wrong prediction that
no aggregation or confidence cut catches, because its score sits comfortably
above threshold. Regenerate with
`python scripts/visualize.py --split data/test --ids 3 5 153 --out results/examples.png`.

## Model & training

Siamese correlation network, 19 `Conv2d` layers, no attention — a
dilated-convolution `ContextBranch` over the response map supplies the
long-range context needed to disambiguate periodic repeats. The architecture
is configurable (`width`/`ctx`/`head`); the **shipped checkpoint**
(`weights/driftsense.pt`) is the wide variant, `width=96/ctx=48/head=96`,
**1.02M parameters** — loaded via `net_from_checkpoint()`, which reads the
architecture from the checkpoint's own `arch_kwargs` rather than assuming the
class defaults (`width=64/ctx=32/head=64`, 0.46M, the earlier Phase 1-era
size). Verify what's actually loaded with:

```bash
python -c "import torch; ck=torch.load('weights/driftsense.pt', weights_only=True); print(ck['arch_kwargs'])"
```

Trained in four phases (base → speckle fine-tune → streamed unlimited data →
large-pool fine-tune), then a further Phase 2 retraining lineage that
produced the shipped wide checkpoint, entirely on generated data with
disjoint seeds between every split.

Full methodology, every training command, checkpoint-selection evidence,
negative results, and reproduction steps: **[`TRAINING.md`](TRAINING.md)**.

## Ground truth convention

The upstream generator computes ground truth from the pre-imaging crop origin
(`gt_x`, `gt_y`), but the search frame is then warped by raster drift and
barrel distortion — moving the pattern relative to the frame the label was
written against, uncorrected. Both conventions are written to every manifest;
**train and evaluate on `gt_x_corr` / `gt_y_corr`**, not `gt_x` / `gt_y`. Full
empirical validation of why: [`TRAINING.md` §2](TRAINING.md).

## Repository contents

| path | what it is |
| --- | --- |
| [`register.py`](register.py) | **Phase 2 entry point** — `pairs.csv` → `predictions.csv`, the graded command |
| [`driftsense/`](driftsense/) | the package: model, matching/pose-search, generation core, shared runtime |
| [`generator/`](generator/) | vendored upstream synthetic-data generator, plus the Phase 2 generator deliverables (`generate_phase2.py`, `baseline.py`, `score.py`, `contact_sheet.py`, `REPORT.md`) |
| [`weights/driftsense.pt`](weights/) | trained model weights, loaded automatically |
| [`generate_dataset.py`](generate_dataset.py) | dataset generator — architecture, pair count, `--phase2` pose/absence sampling |
| [`train.py`](train.py) | training script that reproduces the shipped weights |
| [`evaluate.py`](evaluate.py) | batch evaluation vs. the classical ZNCC baseline |
| [`infer.py`](infer.py) | legacy single-pair CLI (Phase 1 era); not the graded entry point |
| [`tests/`](tests/) | `pytest` suite over coordinate, label, and CLI invariants |
| [`scripts/`](scripts/) | development tooling — generation, verification, analysis, and the submission ZIP builder/checker |
| [`failure_analysis.pdf`](failure_analysis.pdf) | the required 2-page failure analysis, built by `scripts/failure_analysis.py` from a results CSV — independently authored from, and not auto-synced with, `FAILURE_ANALYSIS.md`'s prose; keep both current by hand |
| [`requirements.txt`](requirements.txt) | full `pip freeze` of the environment `register.py` runs in |
| [`phase1/`](phase1/) | frozen archive of the pre-Phase-2 codebase, kept for history — not part of this submission |

## Building the submission ZIP

The graded artifact is a ZIP the organizers extract and run, not a link to this
repo, so it is built from an explicit allow-list rather than from an archive of
`main` — which would otherwise carry `phase1/`, `.agents/`, and 48 MB of unused
checkpoints into whatever a judge opens.

```bash
python scripts/build_submission_zip.py --out dist/submission.zip
```

That one command builds *and* audits: an unaudited artifact is not evidence of
anything, so the checker runs on the finished ZIP automatically and its verdict
is the command's exit status. The builder prints what it shipped and aborts if
a manifest entry has gone missing or a denied path would leak; `--list` prints
the resolved manifest without writing anything, and `--no-audit` skips the
audit when inspecting a deliberately partial build.

The audit extracts the finished ZIP into a temporary directory and audits
*only* that extraction — layout, a real `torch.load` plus `infer.load_model` of
the shipped checkpoint, `--help` smoke tests, an import-closure network scan,
PDF page count, and requirements pins. It can also be run by hand against any
ZIP:

```bash
python scripts/check_submission_zip.py dist/submission.zip
```

`tests/test_submission_manifest.py` holds the manifest itself to contract.

## Further reading

- [`FAILURE_ANALYSIS.md`](FAILURE_ANALYSIS.md) — current failure modes, what was tried and measured, what's still open
- [`TRAINING.md`](TRAINING.md) — full training methodology, checkpoint selection, reproduction
- [`CITATIONS.md`](CITATIONS.md) — references behind the physics, noise, and design choices

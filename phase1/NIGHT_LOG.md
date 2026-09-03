# Overnight run log — 2026-08-16 → 08-17

Working autonomously while you sleep. **Nothing that ships today has been
touched.** `weights/driftsense.pt` is backed up byte-for-byte at
`weights/driftsense_ship_backup.pt`; if every experiment below fails, restore
that file and the submission is exactly as you left it.

Machine held awake with `caffeinate -i -m -s -t 43200` (12 h). On AC the system
already had `sleep 0`, so this is belt-and-braces.

---

## The plan, and why this order

Three candidate levers, sequenced cheapest-information-first rather than by
expected size of gain — the cheap one changes what the expensive ones should be.

### E1 — BatchNorm recalibration at full frame *(~15 min, forward-only)*

Your headroom note ranks "larger training crop" second, on the grounds that the
head is trained on a 104×104 response map and tested on 226×226. Searching the
literature for that mismatch lands on Touvron et al., *Fixing the train-test
resolution discrepancy* (arXiv:1906.06423), which makes a sharper claim than
"train bigger": most of the damage from a test-time resolution change is
**skewed activation statistics**, and the cheapest fix is to re-estimate
BatchNorm's running mean/variance at test resolution while leaving every weight
untouched.

So I measured it before running it, with `scripts/bn_stats_gap.py`: forward-hook
every BN layer, compare the activation statistics it actually sees at inference
against the running statistics it has stored. **The prediction was wrong, and
the measurement found something bigger.**

```
BN layer            d_mean@512  d_mean@1000   var@512   var@1000
corr_mix.1               0.171        0.152     0.855      0.798
context.body.3.1         0.154        0.160     0.929      1.033
head.0.1                 0.180        0.149     0.835      0.792
head.1.1                 0.163        0.152     0.779      0.741
head.2.1                 0.169        0.162     0.613      0.599
head.3.1                 0.255        0.197     0.522      0.521
```

`var` is the ratio of true activation variance to stored running variance; 1.0
means the stored statistics are right.

Two readings, and the second is the important one:

1. **The resolution change barely matters.** Every layer's variance ratio moves
   only a few percent between a 512 and a 1000 px frame — head.3.1 goes 0.522 →
   0.521. The padding-fraction argument above is real but tiny. So a larger
   training crop is *not* the lever your headroom note hoped it was, and E3 is
   worth less than I thought an hour ago.
2. **The statistics are badly wrong at both sizes.** head.3.1 sees activations
   with barely *half* the variance BN has stored (0.52), and means off by 0.2–0.26
   running-std units. That is not a resolution artefact — it is the photometric
   augmentation. Training jitters gamma, gain, additive noise, speckle up to
   sigma 0.40 and impulse noise; inference sees none of it. The running stats
   were averaged over augmented batches, so at inference every normalised
   activation in the head is scaled down by roughly sqrt(0.52) ≈ 0.72 relative
   to what the learned affine was fitted to expect. The deeper into the head,
   the worse it gets.

That is a systematic miscalibration sitting in the shipped model right now, and
it is exactly what recalibration removes — `scripts/bn_recalibrate.py` re-estimates
on clean, un-augmented full frames, matching inference on both axes at once. It
runs forward-only with `momentum=None` (exact cumulative average, not an EMA):
no gradients, no weight updates, labels never read. It cannot overfit; it is not
a training run.

So E1 keeps its slot — but for a different and better reason than the one that
put it there. It is now the most promising cheap thing on the list, and it
demotes E3 rather than motivating it.

### E2 — Long streaming continuation at crop 512 *(fills the night)*

Your preference #1, resumed from where you stopped it so the one-cycle LR
schedule completes rather than restarting. Targets the deficiency the data
actually shows — both fixed-dataset phases overfit a 3 500-scene pool.

### E3 — Full-frame (crop 1000) streaming fine-tune *(~1.5 h, last)*

`train.py`'s own docstring already describes this as the intended final step and
it was never run. Sequenced last so it fine-tunes the *best* crop-512 model
rather than a stale one — doing it first would mean redoing it afterwards.

### Not attempted, and why

- **TTA aggregation.** You swept it flat; 13 test failures are wrong under every
  rule and under the single view. Better proposals, not better arbitration.
- **Wider model (64 → 96).** Capacity is not the binding constraint until the
  data question is settled, and one night cannot settle both.
- **Distractor-aware loss** (DaSiamRPN-style explicit hard-negative mining on
  the decoy peaks). Genuinely promising for the wrong-repeat failure mode and
  the closest thing to a *new* idea here — but it is a loss change that needs a
  full retrain to evaluate, which does not fit alongside E2 and E3. Flagged for
  a future night, not started.

---

## Results

All on the **full 300-scene `data/val`**, with the shipped decode (8-way TTA +
ZNCC-verified aggregation) — i.e. run exactly the way `infer.py` runs. The
100-scene subset used inside the training loop cannot separate these.

| checkpoint | median | acc@1px | acc@2px | **acc@5px** |
| --- | ---: | ---: | ---: | ---: |
| `driftsense.pt` (shipped) | 0.67 px | 0.647 | 0.840 | 0.947 |
| `driftsense_v3.pt` (stream, epoch 1) | 0.66 px | 0.647 | 0.840 | **0.953** |
| `driftsense_v3_last.pt` (stream, epoch 3) | **0.64 px** | **0.653** | 0.840 | **0.953** |
| `driftsense_bnrecal.pt` (E1) | 0.69 px | 0.640 | 0.830 | 0.937 |

### The streaming run is genuinely ahead — but only just

Both streaming checkpoints beat the shipped model by 0.6 points, which is **+2
samples out of 300**. Treat that as encouraging, not as established: the
binomial standard error at p ≈ 0.95, n = 300 is ±1.3 points, so a 2-sample lead
is well inside the noise. What makes it worth continuing is not the size of the
lead but its direction combined with *where in the schedule it was taken* —
epoch 3 of 12, with the one-cycle LR still at 4.4e-4 and barely into its anneal,
and with sub-pixel precision still visibly improving (median 0.67 → 0.64,
acc@1px 0.647 → 0.653 while acc@5px held). Nothing looks converged.

One correction to the note you left: `weights/driftsense_v3.pt` is the **epoch-1**
checkpoint, not the best of the run. Epochs 1 and 3 tie exactly on the selection
score `(1-acc@5px)*1000 + median` at n=100 (both 60.6045), and the save is
gated on strict `<`, so epoch 3 never overwrote it. Epoch 3 is the better model
on the full split. The long run resumes from `_last.pt`, so it continues from
epoch 3 correctly.

### E1 — BatchNorm recalibration: rejected, it makes things worse

**0.937 vs 0.947 — three samples worse.** The intervention was well-motivated
and the diagnosis behind it was measured, not assumed, and it still failed.

The reason is worth writing down, because it says the "miscalibration" I
measured is not a defect. During training BN normalises with the **batch**
statistics of *augmented* data, so every downstream weight was fitted to expect
inputs scaled that way. The stored running statistics are the average of exactly
those augmented batch statistics — so using them at inference *reproduces the
training-time normaliser*, which is the consistent thing to do. Re-estimating
them on clean frames replaces that with a normaliser the network was never
trained under. The variance ratio of 0.52 in the deep head is not an error to be
corrected; it is the network correctly seeing that inference data is cleaner
than training data.

So the shipped behaviour was right and the FixRes remedy does not transfer here.
Recorded rather than deleted, since "we tried the textbook fix and it cost a
point" is the useful form of this result. `weights/driftsense_bnrecal.pt` is
kept only as evidence and is **not** a promotion candidate.

### E3 — full-frame fine-tune: deferred, with one caveat against myself

`scripts/bn_stats_gap.py` shows the 512 → 1000 frame change moves activation
statistics by only a few percent per layer (head.3.1: 0.522 → 0.521), so the
*normalisation* half of the train/test gap is negligible.

**That does not refute your original argument, and I should not claim it does.**
Your point was about the number of decoys, not the activation statistics: at
test the head picks an argmax over 226² = 51 076 cells having only ever been
trained to pick one over 104² = 10 816, so it faces 4.7× more chances to be
beaten by a false peak. Nothing measured tonight speaks to that. It is still an
open, plausible lever for the wrong-repeat failures specifically.

What argues for deferring it is scheduling, not evidence: it costs ~1.5 h, it
has to run *after* the streaming run to be worth anything (fine-tuning a stale
model then redoing it is wasted), and the slack after the run finishes is better
spent making the promotion decision reliable than on a second speculative model.
Mitigating the risk in the meantime: the in-loop validation already runs at full
1000 px frames, so every checkpoint is *selected* on full-frame behaviour even
though it is trained at 512.

Worth doing on a future night. Not dropped on the merits.

### Bug found: the streaming run was never actually streaming

**`driftsense/stream_dataset.py` was replaying the same 16 000 scenes every
epoch.** The v3 run you left was not training on unlimited fresh data. It was
training on a fixed pool, four times over.

I caught it because the resumed run's loss was wrong. Same weights, same epoch,
same LR — and a completely different loss:

```
v3  e4 [400/2000]  loss 0.271  focal 0.167  batch median err 0.6px
v4  e4 [400/2000]  loss 0.431  focal 0.328  batch median err 1.8px
```

v4 at epoch 4 was behaving like v3 at epoch *1* (loss 0.432). A resumed run
cannot be three epochs less trained than the checkpoint it resumed from, so
either the resume was broken or the two runs were not seeing the same data.

The cause is `persistent_workers=True` in `train.py`'s DataLoader. Persistent
workers are forked once and keep **their own copy** of the dataset object, so
`train_ds.set_epoch(epoch)` — which the streaming dataset relies on to shift its
seed stream — mutates only the parent's copy and never reaches them. Each
worker's `self.epoch` stays frozen at its value when it was forked, so every
epoch re-derives `SeedSequence([seed, epoch, wid])` with identical inputs and
regenerates identical scenes.

Verified directly rather than argued (`persistent_workers` toggled, everything
else held fixed):

```
persistent_workers=True    epoch0: ['4e7fc5db…','5ff5ae1e…','6918f1d5…','7c57c72a…']
                           epoch1: ['4e7fc5db…','5ff5ae1e…','6918f1d5…','7c57c72a…']  identical
persistent_workers=False   epoch1: ['aeaa1d06…','b81937c6…','cf497147…','e444f776…']  fresh
```

Why the resumed run exposed it: a fresh process forks its workers *after*
`set_epoch(4)`, so v4 drew genuinely new scenes and reported the honest loss on
data it had never seen. v3 forked its workers at epoch 0 and never escaped that
pool. So v3's falling loss (0.454 → 0.297) was substantially **memorisation of a
fixed 16 000-scene set**, not learning — the exact failure mode the streaming
dataset was written to eliminate, reproduced inside the fix for it.

Two consequences worth being clear about:

- The measured accuracies stand. `driftsense_v3_last.pt` really does score 0.953
  on the full val split; how it was trained does not change what it scores.
- **The experiment had not actually been run.** "Does unlimited fresh data close
  the gap?" was never tested, because the data was never unlimited. That makes
  tonight's run more interesting than a continuation, not less.

Fixed in two places, and the run was restarted from `driftsense_v3_last.pt`
having written no checkpoints, so nothing was lost:

- `train.py` — `persistent_workers=False` on the streaming path. Re-forking four
  workers costs seconds against a ~45 min epoch.
- `stream_dataset.py` — a `_pass` counter stored on the worker's own copy, which
  survives between epochs *precisely because* persistent workers are long-lived,
  and so advances the stream even when `set_epoch` cannot reach it. The dataset
  is now correct under either loader configuration rather than depending on the
  caller to configure it correctly. Both paths re-verified.

### E2 — the overnight run *(in progress)*

```
train.py --stream --stream-length 16000 --crop 512 --batch-size 8 --epochs 12 \
  --lr 5e-4 --workers 4 --val-limit 100 --keep-epochs \
  --resume weights/driftsense_v3_last.pt --out weights/driftsense_v4.pt
```

Resumed at epoch 4 with the optimizer and one-cycle position restored (LR picked
up at 4.35e-4, continuing the anneal rather than restarting), now with genuinely
fresh scenes each epoch. Started 22:43, ~44 min/epoch, 8 epochs → finishes about
04:35. Checkpoints land every epoch, so it can be stopped at any boundary.

**Worker count is capped by memory, not by cores.** Generation is the
bottleneck — the 4 workers sit at ~93% CPU while the main process idles at 18%
waiting for data, with 5 of 10 cores free — so more workers would buy close to
linear throughput. They are not affordable: each worker holds a 10 000² canvas
and runs ~550 MB RSS, and the machine has 3.0 GB free with swap already at
5.2 GB of 6 GB. Three more workers would push it into swap-thrash and put the
run at risk of an OOM kill, which costs far more than the ~2 h the extra
throughput would save. Left at 4.

`--keep-epochs` is new (a four-line addition to `train.py`, off by default): it
writes `driftsense_v4_e{N}.pt` every epoch. The in-loop metric runs single-view
on 100 scenes and is far too coarse to separate epochs that differ by two or
three samples — it tied epochs 1 and 3 exactly. Keeping every epoch lets the
final choice be made afterwards on a split that can actually resolve it.

---

## Decision protocol — fixed in advance

Written before the numbers exist, so the rule is not chosen to fit the result.

The binding problem is that `data/val` is too small to select on. At n = 300 and
p ≈ 0.95 the standard error is ±1.3 points; the effects being chased are ~0.5.
Selecting a checkpoint on a metric that cannot see the effect is picking noise,
and noise does not transfer to test. So:

1. **Shortlist** 3–4 epochs from the in-loop history (free, already computed).
2. **Separate them on 1 200 freshly generated scenes**, single-view decode, via
   `scripts/stream_eval.py`. Generated in dataloader workers and consumed
   immediately — no disk, respecting the 20 GB limit — seeded in a namespace
   disjoint from training and from every on-disk split, and *paired*: the scene
   set is fixed by `--num`, independent of worker count (verified). 1 200 scenes
   cuts the standard error to ±0.6 points.
3. **Confirm the leader** against the shipped model on the full 300-scene
   `data/val` with the real 8-way TTA decode, compared by paired bootstrap
   (`scripts/compare_checkpoints.py`).
4. **Promote only on a consistent win**: ahead on acc@5px *and* not worse on
   median / acc@1px / acc@2px. A single-metric lead inside the noise is not
   enough to justify replacing a verified submission.
5. **Then, once**, run `data/test data/test_medium data/test_severe` — reported
   whatever they say, promotion already decided, so the test splits stay a
   measurement rather than a selection.
6. If nothing clears the bar, restore `weights/driftsense_ship_backup.pt` and
   the submission is unchanged. **That is an acceptable outcome**, and given the
   effect sizes it is a likely one.

---

## Outcome — the fix paid off, decisively

The run completed all 12 epochs. Loss fell 0.410 → 0.308 across the eight
fresh-data epochs, and the in-loop subset went 0.950 / 0.940 / 0.970 / 0.980 /
0.980 / 0.980 / 0.980 / 0.970 — against a ceiling of 0.94 for all four of v3's
repeated-pool epochs.

### Step 2 — 1 000 freshly generated scenes, single-view decode

This is the comparison the night was built around, and it is unambiguous.

| checkpoint | median | acc@1px | acc@2px | **acc@5px** | Δ vs shipped | 95% CI | P(worse) |
| --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| **v4 epoch 11** | 0.61 px | 0.658 | 0.860 | **0.962** | **+2.9 pts** | [+1.8, +4.0] | 0.000 |
| v4 epoch 9 | 0.61 px | 0.657 | 0.858 | 0.961 | +2.8 pts | [+1.7, +3.9] | 0.000 |
| v4 epoch 7 | 0.61 px | 0.657 | 0.856 | 0.959 | +2.6 pts | [+1.5, +3.8] | 0.000 |
| shipped | 0.64 px | 0.639 | 0.837 | 0.933 | — | — | — |

Every confidence interval excludes zero by a wide margin, and the gain shows up
at *every* tolerance — acc@1px, acc@2px, acc@5px and median all improve
together. That matters: sub-pixel precision and wrong-repeat rejection are
different failure modes, and an intervention that only moved one of them would
be suspicious.

The three late epochs are separated by 1–3 samples in 1 000, i.e. not
distinguishable. Epoch 11 is chosen as both the top scorer and the principled
pick — it is the fully annealed end of the one-cycle schedule.

**The n=100 subset would have picked the wrong one.** `driftsense_v4.pt`, saved
by the in-loop selector, is epoch 7 — the *worst* of the three on the real
measurement. Keeping every epoch and deciding afterwards was worth a fifth of
the total gain on its own.

### Step 3 — full 300-scene val, real 8-way TTA decode

| checkpoint | median | acc@1px | acc@2px | acc@5px | Δ vs shipped | 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| **v4 epoch 11** | **0.65 px** | **0.653** | **0.843** | **0.960** | +1.3 pts | [−0.3, +3.3] |
| v3 epoch 3 | 0.64 px | 0.653 | 0.840 | 0.953 | +0.7 pts | [−1.0, +2.7] |
| shipped | 0.67 px | 0.647 | 0.840 | 0.947 | — | — |

Exactly as the protocol predicted, 300 scenes cannot establish this on their own
— the interval still spans zero. It confirms *direction* on all four metrics,
and the 1 000-scene test supplies the significance. This is why the streamed
evaluation existed.

### Promotion

All four criteria met: ahead on acc@5px, and better on median, acc@1px and
acc@2px simultaneously, with the effect established at P(worse) = 0.000 on the
larger set. **Epoch 11 promoted to `weights/driftsense.pt`**; `infer.py`
smoke-tested end to end against it. The previous model remains byte-for-byte at
`weights/driftsense_ship_backup.pt`.

### Step 5 — the test splits, run once

| split | acc@5px before | **after** | acc@2px before | **after** | failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| `test` | 0.953 | **0.970** | 0.863 | **0.877** | 14 → **9** |
| `test_medium` | 1.000 | **1.000** | 1.000 | **1.000** | 0 → **0** |
| `test_severe` | 0.945 | **0.960** | 0.670 | **0.685** | 11 → **8** |
| **all 700** | 0.964 | **0.976** | — | — | 25 → **17** |

Median holds at 0.64 px; the wrong-repeat rate falls **3.4% → 2.4%**. A third of
the residual error is gone.

Note the test gain (+1.7 pts) is smaller than the 1 000-scene single-view gain
(+2.9 pts). That is expected, not a discrepancy: TTA was already repairing some
of what the better model now gets right on its own, so the two improvements
partly overlap. The same effect shows in TTA's value shrinking — it is now worth
+0.7 points on `test`, down from what it was worth to phase 1.

### Everything updated

- `README.md` — headline result, all three results tables, wrong-repeat rates,
  the failure-analysis progression, the TTA and confidence tables, the residual
  failure counts, and a new **Phase 3** training section documenting the
  streaming run, the `persistent_workers` bug, and the selection method.
- `results/examples.png` — regenerated. The old row-3 example (id 35) is now
  wrong under *both* decodes, so the figure uses id 153, which is the one
  remaining `test` case where TTA breaks a correct single-view prediction. Its
  confidence is 0.755, so — unlike the phase-2 example — the reject path does
  **not** catch it. Written up as such rather than papered over.
- `results/results.json`, `results/cmp_*`, `results/stream/*` — all measurements.
- Verified end to end from outside the repo: all 7 deliverables present,
  `infer.py` working in positional, `--json` and `--no-tta` forms, and
  `generate_dataset.py` → `infer.py` round-tripping to 0.66 px and 3.67 px on
  two freshly generated pairs.

### What I would do next

1. **Train longer on streamed data.** This is now the clear lever and it is not
   exhausted — held-out accuracy was still rising when the 12-epoch one-cycle
   ran out, with no overfitting signature anywhere in the run. A 24-epoch
   schedule is the obvious next experiment and needs no new ideas.
2. **Then** the full-frame fine-tune (your original item 2), which remains
   untested and is aimed squarely at the wrong-repeat residue.
3. A wider model (64 → 96) now makes sense in a way it did not before: the data
   constraint that would have made extra capacity overfit is genuinely gone.

Not worth revisiting: TTA aggregation (unchanged conclusion — better proposals
beat better arbitration, which is exactly what phase 3 demonstrated), and BN
recalibration (measured, rejected, explained above).


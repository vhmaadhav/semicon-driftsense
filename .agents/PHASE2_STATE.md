# Phase 2 — current state, and how to not waste a session

**Read this file first.** It is the single source of truth for where Phase 2
stands. Last updated 2026-08-29. Deadline: **3 September 2026**, code frozen,
no resubmission.

If you only read one section, read *Do not re-derive these* and *Dead ends
already paid for*.

---

## 0. The three things Phase 2 adds

Everything in Phase 1 still applies — same 1000×1000 sizes, same top-left
origin, same nearest-to-centre tie rule, same Python-only submission. Exactly
three assumptions were removed:

1. **Zoom is unknown**, uniform in `[8, 12]`, redrawn per pair.
2. **Rotation is unknown**, ±5°, CCW positive, and must be *reported*.
3. **The reference may be absent** — ~20% of pairs contain no true instance.

Output contract, one row per `pair_id`, every id exactly once:
`pair_id, x, y, theta, scale, found, score`. When `found=0`, write `0` in all
pose columns. A missing row scores zero.

Entry point, exactly: `python register.py --input pairs.csv --output predictions.csv`

Reference machine: 4-core x86, 8 GB, **no GPU, no network**, Python 3.11.
Weights ship inside the ZIP. Median ≤5 s/pair, hard timeout 20 s.

## 1. Scoring, and the two subtleties that cost us points

| Component | Pts | Notes |
| --- | --: | --- |
| Localisation | 40 | Sets A+B present pairs, tiered 1/2/3/5 px, weighted **0.45·A + 0.55·B** |
| Pose | 20 | scale 10 + rotation 10, **only where localisation already scored** |
| Rejection | 15 | F1 on the `found` flag over all 180 grayscale pairs |
| Calibration | 10 | AUC of our `score` column vs per-pair correctness |
| Efficiency | 5 | relative quartile ranking on **median** wall-clock per pair |
| Generator, citations, failure analysis | 10 | carried forward, re-judged |
| **Bonus** | +10 | +6 for Set D, +4 if rejection F1 ≥ 0.90 |

**The Set D bonus is out of reach — treat it as unavailable.** The slide reads
"+6 if Set D credit >= 0.40 with Sets A-C >= 0.50", which we clear easily (Set D
0.938, A-C 0.814). But the gate confirmed with the organisers is that **Sets
A-C must be above 95**, not 0.50, and we are at 72.55 of the 95 measurable. The
briefing call's phrasing — it "only unlocks if your scores are extremely good on
grayscale" — matches the stricter reading. So Set D work has no expected value
at our current standing, and the only reachable bonus is the **+4 at rejection
F1 >= 0.90**. Plan against 100, not 110.

Credit tiers — localisation: 1.00 ≤1px, 0.80 ≤2px, 0.60 ≤3px, 0.40 ≤5px, else 0.
Scale: 1.00 ≤1%, 0.60 ≤2%, 0.30 ≤5%. Rotation: 1.00 ≤0.25°, 0.60 ≤0.5°, 0.30 ≤1.0°.

**Subtlety 1 — a declined pair loses its localisation and pose credit too.**
`register.py` writes `x=y=theta=scale=0` when it reports `found=0`, so wrongly
rejecting a *present* pair forfeits its 40-point localisation credit and its
20-point pose credit on top of hurting F1. Any threshold tuned on F1 alone sits
too high. Tune against the total: `scripts/optimize_threshold.py`.

**Subtlety 2 — the rejection F1 positive class is genuinely ambiguous.**
The scoring slide says "never rejecting scores zero here", which is only true
with *reject* as the positive class (an always-found system scores exactly
0.000 that way on 140 present / 40 absent). The briefing call instead said
"F1 on the found flag", and another slide said only "cannot score well" — both
read as *present* positive, where the same system scores 0.875. **The two
readings differ by ~1.7 points.** Report both; plan against reject-positive
because that is the reading that hurts if we guess wrong. Our tooling prints
both and never quotes one as "the" F1.

## 2. Where we actually stand

Measured on **`data/ext_p2/`** — 2500 pairs with full ground truth, from the
Drive dataset `driftsense_phase2_synthetic_v1`.

**It is not an independent generator.** Its manifest is a strict superset of
ours (all 44 of our columns plus 13 Phase-2 bookkeeping fields) and the twelve
architecture presets match exactly, so it is *our* generator in a Phase-2
harness that adds the A/B/C/D split, a four-level severity ladder, polygon
scaling and content hashes. An earlier note called the score drop a "domain
gap"; that was wrong.

### Final, full-set numbers (2500 pairs, shipped configuration)

| Component | Credit | Points |
| --- | --- | ---: |
| Localisation (40) — set A 0.942, set B 0.730 | 0.8138 | 32.55 |
| Pose — scale (10) | 0.9039 | 9.04 |
| Pose — rotation (10) | 0.9054 | 9.05 |
| Rejection (15), reject-positive F1 | 0.8084 | 12.13 |
| Calibration (10), AUC | 0.9777 | 9.78 |
| **Total of the 95 measurable** | | **72.55** |

Localisation and pose are credited **zero on declined pairs**, as the grader
will see them. Runtime **3.35 s median** (p90 4.16, max 4.42) single-process at
4 threads on an idle machine, against a 5 s median target and a 20 s hard
timeout. Set D scores 0.938 untouched but the **+6 is not reachable** (it
requires Sets A-C above 95; see §1), and rejection F1 is below 0.90 so the
**+4** is not earned either. Assume **zero bonus**.

### Beware the subsample

The 500-pair stride-5 subsample used during development read **~1.5 points
optimistic** against the full 2500 (74.3 vs 72.8 on the same configuration).
Use stride 5 to rank variants against each other; quote only full-set numbers.

## 3. Do not re-derive these

* **Runtime is fine.** 2.92 s median at 4 threads idle. Any figure near 20 s
  came from running the parallel eval harness — this machine is
  memory-bandwidth bound and tops out near **0.42 pairs/s total** regardless of
  the worker/thread split, so per-pair timings inside a parallel run are
  meaningless. Measure timing single-process, 4 threads, idle, or not at all.
* **The template quantisation defect is fixed.** `make_template` used to render
  at `round(1000/m)`, making only **43 magnifications realizable across [8,12]**
  in steps 0.81–1.22% wide — as wide as the entire ≤1% full-credit tier. The
  residual sub-integer scale is now folded into the affine already paid for to
  apply rotation. Realisation error 0.26% → 0.012% median, nominal 10× still
  bit-identical.
* **`TM_CCOEFF_NORMED` is biased across template sizes.** Fewer pixels correlate
  better by chance, which pulled the scale estimate high. `polish_pose` now pins
  the canvas across its sweep. Together with the above: scale credit 0.824 →
  0.906, median *signed* error +0.65% → +0.037%.
* **The residual >5 px failures are acquisition-severity failures, not pose
  failures.** Effect sizes (`scripts/diagnose_failures.py`): drift jitter 1.23,
  salt-pepper 1.21, charging 1.20, speckle 1.18, detector noise 1.15, shear
  1.13 — against **|rotation| 0.15** and **polygon scaling −0.12**. Set B fails
  at 17.1% vs Set A's 3.4%.
* **…and the reason is that severity 4 is outside the training distribution.**
  Measured against the Set B manifests, our training sampler's ceiling sat
  below the level-4 ceiling on six of seven knobs:

  | knob | trained max | Set B sev-4 max |
  | --- | ---: | ---: |
  | beam spot (defocus) nm | 8.0 | **10.00** |
  | detector noise sigma | 12.0 | **13.00** |
  | charging streak prob | 3.5 | **4.00** |
  | shear px | 4.5 | **4.80** |
  | drift jitter px | 2.0 | **2.10** |
  | astigmatism ratio | 1.35 | **1.42** |
  | speckle sigma | 0.30 | 0.32 |

  Failure rate is 13.3% at severity 4 against 5.0% at severity 1. This is a
  *data coverage* defect, not a capacity or architecture defect, and it is the
  single largest remaining item. `driftsense/generate.py` now carries a
  `SEVERITY_LADDER` with these measured endpoints, widened 12%, driven by one
  latent severity so the knobs move together (independent draws almost never
  produce the all-bad corner that actually breaks matching).
* **K=5 returns identical results to K=3 — but not because hypotheses are
  "exhausted".** The coarse sweep samples [8,12] at 17 points, a 2.5% step,
  while the correlation peak it hunts is 1–2% wide, so it can step over the
  true peak entirely and that basin never becomes a local maximum. Extra
  candidate slots cannot hold a peak the grid never sampled. Raising the
  sample count does *not* fix it either (41 points measured −0.006 pts), so
  the binding constraint is that the coarse *score* is noise-dominated, not
  the resolution. See §3b.
* **Set D needs no work.** 0.976 credit as-is.

## 3b. Hypothesis verification: what the scores can and cannot reach

Two independent investigations landed on this in parallel — `scripts/verify_scores.py`
here, and PR #3 (`dev/phase2-robust-verification`, now merged). They used
different harnesses and different data and agree on the important parts.

### The ceiling: verification reaches 24% of Set B failures

Measured on 270 Set B pairs from `data/ext_p2/`: of 90 pairs that currently
fail at >5 px, only **22 (24%)** had a correct hypothesis *generated* and then
not selected. The other **76% never had a right answer among the candidates**,
so no similarity measure can rescue them — the coarse *search* is what failed,
not the ranking. PR #3's oracle says the same thing in its own units: a
recoverable gap of 2.68 pp (dev) and 2.41 pp (confirmation).

**Consequence:** verification work is worth a few tenths of a point, not units.
Budget it accordingly.

### Which score, measured as an independent selector

Each score replaces native ZNCC as the hypothesis picker. "Breaks" counts
pairs that currently succeed and would stop succeeding — a score that rescues
14 and breaks 13 is a loss, and only reporting rescues would hide that.

| score | recovers | breaks | net |
| --- | ---: | ---: | ---: |
| `zncc` (incumbent) | 12/22 | 5/180 | +7 |
| `zncc_rank` | 14/22 | **13/180** | +1 |
| `zncc_dog` | **14/22** | **4/180** | **+10** |
| `zncc_grad` | 13/22 | 7/180 | +6 |
| `zncc_clip` | 12/22 | 5/180 | +7 |

**Rank/census is the trap.** It rescues the most and is still the worst
practical choice, because it discards too much on clean pairs. PR #3 reached
the same verdict from a different harness (net −6 dev, −4 confirmation) and
also rejected it. Do not revisit it on the strength of the ECCV citation
alone; two measurements say no.

### The safer construction: consensus, from PR #3

Rather than swapping the selector, PR #3 keeps native ZNCC and overrides it
only when **rank and band both agree** on the same different hypothesis
(`verification="consensus"`, default `"zncc"`). Measured there: +2 rescued / 0
broken (dev) and +1 / 0 (confirmation), about **+0.31 to +0.37 points**. Zero
broken successes in both splits is the property that makes it worth having.

**Two caveats on that report's numbers, which are not the code's fault:**

1. Its branch was cut from `bd51fee`, **before PR #2**, so its baseline lacks
   the continuous-scale template, canvas-pinned polish, pose clamping and
   `min(score, zncc)` confidence — the +1.94 points already banked.
2. Its Set B is a **local generator proxy** scoring credit 0.888 on 149 dev /
   83 confirmation present pairs. The real external Set B scores **0.723 on
   875 present pairs**. The report says outright that an external shard was
   unavailable. We have one. **`verification="consensus"` has not yet been
   measured on `data/ext_p2/` — do that before enabling it by default.**

### Band-pass moved one stage earlier

Because verification only reaches 24% of failures, `_band()` (difference of
Gaussians) is also applied inside `pose_candidates`, where it can affect the
other 76% by changing which candidates get generated at all. Set B's
degradations sit at both spectral ends — charging low-frequency, shot and
impulse noise high-frequency — with the layout structure between them, so a
band keeps what the search needs and discards both noise families. Switchable
via `band=` / `--no-band`; the A/B against the incumbent is the open item.

## 4. Dead ends already paid for

Do not spend a session re-attempting these. Each was measured, not guessed.

| Attempt | Result | Why |
| --- | --- | --- |
| `refit_xy` — re-snap x,y with the polished template | **+0.04 pts** | Set A up, Set B down; a wash. Left off (default). |
| Clamping pose to the disclosed box | +0.06 pts | Real but tiny; shipped because it is free and provably correct (9/400 predictions fell outside `[8,12]`). |
| K=5 pose hypotheses | 0.000 | Identical to K=3. |
| Rotation-aware coarse scale sweep | *not attempted, and should not be* | |rotation| separates failures at d=0.15. It is not what is failing. |
| Modelling polygon scaling harder | *not attempted, and should not be* | d=−0.12. It is modelled now anyway, but it is not the Set B problem. |
| ECC sub-pixel pose refinement | superseded | The old plan's premise ("correlation-vs-scale is flat") was right but misdiagnosed — the objective was a 43-step staircase. Fixing the staircase got the points without ECC. |
| Fine-tuning on corrected labels | 0.000 over 15 epochs | Recorded in the prior `IMPROVING.md`; val acc@5 stayed at 0.817. Superseded — that run predated the label fix *and* the severity ladder. |
| Gated dihedral TTA (`tta_gate`) | **+0.11 pts, rejected** | Measured, and not worth it. See below. |

### Why gated TTA was rejected

Implemented and measured on 500 pairs: localisation credit 0.8440 -> 0.8471,
Set B 0.7566 -> 0.7623, **+0.11 points total**. Cost: median runtime +35%
(23.4s -> 31.6s contended) and p90 **3.6x** (32.6s -> 117.8s). Efficiency is
scored on *median* wall-clock and there is a 20 s hard timeout that zeroes a
pair, so that tail is a real risk for a tenth of a point.

The gate also fired on **42%** of pairs, not the 29% predicted. The prediction
was measured on *present* pairs only, and nearly every *absent* pair also
scores below 0.60 — so the gate spent its budget on pairs that by construction
can never benefit. Of 212 firings only 31 verified better under native ZNCC;
181 were wasted 8x forward passes.

Tightening the gate cannot rescue it: even a perfect gate that fired only on
the 31 pairs that helped would still cap at +0.11. The code remains in the
scratch tree behind `tta_gate=0.0` (off) if anyone wants to re-check.

**Lesson worth keeping:** when sizing a gate that keys on a confidence score,
compute the firing rate over *all* pairs the gate will see, absent ones
included — not over the subpopulation you hope to fix.

## 5. Landed this session

All in the working tree on branch `phase2-unknown-pose` (**not yet committed**).

* `driftsense/matching.py` — continuous-scale `make_template` (+ `canvas=` pin);
  `polish_pose` polishes *both* axes with a pinned canvas; `locate_phase2`
  adopts the polished scale and clamps the reported pose to `[8,12]` / `±5°`;
  optional `refit_xy` (off) and `tta_gate` (under test).
* Generator — `polygon_scale_fraction`, a *multiplicative* CD change with pitch
  held fixed, threaded through `pipeline.py` → `zones.py` → `dram.py`/`finfet.py`,
  exposed as `--polygon-scale-range` and defaulted on under `--phase2`. It is
  **opt-in and drawn from the pose stream**, so no draw is made when disabled
  and the Phase 1 splits still reproduce byte-for-byte.
* New tooling: `scripts/eval_ext.py`, `compare_ext.py`, `optimize_threshold.py`,
  `diagnose_failures.py`, `tune_rejection.py`, `profile_pair.py`,
  `failure_analysis.py`.
* `CITATIONS.md` §6/§10, `README.md` method + Set B coverage, `IMPROVING.md`
  rewritten against the external measurement.
* All 118 tests pass.

## 6. Ranked next work

Ordered by measured expected value, not by how interesting the idea is.

### 1. Rejection → F1 ≥ 0.90. ~2.9 points, plus a 4-point bonus.

Still the largest and most certain item. Rejection is 12.13/15 and calibration
9.78/10, both decided by one scalar. The shipped scalar is `min(score, zncc)` —
a hand-picked rule over two of the four signals the pipeline computes.
`peak_ratio` (how contested the decision was) and `pose_peak` (whether any pose
explains the frame) are computed and discarded. `scripts/fit_rejector.py` fits
a 5-parameter logistic over all four, on *training* shards, thresholded on a
training fold, reported out-of-sample.

### 2. Measure `verification="consensus"` on `data/ext_p2/`.

PR #3 shipped it as opt-in and could not measure it on external data. This is
the cheapest open question in the repo: one eval run answers whether its
+0.31–0.37 holds on the real Set B, or whether it was an artefact of a
149-pair local proxy. Until then, leave the default at `"zncc"`.

### 3. Band-passed coarse sweep — A/B in flight.

Targets the 76% of Set B failures that verification cannot reach, by changing
which candidates are generated. Compare `--no-band` against the default.

### 4. Training, scoped to the half it can reach.

The true-pose oracle bounds this: Set B's ceiling with a *perfect* pose is
**87.1%**, not 99%. So ~52% of failures are pose-side and only **48% are the
network's own error** — that is all training can address. A 4-epoch probe moved
Set B by 0.000, but it was mis-configured (see §6e) and its one-cycle schedule
never left warmup. One properly configured run is justified; an open-ended
search is not. **Promote only if Set B credit improves on the external 2500-pair
test** — training loss is not a proxy: the last run's loss fell 0.404 → 0.32
while Set B credit moved exactly zero.

### 5. Not worth doing yet

* **Lattice fallback.** The failures track acquisition severity (drift jitter
  d=1.23, salt-pepper 1.21, charging 1.20), not periodicity. Attack the noise
  first; the same pairs are reachable more directly.
* **Rank/census as a selector.** Rejected twice, by two harnesses. §3b.
* **DINO / LoFTR / RoMa.** Beyond the rules question, the current 0.46M network
  is already 86% of a 3.35 s pair against a 5 s median budget, paid three times
  over for three hypotheses. There is no room.

## 6b. Training on the Drive shards — verified recipe

This path is **checked, not assumed**. `data/ext_train/s000` was downloaded,
extracted and loaded through `DriftSenseDataset`: 500 pairs, tensors
`template (1,100,100)`, `search (1,512,512)`, `heat (1,104,104)`, plus `offset`,
`peak`, `found`, `gt`. The train shards are already in our pool format —
100×100 template references, 1000×1000 search, and all 44 of our manifest
columns — so a shard directory *is* a pool shard. No conversion needed.

```bash
# one shard -> data/ext_train/sNNN/{manifest.csv,reference/,search/}
tar xf <shard>.tar -C data/ext_train/s000

./venv/bin/python train.py     --train-dirs data/ext_train/s000 data/ext_train/s001 ...     --val-dir data/val_p2 --phase2     --resume weights/driftsense.pt --finetune --lr 1e-4     --out weights/driftsense_p3.pt
```

**Listing the Drive folder.** The normal folder URL server-renders only the
first **50** entries, which is why an early look appeared to show no `train_B`.
Use the embedded list view instead — it returns everything in one plain page:

```bash
curl -sL "https://drive.google.com/embeddedfolderview?id=1w5BoAvPIXQJH1gWfQQ8-ADsUSQ3J99tj#list"
# parse: id="entry-<FILE_ID>" ... <div class="flip-entry-title">NAME</div>
```

As of 2026-08-28 that returns **586 entries** across **two independent
generation runs**, distinguished by the dataset id in the filename:

| dataset id | train_A | train_B | train_C | train_D | test |
| --- | --: | --: | --: | --: | --- |
| `a06d9df298761144a64c` | 32 | 32 | 18 | 9 | A2 B2 C1 D1 |
| `023624106c02db2986a9` | 69 | 62 | 34 | 12 | A2 B3 D1 |

**Train on `023624106c02db2986a9` and keep `a06d9df298761144a64c` for
evaluation.** Our `data/ext_p2/` test shards are all from `a06d9df…`, so
training on the other run makes train/test separation structural — a different
seed namespace entirely — rather than relying on the data card's claim that
splits within one dataset are disjoint. It also turns the eval into a
cross-run generalisation check for free.

**Which sets are worth the disk.** The value is in **B and C**, not A. Training on
`train_A` alone will not fix Set B and will probably make it worse — `train_A`
is *milder* than the data the model already trains on:

| knob | our current training (p50/p95) | Drive `train_A` / Set A (min..max) |
| --- | --- | --- |
| drift jitter px | 0.94 / 1.89 | 0.15 .. 0.80 |
| charging streak prob | 1.00 / 3.50 | 0.00 .. 0.40 |
| beam spot nm | 5.74 / 7.76 | 3.80 .. 6.50 |

Set A already scores credit 0.95. The 6 remaining localisation points are all
in **Set B**, and the 25 rejection+calibration points need **Set C** (absent
pairs). So the generation run should be weighted to **B and C**, not A. The
data card's own plan (A 15750 / B 15750 / C 9000 / D 4500) is a reasonable
ratio — roughly 35% B and 20% C.

**Do not train on the `test_*` shards.** They are the only held-out measurement
we have, and mixing test data into training in either direction is an explicit
disqualifier.

## 6c. What "more powerful model" does and does not mean here

* **Worth doing:** train on the severity ladder. The failure is that the model
  has never seen severity 4, not that it lacks capacity. This is a data-coverage
  fix and it is the largest remaining item.
* **Not worth doing: reinforcement learning.** RL is for sparse-reward problems
  with no gradient to the right answer. Here the generator supplies exact
  `x, y, theta, scale` for every pair for free — dense supervision. RL would
  discard that and estimate the same signal with far more variance. There is no
  version of this problem where it is the right tool.
* **Capacity is not obviously the constraint, and it is not free.** The network
  is 0.46M params and already 84% of a 2.92 s/pair budget against a 5 s median
  target, paid three times over for the three pose hypotheses. Any width
  increase multiplies by three. Measure the severity-trained model first; only
  reach for capacity if it saturates.
* **Retraining has failed before.** A fine-tune scored 0.000 over 15 epochs
  (val acc@5 stuck at 0.817). That run predated both the 0.45 px label fix and
  the severity ladder, so it is worth one more attempt — but budget it as one
  attempt with a clear stopping rule, not an open-ended search.

## 6d. The overnight retrain (2026-08-28 22:12)

* **GPU:** RTX 4060 Laptop, 8 GB. Use `venv-train/bin/python` (torch
  2.13.0+cu130) for training — the plain `venv` is a **CPU-only** build and
  `torch.cuda.is_available()` is False there. Evaluation deliberately stays on
  the CPU venv, because the grading machine has no GPU.
* **Throughput:** ~74 img/s at batch 16 with AMP, 1.45 GB VRAM. 15k samples =
  ~3.4 min/epoch. CPU training would have been ~40x slower and pointless.
* **Pool:** 39 shards, ~19.5k pairs — 21 `train_B`, 10 `train_C`, 8 `train_A`.
  Deliberately B-heavy: Set A is already at credit 0.95, so the A shards are
  ballast against regressing nominal, not the point.
* **Separation:** training shards are from generation run
  `023624106c02db2986a9`; the eval set `data/ext_p2/` is entirely from
  `a06d9df298761144a64c`. One 500-pair `train_A_0000` shard from the eval run
  did get included (~2.5% of the pool) — it is a *train* shard, disjoint from
  the *test* seed namespace, but the separation is 97.5% structural rather
  than 100%.
* **Config:** `--resume weights/driftsense.pt --finetune --lr 1e-4`, one-cycle,
  30 epochs x 15k, `--refresh-pool` so shards still downloading are picked up.

## 6e. Operational traps that cost real time here

Each of these wasted at least one run. They are environment facts, not
insights, which is exactly why they are easy to rediscover the hard way.

**`train.py` needs `--phase2`, or validation is meaningless.** `evaluate()`
takes `phase2=args.phase2`. Without the flag it scores posed scenes (mag 8–12,
rot ±5°) with a *nominal* 10×/0° validator and reads ~chance — 429 px median,
acc@5 0.28 — from epoch 0, before training could have changed anything. A run
was stopped on the strength of that number, which was a false alarm.

**`--val-dir` must hold full 1000 px references.** Pointing it at a *train*
shard gives 100 px template references, and the validator builds its template
from a full reference: val acc@5 reads exactly **0.000**. Use `data/val_p2`
(reference_px 1000, and it carries absent pairs). `data/ext_train/*` and
`data/ext_holdout/*` are training-format shards and are **not** valid
validation sets.

**Background jobs need `setsid`, not `nohup`.** `nohup cmd &` from a tool call
still dies when the session ends. Two multi-hour jobs were lost that way. Use:

```bash
setsid nohup ./script.sh > log 2>&1 < /dev/null &
```

**Long extractions must checkpoint *and* shuffle.** `fit_rejector.py` wrote its
cache only at the end, so a 90-minute run interrupted at 400/2340 lost
everything. Worse, tasks were built shard-by-shard, so the partial cache held
400 present pairs and **zero absent** — useless for fitting a present/absent
decision. Both fixed: it writes every 200 pairs and shuffles first, so any
prefix is a representative sample.

**This laptop runs at 100 °C under any sustained load.** 97 °C on CPU-only
evaluation, 100 °C with GPU training added. It thermally throttles rather than
failing, but it means the machine is the constraint on parallelism: four
concurrent jobs roughly doubled the rejector's ETA. Prefer sequencing the
measurements you actually need over running everything at once.

**Timings inside a parallel run are fiction.** The box is memory-bandwidth
bound at ~0.42 pairs/s total regardless of the worker/thread split. Runtime
claims come from `scripts/profile_pair.py`, single process, 4 threads, idle.

## 7. Data, and the leakage rule

* `data/ext_p2/` — 2500-pair external test set (A 875, B 875, C 500, D 250),
  full ground truth, `reference_mode=full` at 1000 px, from the Drive dataset
  `driftsense_phase2_synthetic_v1`. Its data card states **"no organizer
  data"**. Used for **validation only**. Do not train on it.
* The organizers will publish **~20 sample pairs with full ground truth**. Those
  are an I/O-contract check and a validation fold — mixing organizer test data
  into training in *either* direction is an explicit disqualifier.
* `data/pool_p2/`, `data/val_p2/` — our own generator's training pool and
  validation split.

## 8. Reproducing any number here

```bash
# external rubric score (start with --stride 5 for a ~20 min read)
./venv/bin/python scripts/eval_ext.py data/ext_p2/test_{A_0000,A_0001,B_0000,B_0001,C_0000,D_0000} \
    --jobs 10 --threads 1 --stride 5 --out .agents/run.csv

./venv/bin/python scripts/compare_ext.py base=.agents/ext_base.csv new=.agents/run.csv
./venv/bin/python scripts/optimize_threshold.py .agents/run.csv
./venv/bin/python scripts/diagnose_failures.py .agents/run.csv

# runtime — single process, 4 threads, idle machine, or the number is fiction
./venv/bin/python scripts/profile_pair.py data/ext_p2/test_A_0000 --n 12 --threads 4
```

The interpreter is `./venv/bin/python` (Python 3.11 via uv). The system
`python3` is 3.14 and has no torch.

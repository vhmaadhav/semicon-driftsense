# Phase 2 — current state, and how to not waste a session

**Read this file first.** It is the single source of truth for where Phase 2
stands. Last updated 2026-08-28. Deadline: **3 September 2026**, code frozen,
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
| **Bonus** | +10 | +6 if Set D credit ≥0.40 with A–C ≥0.50; +4 if rejection F1 ≥0.90 |

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
timeout. Set D scores 0.938 untouched, clearing the **+6** bonus gate;
rejection F1 is below 0.90 so the **+4** bonus is not earned.

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
* **Pose hypotheses are exhausted at K=3.** K=5 returns identical results; the
  coarse sweep only produces ~3 local maxima.
* **Set D needs no work.** 0.976 credit as-is.

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

1. **Set B localisation — ~6 of the 40 points.** The entire localisation gap.
   The failures are acquisition-severity failures, so the fixes below attack
   the *verification* signal, not the pose search (which is solved) and not the
   network (retraining scored 0.000 last time).

   **(a) ~~Gated dihedral TTA~~ — measured at +0.11 pts and rejected on
   runtime. See §4.**

   **(b) Rank-transform verification — researched, not yet implemented.** The
   two strongest failure discriminators are impulse noise (salt-pepper,
   d=1.21) and speckle (d=1.18), and ZNCC is a least-squares statistic, so a
   handful of outlier pixels move it a lot. The standard fix is to correlate
   *non-parametric local transforms* instead of intensities: the rank transform
   replaces each pixel by the count of neighbours darker than it, so the
   statistic depends only on local intensity *ordering* and tolerates a
   substantial fraction of outliers, as well as being invariant to monotonic
   intensity change (charging, dose drift, gamma).

   - Zabih, R. and Woodfill, J. "Non-parametric Local Transforms for Computing
     Visual Correspondence", *ECCV* 1994 — the rank and census transforms.
   - Elboher, E. and Werman, M. "Asymmetric Correlation: A Noise Robust
     Similarity Measure for Template Matching", *IEEE TIP* 2013 — a template
     matching similarity invariant to affine illumination change and robust
     under extreme noise.

   Suggested shape: keep ZNCC for sub-pixel *placement* (rank transform
   quantises and would blunt the parabolic fit) and add rank correlation as a
   second *verification* statistic for choosing between pose hypotheses and for
   the confidence column. Cost is one 5×5 rank pass over the search frame
   (~24 shifted compares, well under 0.1 s), so it fits the budget. Measure it
   as its own variant — do not combine it with anything else in the same run.

   **(c) Charging is low-frequency.** Charging streaks are slowly-varying
   horizontal bands, so a high-pass / difference-of-Gaussians prefilter before
   verification should suppress them where a rank transform will not. Cheapest
   of the three to try; test after (b).
2. **Rejection to F1 ≥ 0.90 — ~2 points plus the +4 bonus.** `min(score, zncc)`
   is already the better statistic (the full-res ZNCC was being computed and
   discarded); worth ~+0.65 held-out. Needs to be wired into `register.py`.
3. **Document the confidence scale in the README.** The mentor asked for this
   explicitly on the call so graders can read how our `score` is formed.
4. **Ship `failure_analysis.pdf`** (max 2 pages) — generator exists
   (`scripts/failure_analysis.py`), regenerate from the final results CSV.

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

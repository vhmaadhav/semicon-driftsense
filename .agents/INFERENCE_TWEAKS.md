# Inference-level tweaks: side-by-side score log

**IMPORTANT (2026-08-31, issue #22 P0):** the absolute subtotals below were
produced by the pre-fix scorer that let declined present pairs keep
localisation/pose credit. Corrected re-score of the same CSVs:
band-on 75.78 → **75.50**, no-band 76.23 → **75.92**. The *paired deltas*
(+0.42 for the band flip) remain valid — both configs were scored under the
same inflated semantics. Rows are kept for history with that caveat.

**Protocol:** every tweak is measured on the **same seeded 200-pair draw**
(`eval_ext --sample 200 --seed 200`, shipped shards A/B/C) so the columns are
directly comparable, and on the **full 2,250** for the promotion decision
(paired bootstrap, gate Δ ≥ +0.35 vs the shipped decode). The 200-pair draw
alone carries ±1.2 points of sampling noise — the full-set paired number is
the decider; the 200-pair column is the quick read.

**Data provenance:** every score below is measured on `data/ext_p2` — the
`test` split of **our own** `driftsense_phase2_synthetic_v1` generator run
(byte-compatible shards, verified zero `pair_sha256` overlap with anything
trained on). It is testing-only hold-out data, not organizer data; the
organizers' 200 blind pairs are unreleased. "200-pair grade" means the
scoring *convention*, applied to our own held-out split.

Baseline rows come from the current shipped configuration
(`driftsense.pt`, threshold 0.2018, `verification=zncc`, band on).

| tweak | 200-pair subtotal (seed 200) | full 2,250 subtotal | paired Δ vs shipped | verdict |
|---|---:|---:|---:|---|
| shipped baseline | 75.29 | 75.78 | — | reference |
| 1. `--verification consensus` | 75.26 | 75.89 | +0.11 raw; paired loc Δ +0.02, 95% CI [−0.08, +0.14]; breaks 5 / rescues 6 | **not promoted** — under the +0.35 gate; the proxy's "broken=0" property did not hold at scale |
| 2a. soup e24+e30 (`weights/soup_e24_e30.pt`) | 73.60 | not run | ≈ −1.7 on the paired draw | **rejected** — worse; no gate run warranted |
| 2b. soup e12+e24+e30 (`weights/soup_e12_e24_e30.pt`) | 73.29 | not run | ≈ −2.0 on the paired draw | **rejected** — worse |
| 3. threshold 0.2018 → 0.1974 | (offline) | 75.52 vs 75.50 | +0.02 | **no change** — shipped operating point already optimal |
| 4. E1 template-embedding cache | 75.29 (identical by design) | identical coords | 0 | **kept as free instrumentation** — corrected harness (flag-based cache-off baseline runs 3 template encodes): p50 2.53s vs 2.56s optimized (1.00x). The template branch costs 1–2 ms of a 2–4 s pair. Issue #7's "recomputed 3×" premise measured: real but worthless |
| 5. **`band=False` (no-band)** | **75.70** | **76.23** | **+0.45 raw; paired loc Δ +0.167, CI [+0.014, +0.327], P(≤shipped)=1.7%; rescued 18 / broken 9** | **PROMOTED — new default** |

## Clocks (single process, 4 threads, CPU, interleaved A/B reps, n=20–30 pairs × 3)

| config | p50 | p90 |
|---|---:|---:|
| shipped (band on) | 2.513 s | 3.014 s |
| no-band (promoted) | 2.655 s | 3.218 s |
| E1 cache off vs on (flag-based, corrected harness) | 2.533 s vs 2.563 s | — (1.00x, wash) |
| E3 prune 0.5 vs exhaustive (audited 2026-09-02) | 1.47 s vs 1.44 s | — (0.98x, wash; **not shipped**) |
| rotation-aware scale ranking (issue #37, 2026-09-02) | stage `pose_candidates` 12.69 s vs 13.57 s / 20 pairs (**1.069x**) | pair p50 5.93 s vs 6.34 s — **inside noise**, see below |

Clock takeaway: the efficiency win of `band=False` is ~0 — `_band` on the
half-res probe is cheap. Its value is the **+0.45 accuracy points**, which is
why it promotes on the accuracy gate, not the clock gate. The remaining clock
lever is the coarse sweep itself (66.8% of pair time) — but E3 pruning of it
is now MEASURED as a wash on the 200-pair seeded draw (below), so that
lever is closed for the shipped default (the full-2,250 audit stays
pending, required only before enabling the gate).

## E3 pruning audit (2026-09-02) — 200-pair seeded-draw audit (full-2,250 equality audit stays pending)

Both legs, on the seeded 200-pair draw (A/B/C seed 200), light footprint
(2 workers x 2 threads for the A/B, single process for clocks):

* **Equality: PASSED bit-exactly.** margin 0.5 vs exhaustive: x, y, scale,
  theta, score 0.0e+00 delta on **200/200 pairs**; only `n_hyp` differs
  (15/200 pairs report fewer offered grid points — instrumentation, not the
  answer).
* **Clock: WASH.** single process, 4 torch threads, 20 pairs x 3 interleaved
  reps: exhaustive p50 1.44 s vs pruned 1.47 s (0.98x), means 1.41 vs 1.40
  (1.00x). The skipped coarse evaluations are noise against the network
  forward + refine + polish.
* **Verdict: default stays `E3_PRUNE_MARGIN = None` (exhaustive).** Perfect
  equality but nothing to gain; changing instrumentation semantics for a
  1.00x clock fails the change bar. The 200-pair seeded-draw audit is
  complete (the promised full-2,250 equality audit stays pending and is
  only required before enabling the gate) — see the AUDITED note in
  `driftsense/matching.py`.

## Free regrades and closures (2026-09-02, no new inference)

| check | result | verdict |
|---|---|---|
| early-exit CSVs rescored @ shipped 0.18 (corrected masking) | every gate breaks more than it rescues (0.55: −0.0121 paired loc, 1/9; 0.85: −0.0032, 1/3) | early-exit-off confirmed evidence-backed |
| Set B denoise 3×3 A/B (300 pairs seed 200, fresh inference) | ≤5px 90.7→91.7%, paired loc +0.0140, 12 rescued / 9 broken | under the +0.35 gate — not shipped |
| rejector round 2 (GBM + 7 engineered features, honest 4-fold CV) | best CV 75.72 vs shipped scalar 75.87; AUC 0.9882 vs 0.9876; in-sample oracle F1 1.000 = overfit | post-hoc rejection closed for nonlinear + engineered families too |
| hyp-4 / coarse-29/43 sweeps rescored under masked semantics | Set B credit: hyp-4 0.7726 vs baseline 0.7692 (+0.0034); coarse monotone worse | prior verdicts hold |
| threshold 0.18 vs sweep, held-out with FIXED threshold | full set 75.73 (fixed) vs 75.68 (per-fold fitted); 200-draw 74.87 vs 74.45 | shipped 0.18 confirmed optimal out-of-sample |
| `refit_xy` | prior measurement stands (+0.04, wash) | stays off |

Clock row for E3 added to the table above; the 200-pair seeded-draw E3
audit is recorded in `driftsense/matching.py` (AUDITED note) and commit
bde0c47 (full-2,250 equality audit: pending, not required while the
default is exhaustive).

Reference-sample validation (see `docs/VERIFIED_GROUND_TRUTH.md` §8):
register.py on the 20 reference sample pairs — format exact, overall present loc
credit 0.975 vs the naive reference baseline 0.800, theta sign +gt exact,
scale=z semantics exact, 4/4 absent declined, median 3.12 s. Our own data's
reference-baseline calibration: overall present 0.357 — inside the published
0.30–0.55 band (the "~0.92" claim was our solver's Set A accuracy mistaken
for the baseline's credit; refuted).

## Issue #37 — rotation-aware scale ranking, full 2,250 A/B (2026-09-02)

Closes the measurement issue #37 asked for, and the one the `RERANK_ROTATION`
gate names as its unblocking condition. The **fix arm is `rerank_rotation=True`**
(the re-rank as PR #35 wired it); the **base arm is the pre-#37 rot=0-only
path**, which is what the gate restores when False. **Arms differ in
`matching.py` only**; both decode the identical, already-generated `data/ext_p2` shards, so
the generator-side decoy change on this branch cannot touch the comparison.
Shipped config throughout (`band=False`, `hypotheses=3`, `coarse_scales=17`,
`verification=zncc`, threshold 0.18). Per-pair CSVs:
`.agents/ext_rot37_{base,fix}.csv` (decode) and
`.agents/cand_rot37_{base,fix}.csv` (candidate trace).

### Rubric: a wash

| component | base | fix | Δ |
|---|---:|---:|---:|
| Set A credit | 0.9737 | 0.9746 | **+0.0009** |
| Set B credit | 0.8151 | 0.8130 | **−0.0021** |
| localisation (40) | 35.459 | 35.430 | −0.029 |
| pose (20) | 17.984 | 17.983 | −0.001 |
| rejection F1 @0.18 | 0.9078 | 0.9096 | +0.0018 |
| calibration AUC | 0.9883 | 0.9882 | −0.0000 |
| **subtotal (85 measurable)** | **76.942** | **76.939** | **−0.003** |

Paired bootstrap on per-pair localisation credit (submission-masked):
set A **+0.00091** CI [+0.00000, +0.00274]; set B **−0.00206** CI [−0.00526,
+0.00000]; A+B −0.00057 CI [−0.00229, +0.00091]. 2179/2250 pairs are
byte-identical. Absent-pair (Set C) scores are unmoved: mean 0.0508 → 0.0507,
max 0.5073 both, 47 above the 0.18 gate in both arms.

**Far under the +0.35 promotion gate.** On the rubric alone this is a dud.

### Candidate generation: a real, one-directional improvement

The rubric is not the metric issue #37 is about. The defect is that the true
basin is *discarded before anything can evaluate it*, so the honest metric is
candidate-recall — measured by `scripts/trace_candidates.py`, which records
the shortlist `pose_candidates` offers, with no network and no selector.

| candidate-generation metric (1750 present pairs) | base | fix | Δ |
|---|---:|---:|---:|
| recall@1, near-GT basin | 80.23% | **84.06%** | **+3.83** |
| recall@2 | 91.03% | 92.00% | +0.97 |
| recall@3 | 92.17% | 92.57% | +0.40 |
| basin **never offered** | 137 | **130** | **−7** |
| set B recall@3 | 85.14% | 85.83% | +0.69 |
| tight (1% / 0.5°) recall@1 | 63.66% | 67.20% | +3.54 |

Measured on the coarse grid directly (2,250 cached scale×rotation score
matrices, tolerance one grid step in each axis): rotation-aware ranking
**gains the true basin on 8 pairs and loses it on 0**. The shortlist *set*
differs from the rot=0 rule on 140/1750 pairs, and every one of those changes
is neutral-or-better at the candidate level. This is the acceptance criterion
issue #37 actually states, and it is met.

### Where the points went instead: the selector, not the shortlist

Attribution of every >5 px wrong tile, from the trace:

| | base | fix |
|---|---:|---:|
| wrong tiles | 75 | 76 |
| ..basin **never offered** (candidate generation, #37) | **17** | **13** |
| ..basin **offered then lost** (selection, #5) | 58 | **63** |

Rotation-aware ranking cuts candidate-generation failures by 24% (17 → 13).
The wrong tiles do not disappear — they *migrate* into the selection bucket,
because once a wrong-scale basin is offered, native ZNCC sometimes prefers it.
83% of the remaining wrong tiles (63/76) are now selector failures.

The five pairs whose outcome changed, read at shortlist level:

| pair | GT z / θ | basin rank base → fix | outcome |
|---|---|---|---|
| `test_A_00000580` | 9.301 / **+5.00** | never offered → **0** | 91.5 px → **1.01 px** |
| `test_B_00000547` | 8.269 / +3.24 | never offered → **0** | 247.8 px → **0.45 px** |
| `test_B_00000363` | 10.706 / −4.73 | 0 → **0** | 0.27 px → 403.9 px |
| `test_B_00000662` | 9.759 / −2.67 | 1 → **0** | 1.02 px → 773.5 px |
| `test_B_00000287` | 11.235 / −3.14 | never → never | 3.19 px → 273.4 px |

Both rescues are candidate-generation rescues (2/2), and `test_A_00000580` is
the textbook case: GT rotation sits exactly on the **+5° endpoint**, where the
rot=0 probe sees nothing. **None of the three regressions lost a basin**
(0/3): two had the true basin still in the shortlist — one of them ranked
*first* — and the decode chose a wrong-scale hypothesis whose native ZNCC was
higher (0.3261 vs 0.2739; 0.3512 vs 0.3388), even though the *network* score
preferred the truth (0.4015 vs 0.2997; 0.7803 vs 0.6293). The third never had
the basin offered in either arm; its baseline 3.19 px was a lucky near-miss.
Per-failure trace: `.agents/rot37_failures.csv`.

### Ranking statistic: `max` is not the problem

`max` over 11 rotation samples is upward-biased, and the bias is larger for
noisier (small-scale) basins — the suspected cause of the promoted decoys. So
five statistics were scored offline against candidate-recall on all 2,250
cached matrices: `max`, `top2mean`, `top3mean`, `second`-best, and a
`coherent` variant (peak averaged with its two rotation neighbours).

**All five are recall-equivalent** — recall@3 88.69% and 198 never-offered,
identical to the digit; they differ from `max` on only 12–28 of 1750
shortlists and only at @1 (80.23–80.97%). The gain comes from being
rotation-aware *at all* (rot=0: 88.23%, 206 never-offered), not from the
choice of reduction. **No statistic change is warranted**; `max` stays.

### Clock (single process, 4 threads, interleaved, B_0000 n=20)

Stage `pose_candidates` 12.69 s → 13.57 s per 20 pairs (**1.069x**, +0.044 s
per pair). That stage is ~11% of a pair, so the added rotation work is
**~0.7% of pair time**. The end-to-end p50 (5.93 s → 6.34 s) is NOT resolvable
at this n: the arms' ranges overlap and one fix rep (5.34 s) beat every
baseline rep. Both arms read ~2.4x slower than the clocks recorded earlier in
this file under the same protocol, so **absolute** budget claims from this
bench are machine-specific — both arms miss ≤5 s here, so the change does not
alter budget status. The reference-sample median on record is 3.12 s.

### Verdict: `RERANK_ROTATION` stays False

This A/B is the evidence the `RERANK_ROTATION` gate was waiting on. That
constant's own comment sets the bar: *"To enable: set this True, run the full
2,250-pair A/B, and record the paired delta per component before changing the
default."* Done — the per-component deltas are the tables above. They do not
clear the bar.

* Issue #37's defect is **real and reduced**: candidate-generation failures
  17 → 13, basin never-offered 137 → 130, **zero basins lost, 8 gained**,
  recall@1 +3.83.
* Rubric effect is **neutral**: −0.003 of 85. Set A +0.0009, Set B −0.0021
  with a CI touching zero. Against the campaign's **+0.35 promotion gate**
  this is a clear no.
* Net tile count is **−1** (2 rescued, 3 broken) and the three regressions
  raise, not lower, their confidence — they are new *high-confidence* wrong
  tiles.
* Cost ~0.7% of pair time.

**So the default stays OFF, on the campaign's own rules** — the same standard
that kept the E1 cache and E3 pruning unshipped. The mechanism is kept, gated
and tested, because the defect it removes is unrecoverable downstream when it
fires, and because a different selector would change this verdict.

The finding that matters more than the gate decision: **the binding constraint
is now the native-ZNCC selector** (issue #5). 63 of 76 remaining wrong tiles
are pairs whose true basin *was* offered and rejected, and on two of the three
regressions the network score preferred the truth while ZNCC did not. Widening
or re-ranking the shortlist cannot pay until that is addressed; this A/B is the
strongest evidence yet for prioritising #5.

## Notes per tweak

**1. Consensus verification** (`register.py --verification consensus`):
Every component non-inferior on the full set (loc B +0.2 pp, rejection F1
+0.43 pp, calibration −0.0003 AUC), total +0.106 — but the pre-registered
gate is +0.35 and the paired CI spans zero. The 149-pair proxy's headline
property (zero broken successes) inverted at scale: 5 broken vs 6 rescued.
Left unshipped; issue #9's consensus A/B is answered: real but small.

**2. Checkpoint soups:** issue #10's actual ingredients (p6_last / p8_last /
p9_last) are not on this machine — only three same-run checkpoints from
epochs 12 / 24 / 30. Trajectory averaging (SWA-style) was tried anyway and
clearly hurts (−1.7 / −2.0): epochs 12 and 24 sit in different basins than
the epoch-30 fine-tune, which is exactly the "different basins produce
garbage" risk the averaging script warns about. The real soup experiment
stays with the training host, where the named checkpoints live.

**3. Threshold:** `optimize_threshold` picks 0.1974 over the shipped 0.2018
for +0.02 points. Shipped operating point confirmed; nothing to do.

**4. E1 template-embedding cache** (`ref_feat` plumbing + single-slot cache
in `DriftSenseNet.forward`, training-guarded): output-identical (7 tests,
incl. a counting test proving the template encoder runs once for 3
hypotheses). Measured clocks with the corrected flag-based harness (1.00x) — the template branch is
1–2 ms of a 2–4 s pair because the template input is 100x100 while the
search input is 1000x1000. The issue's premise ("recomputed identically 3x")
was true but irrelevant to wall-clock. Kept: the plumbing is the hook for
any future template-side reuse, and the cache is free.

**5. `band=False` promoted.** The band-pass default shipped mid-PR without
an A/B (issue #9 flagged exactly this). Measured on the full 2,250: no-band
wins +0.45 raw (75.78 → 76.23), paired loc delta +0.167 with CI excluding
zero, rejection F1 +1.1 pp (0.8772 → 0.8878), lost-real 76 → 65, and Set B
>=5px 88.5% → 89.3%. Clocks neutral. `locate_phase2` default flipped;
`eval_ext` gets `--band` opt-in (`--no-band` kept as a no-op for
compatibility). 200-pair draw: 75.70 vs 75.29 baseline.

## Status of the campaign

- Round 1 quick levers (consensus, soups, threshold): measured, none promoted.
- **Round 2: `band=False` promoted — the shipped decode is now worth ~76.23
  on the full 2,250 (+0.45), at identical clocks.** Highest-scoring
  configuration measured to date.
- E1 cache: kept as instrumentation; measured worthless for clocks (honest
  negative against issue #7's stated premise).
- Remaining levers: **E3 SEA-style coarse-sweep elimination** (the 66.8%
  clock share — the actual judged-efficiency lever), **E4 hypothesis
  pruning**, and **#5 margin-gated rescue pass** (accuracy).

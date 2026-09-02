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

## Campaign 2026-09-03 — inference-only score & CPU pass (no weight changes)

| tweak | evidence | verdict |
|---|---|---|
| fused 6-feature confidence (calibrated P(present), threshold 0.4870) | 4-fold CV on the 2,250 holdout: held-out AUC 0.9877 -> **0.9915**; in-sample total 75.71 vs scalar 75.35; margin/derived/9-feat all measured out (B_CALIBRATION_REPORT.md ADDENDUM 2); official-20 coordinates byte-identical, found decisions unchanged | **SHIPPED** (`SHIPPED_CONFIDENCE="fused6"`, `driftsense/calibration.py::calibrate_shipped`) |
| register.py thread caps (torch+cv2 = min(4, cores)) + stderr per-pair timings | untuned 2.98 s vs tuned-env 1.58 s median on the same 20 pairs (dev Mac); macOS GCD ignores cv2 caps (no-op here) — effect lands on the Linux x86 grader box; official-20 output byte-identical; resolves the "per-pair timing unproven" report gap | **SHIPPED** |
| bicubic sub-pixel placement (C workstream) | official-20 +0.40 credit (rescues p019/p020) but 60-pair holdout p95 shift 0.271 px > 0.15 gate (my leg) and gate-a break + credit −0.01 (C's independent leg); does not rescue p014 (the loc-tie pair — its error is upstream of sub-pixel) | **NOT SHIPPED** — flag `SHIPPED_SUBPIXEL="parabola"` retained; module flag-gated |
| upsampled-DFT sub-pixel (C workstream) | credit-neutral on holdout, gate-c fail (0.255 px), moved p019 the wrong way on official-20 | **NOT SHIPPED** |
| raw-surface rotation cross-check (integrator probe) | fixes p010 (+0.4), breaks p020 (−0.4): net 0.00 on official-20 | **NOT SHIPPED** (probe: `.agents/rot_crosscheck_tmp.py`) |
| coarse-sweep FFT search-DFT reuse (A workstream D3) | value parity 4.8e-08, 0/150 argmax disagreements, but net ~48 ms/pair ≈ 1.5% — 50 of 214 matchTemplate calls share the probe DFT; template construction (~47% of coarse cost) is FFT-immune | **NOT SHIPPED** — `driftsense/coarse_fft.py` flag-off instrumentation |

Full-suite state after the campaign: **314 passed, 0 failed**. Official-20 end-to-end
(graded entry point, subprocess): loc 39.27, pose 19.71, F1 1.000 (both
conventions), subtotal 73.98 — coordinates byte-identical to pre-campaign; the
score column now carries the calibrated statistic. AUC on the 20-pair sample
remains not estimable (single correctness class); the calibration evidence is
the 2,250-pair held-out CV. Known pre-existing property: cross-process score
bimodality (found/x/y/theta/scale never move; root-caused by A).

### Campaign rebase addendum (2026-09-03, onto origin/phase2 @ b3949a5)

The campaign branch was rebased onto the new phase2 tip (Set C fine-tune
weights, channels_last CPU, SHIPPED_SUBPIXEL_ROWS). Consequences, all measured:

* Fresh 500-pair holdout (seed 200, --features, NEW checkpoint): shipped fused
  config totals **78.45/85**, AUC **0.9927**. Same-pair comparison: legacy
  min() AUC **0.9689** vs fused **0.9561->0.9927?** — the paired figure is
  fused 0.9561 vs min 0.9227 on present pairs with err computed from gt_x/gt_y
  (see the CSV analysis); eval_ext's scored AUC (correctness-based) is
  0.9689 -> 0.9927. The fusion delta on the new checkpoint is ~9x the old one.
* Threshold: totals flat 78.45 (0.4870) / 78.50 (0.54) / 78.51 (0.6057) but
  lost-real present pairs double 3 -> 6; downward-bias instruction +
  noise-flat totals keep **0.4870 shipped**.
* Frozen constants kept (not re-fit): already validated on the new checkpoint;
  honest refit needs the full 2,250 re-decode (post-freeze).
* Official-20 attribution: merged-tree coordinates are IDENTICAL under
  fused vs legacy_min for 19/20 pairs (p012's single diff is a threshold-units
  artifact of forcing 0.487 onto legacy scores). Upstream-equivalent
  (legacy @ 0.18) reproduces the merged tree's totals exactly (loc 38.82,
  pose 19.71, F1 1.000): the p007 0.554->1.100 regression and the p020
  1.018->0.763 gain belong to origin/phase2's new checkpoint/row-drift, NOT
  to this campaign. This PR is metric-neutral on the official 20 and delivers
  the holdout AUC gain + runtime determinism.
* Full suite on the merged tree: **324 passed, 0 failed**.

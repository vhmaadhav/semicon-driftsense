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
is now MEASURED as a wash (below), so that lever is closed as specified.

## E3 pruning audit (2026-09-02) — closes the audit the default was pinned on

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
  1.00x clock fails the change bar. The "pending audit" state is closed with
  evidence — see the AUDITED note in `driftsense/matching.py`.

## Free regrades and closures (2026-09-02, no new inference)

| check | result | verdict |
|---|---|---|
| early-exit CSVs rescored @ shipped 0.18 (corrected masking) | every gate breaks more than it rescues (0.55: −0.0121 paired loc, 1/9; 0.85: −0.0032, 1/3) | early-exit-off confirmed evidence-backed |
| Set B denoise 3×3 A/B (300 pairs seed 200, fresh inference) | ≤5px 90.7→91.7%, paired loc +0.0140, 12 rescued / 9 broken | under the +0.35 gate — not shipped |
| rejector round 2 (GBM + 7 engineered features, honest 4-fold CV) | best CV 75.72 vs shipped scalar 75.87; AUC 0.9882 vs 0.9876; in-sample oracle F1 1.000 = overfit | post-hoc rejection closed for nonlinear + engineered families too |
| hyp-4 / coarse-29/43 sweeps rescored under masked semantics | Set B credit: hyp-4 0.7726 vs baseline 0.7692 (+0.0034); coarse monotone worse | prior verdicts hold |
| threshold 0.18 vs sweep, held-out with FIXED threshold | full set 75.73 (fixed) vs 75.68 (per-fold fitted); 200-draw 74.87 vs 74.45 | shipped 0.18 confirmed optimal out-of-sample |
| `refit_xy` | prior measurement stands (+0.04, wash) | stays off |

Clock row for E3 added to the table above; the E3 audit is closed in
`driftsense/matching.py` (AUDITED note) and commit bde0c47.

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

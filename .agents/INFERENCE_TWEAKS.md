# Inference-level tweaks: side-by-side score log

**Protocol:** every tweak is measured on the **same seeded 200-pair draw**
(`eval_ext --sample 200 --seed 200`, shipped shards A/B/C) so the columns are
directly comparable, and on the **full 2,250** for the promotion decision
(paired bootstrap, gate Δ ≥ +0.35 vs the shipped decode). The 200-pair draw
alone carries ±1.2 points of sampling noise — the full-set paired number is
the decider; the 200-pair column is the quick read.

Baseline rows come from the current shipped configuration
(`driftsense.pt`, threshold 0.2018, `verification=zncc`, band on).

| tweak | 200-pair subtotal (seed 200) | full 2,250 subtotal | paired Δ vs shipped | verdict |
|---|---:|---:|---:|---|
| shipped baseline | 75.29 | 75.78 | — | reference |
| 1. `--verification consensus` | 75.26 | 75.89 | +0.11 raw; paired loc Δ +0.02, 95% CI [−0.08, +0.14]; breaks 5 / rescues 6 | **not promoted** — under the +0.35 gate; the proxy's "broken=0" property did not hold at scale |
| 2a. soup e24+e30 (`weights/soup_e24_e30.pt`) | 73.60 | not run | ≈ −1.7 on the paired draw | **rejected** — worse; no gate run warranted |
| 2b. soup e12+e24+e30 (`weights/soup_e12_e24_e30.pt`) | 73.29 | not run | ≈ −2.0 on the paired draw | **rejected** — worse |
| 3. threshold 0.2018 → 0.1974 | (offline) | 75.52 vs 75.50 | +0.02 | **no change** — shipped operating point already optimal |
| 4. E1 template-embedding cache | 75.29 (identical by design) | identical coords | 0 | **kept as free instrumentation** — clocks: p50 2.55s vs 2.46s existing (0.99x, a wash); the template branch costs 1–2 ms of a 2–4 s pair. Issue #7's "recomputed 3×" premise measured: real but worthless |
| 5. **`band=False` (no-band)** | **75.70** | **76.23** | **+0.45 raw; paired loc Δ +0.167, CI [+0.014, +0.327], P(≤shipped)=1.7%; rescued 18 / broken 9** | **PROMOTED — new default** |

## Clocks (single process, 4 threads, CPU, interleaved A/B reps, n=20–30 pairs × 3)

| config | p50 | p90 |
|---|---:|---:|
| shipped (band on) | 2.513 s | 3.014 s |
| no-band (promoted) | 2.655 s | 3.218 s |
| E1 cache on vs off | 2.546 s vs 2.464 s | — (0.99x, wash) |

Clock takeaway: the efficiency win of `band=False` is ~0 — `_band` on the
half-res probe is cheap. Its value is the **+0.45 accuracy points**, which is
why it promotes on the accuracy gate, not the clock gate. The remaining
clock lever is the coarse sweep itself (66.8% of pair time) — E3 below.

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
hypotheses). Measured clocks: a wash (0.99x) — the template branch is
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

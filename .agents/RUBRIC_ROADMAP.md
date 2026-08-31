# Rubric roadmap: future inference-level and model-weights levers

**Date:** 2026-08-31 · **Baseline:** 76.23 / 85 measurable (no-band decode, full
2,250) · **200-pair grade draws:** 75.8 ± 1.2 (own-generator test split, never trained on) · **Efficiency:** 5 judged pts ·
**Generator/report:** 10 judged pts · **Ceiling with all components:** 85 + 5 = 90.

## Compute protocol (how we work from here)

| task class | hardware | why |
|---|---|---|
| **Model evaluation & accuracy prototyping** (full-set evals, sweeps, retraining experiments) | **GPU** — RTX 4060 local venv or Kaggle TPU-VM host | a full 2,250-pair eval drops from ~12 min (Air, 5 workers) to ~1–2 min; sweeps and retrains become interactive instead of overnight. Rapid iteration is worth more than clock purity here. |
| **Efficiency / clock measurements** (any number quoted for the judged 5 pts) | **4-core CPU reference method** — single process, 4 threads, idle, interleaved A/B reps (`scripts/bench_existing_vs_optimized.py`) | the judge clocks CPU. GPU numbers are meaningless for this component; thermals and thread counts must be controlled exactly as the bench harness does. |
| **Rapid development loop** | either | every change lands behind a flag; `eval_ext --sample 200 --seed 200` gives a 2-minute read; the paired full-set run is the only promotion gate. Scorecard of record: `.agents/INFERENCE_TWEAKS.md`. |

Never compare clocks across machines or across thermal states; never promote
accuracy from a 200-pair draw alone (σ = 1.2).

## Where the remaining points are

| component | points | standing | realistic headroom |
|---|---|---|---|
| Localisation | 40 | 35.07 (no-band) | +1.8 (rescue pass on 109 Set B gross failures) |
| Pose scale + rotation | 20 | 17.96 | +1.2 (same failures carry pose) |
| Rejection | 15 | 13.32 (F1 0.888) | +4 bonus at F1 ≥ 0.90 — **training path only** (measured: post-hoc ceiling 0.885) |
| Calibration | 10 | 9.87 (AUC 0.987) | ~0.1 — saturated |
| Efficiency (judged) | 5 | untouched | up to +5 via ≥1.5× runtime |
| Generator/report (judged) | 10 | untouched | docs/citations/failure-analysis quality |

## Inference-level future work (no training)

1. **E3 — SEA/FGSE-style coarse-sweep elimination** (issue #7 spec a). Skip
   `make_template`+`_band`+`_peak_score` for grid points whose half-res probe
   cannot beat the running k-th best. The coarse sweep is 66.8% of pair time —
   this is the real judged-efficiency lever. Equality audit on the full 2,250
   (identical found/x/y or paired CI ⊂ [−0.1, +0.1]); claim "empirically
   decision-equal", never "provably" (probe margin ≠ ZNCC bound).
2. **E4 — hypothesis pruning on coarse margin** (issue #7 spec c). Skip
   full-resolution attempts 2..k when the top coarse peak dominates. Flagged
   off by default; interacts with #5's rescue gate.
3. **#5 margin-gated rescue pass** — the biggest accuracy lever left. Gate on
   the shipped `winner_margin` (fires ~13%), re-sweep at grid midpoints on
   flagged pairs, accept only if the rescue wins native ZNCC by δ and passes
   rank+band consensus. Ceiling ≈ +1.8 loc + ~1.2 pose points. Tune on GPU.
4. **Re-run the small A/Bs on top of the band flip.** Consensus verification
   (+0.11 with band on) and the threshold re-tune (+0.02) were measured
   against the old decode; both are one-command re-runs and might move.
5. **Coarse-grid tuning:** `coarse_scales` 17 → 12 with midpoint polish
   compensation; `POSE_PROBE_DOWNSCALE` sensitivity sweep. Each is one GPU
   eval + paired test.
6. **Dihedral TTA under unknown pose (gated):** the Phase-1 TTA machinery
   exists; applying it only on low-`winner_margin` pairs bounds the cost.
   Measure on the periodic-ambiguity subset first (Set B sev4).

## Model-weights future work (GPU required)

1. **#11 Set C expansion + found-head retrain** — the +4 bonus path
   (rejection F1 0.878 → 0.90). 28 of 198 C shards held last time; p9 recipe
   (`--jitter-power -1`, one-cycle, `--finetune`) is the verified start;
   keep the ~35% B / 20% C pool ratio; acceptance = F1 ≥ 0.90 on the full
   2,250 with loc/pose non-inferior under the paired bootstrap.
2. **Real checkpoint soup (issue #10)** — p6_last/p8_last/p9_last live on the
   training host; the local epoch-12/24/30 trajectory average measured
   −1.7/−2.0 (different basins). The named fine-tunes are the textbook
   same-init setup; promote only on paired Δ ≥ +0.35.
3. **Retrain on corrected generator labels (audit C-02)** — the vendored
   generator's labels ignore search-side shear/drift/barrel; the main path's
   `correct_gt` machinery and the `label_gap`/`label_noise_floor` findings
   quantify the residual. Correcting the vendored generator and retraining
   attacks the label-noise floor that caps Set B sub-pixel accuracy
   (measured: sub-pixel precision is label-bounded).
4. **Band-aware fine-tune:** the network was trained with inputs that the
   (now removed) coarse band-pass no longer pre-filters — a 2–3 epoch
   fine-tune on the GPU host with the promoted decode in the loop may recover
   additional Set B credit. Cheap on GPU; gate as always.
5. **Calibration head:** fit per-set temperature scaling on held-out shards
   (GPU training, trivial compute) — protects the 10 calibration points if
   the 0.90 F1 push shifts the score distribution.

## Sequencing recommendation

1. GPU: #5 rescue pass prototype + sweep (highest accuracy ceiling).
2. CPU: E3 coarse-sweep elimination (only noise-free judged component).
3. GPU: #11 retrain (the +4 bonus; the largest single jump available).
4. GPU: soup + band-aware fine-tune (cheap lottery tickets, pre-registered gates).

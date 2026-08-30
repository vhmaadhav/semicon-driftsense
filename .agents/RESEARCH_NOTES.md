# Research notes — autonomous-research pass, 2026-08-30

Output of an `/autonomous-research` run (literature grounding → novelty checks →
measurement plan). Written for the team: every claim below is either measured here or
carries a citation; the literature grounding and three novelty reports live outside the
repo in the research workspace, and the distilled specs are mirrored in the GitHub
issues created from this document.

**Machine constraint, from our own docs (PHASE2_STATE §3/§6e):** this laptop hits
100 °C under sustained load and is memory-bandwidth bound at ~0.42 pairs/s; the grading
reference is 4-core x86, no GPU. Heavy evals belong on the reference box (idle,
overnight) or the Kaggle TPU-VM host (96 cores) that already hosts the corpus on Drive.
Only the ~1-minute profile below was run locally.

## 0. Environment facts found on the way (all fixed or worked around)

* The old `venv` interpreter is broken (`encodings` import failure); `venv313` works
  (torch 2.13.0 CPU, numpy 2.5.2). Runbook lines saying `./venv/bin/python` are dead.
* `fetch_shards.sh` had three GNU-only idioms (`xargs -a`, `stat -c%s`,
  `export -f`+`bash -c` dispatch) that silently did nothing on macOS — plus a
  **wrong-run hazard**: two generation runs share each (split, set, index) on the Drive
  folder, and the unpinned queue picked the wrong one. Fixed in this PR:
  portable worker loop, `wc -c` size check, `RUN_ID` filter (pin `a06d9df…`).
* `data/ext_p2` was restored byte-compatible via the notebook's OAuth path
  (`drive/v3/files/<fid>?alt=media` with the refresh token from the Kaggle secret):
  A 875 + B 875 + C 500 = **2,250 pairs, matching `cand_driftsense_p9_last.csv`**, D 250.

## 1. Measured runtime split — the docs' "86% network" is stale

`scripts/profile_pair.py data/ext_p2/test_A_0000 --n 12 --threads 4` (current PR code,
venv313): **median 1.42 s, p90 1.71 s** (docs' 3.35–3.5 s figure not reproduced; it
likely reflects the mixed 2500-pair set / earlier config — re-measure on the full set).

| stage | share |
|---|---:|
| `pose_candidates` (17-pt coarse sweep, band-passed) | **66.8%** |
| `locate` (network, 3 hypotheses) | 21.3% |
| `polish_pose` | 10.8% |
| `canonicalize` + `refine_zncc` | ~0.1% |

Consequence: the efficiency lever is the **coarse sweep**, not the network. The
network's template branch is still recomputed identically per hypothesis
(`model.py:197-198`; `matching.py:927`) and remains worth caching, but the bigger,
previously unnamed target is the sweep itself (make_template + band per scale point).

## 2. Idea A — margin-gated second pose search on flagged pairs (accuracy)

* **Targets:** the only above-noise accuracy bucket — 109 Set B gross failures
  (1.79 loc pts + ~1.2 pose pts). Window: 24% of failures had a correct candidate
  (selection-limited), 58% are fixed by the true pose (search-limited).
* **Spec:** when the winner's `min(score,zncc)` margin to the runner-up is below τ
  (~13% of pairs), re-run a *differently-sampled* coarse sweep (e.g. midpoints between
  the current grid points — the current 2.5% step can step over a 1–2% peak), polish,
  and accept only if the new hypothesis wins native ZNCC by δ **and** passes the
  rank+band consensus (`verification="consensus"` machinery, already merged).
* **Novelty (verified):** confidence-gated re-detection is standard in long-term
  tracking — LTMU (CVPR 2020), Reliable Re-Detection (TCSVT 2018), EURASIP JASP 2021
  (max-response+APCE), CATUR (Sensors 2026); Tesseract does it in OCR. **No published
  instance for template matching/SEM pose search**; frame as a transplant, not a new
  principle. Corroborates our measurement that gated beats always-on widening.
* **Protocol:** stride-5 pre-read on ext_p2, then full-2250 paired bootstrap
  (promote only if paired Δ ≥ +0.35 ≈ 2σ); runtime p50/p90 before/after (must stay
  inside the 5 s median / 20 s timeout budget).
* **Risk:** decoys — the same reason *global* widening measured monotonically worse;
  the margin gate + consensus acceptance is precisely what that experiment lacked.

## 3. Idea B — rank/band as present/absent rejector features (rejection F1 → 0.90)

* **Targets:** F1 0.878 → 0.90 = +0.33 pts **and the +4 bonus**; the reject-option
  literature (Chow; arXiv 2101.12523) says only score *quality* moves F1.
* **Gap, proven in code:** the rejector feature universe is exactly six
  (`rejector_cv.py:32`, `eval_ext.py:81-114`); rank/band are computed only under
  `need_verification_scores` (`matching.py:733`), consumed only by `choose()`
  (:806-817), popped from the result contract (:910-914) — never offered to the
  present/absent decision, although they target the measured failure discriminators
  (salt-pepper d=1.21, charging d=1.20).
* **Novelty (verified, moderate-high confidence):** first fitted multi-feature
  present/absent rejector for SEM template matching. Closest priors are adjacent, not
  equal: Nandakumar et al. (TPAMI 2007, LR score fusion), Hu & Mordohai (TPAMI 2012,
  confidence to reject incorrect matches), rank-reliability in stereo (IEEE SMC-B
  2001), MOSSE/PSR gate (CVPR 2010), TLD (TPAMI 2012). EM-domain prior (Buniatyan
  2017, full-text checked) uses band-pass only as a baseline and threshold-NCC "rejection".
* **Spec:** extend `rejector_cv.py` with `rank`, `band`, and winner-margin features;
  4-fold CV on the 2,250; fit on the *total rubric*, not F1; compute cost ~0.1 s/pair
  when verification features are on (gate-firing-rate lesson applies: absent pairs included).
* **Risk:** the measured AUC↑/F1↓ cancellation (+0.11 prior). If it reproduces with the
  new features, that closes the post-hoc rejection question for good.

## 4. Idea C — runtime: prune the coarse sweep, cache the template branch (efficiency)

* **Retarget after §1:** cut `pose_candidates`, not the network — candidate cuts:
  (a) skip the per-scale-point `make_template`+`_band` for grid points whose
  half-res probe cannot beat the running k-th best (SEA/FGSE-style elimination —
  pixel-window literature: MSEA 2012 "exactly the same result as exhaustive search";
  ZNCC bounded partial correlation, PRL 2005), (b) cache the invariant template-branch
  encoder output across hypotheses (`model.py:197-198`), (c) prune hypotheses 2..k on
  coarse margin (heuristic — the docs themselves say the probe cannot separate basins:
  `matching.py:338-349`, `:717-724`).
* **Claim discipline:** "empirically decision-equal on the full 2,250" — not "provably"
  — unless a certified bound on native ZNCC is derived; that bound is the ownable
  novelty (SiamFC already shares the encoder across scales via a scaled mini-batch).
* **Protocol:** equality audit on 2,250 (identical found/x/y or paired Δ CI ⊂ [−0.1, 0.1]);
  runtime p50/p90 single-process 4 threads idle; target ≥1.5× median. Efficiency is
  quartile-ranked (5 judged pts) — the only noise-free component left.

## 5. Baseline stage (code already in this PR, measurement pending)

* `register.py` now wires `--verification {zncc,consensus,majority}` through to
  `locate_phase2` (default `"zncc"` = bit-identical shipped behaviour). Measure
  `consensus` on the full 2,250 before enabling (PR #3's +0.31–0.37 was a 149-pair proxy).
* `eval_ext.py` default threshold aligned to the shipped 0.2018.
* `--no-band` A/B still open (band became the coarse-sweep default without a recorded
  full-set A/B).

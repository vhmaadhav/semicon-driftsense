# Failure Analysis — Phase 2

Living source for the final `failure_analysis.pdf` (max 2 pages). Keep this evidence-backed and concise; update it whenever a meaningful code, model, dataset, benchmark, bug-fix, or validation result changes the failure picture.

## Current failure modes

### 1. Set B sub-pixel error — centre-row raster drift
- **Observed:** before the row correction, Set B median centre error was ~0.83–0.88 px on the measured stacks; the fresh 500-pair holdout improved to **0.48 px** with the correction, with localisation **+1.00/85** overall.
- **Cause:** search-side raster jitter perturbs x independently by row; rigid ZNCC estimates the row-average displacement while the label follows the target centre row.
- **Mitigation:** `drift_row_refine` estimates/dewarps row offsets and reapplies the centre-row displacement; shipped behind `SHIPPED_SUBPIXEL_ROWS`.
- **Remaining limitation:** this mechanism is matched to our generator's row-jitter model. It no-ops when the signal is not measurable and cannot rescue wrong pose basins.

### 2. Set B gross failures — wrong pose basin / periodic ambiguity
- **Observed:** widening the pose search and a conditional wide rescue did **not** fix gross failures; the measured rescue experiment fixed zero gross failures and was not promoted.
- **Cause:** periodic semiconductor structure creates plausible wrong-scale/rotation basins; more candidates can add decoys faster than useful coverage.
- **Mitigation:** keep the validated three-hypothesis path; rotation-aware re-ranking exists but is **OFF by default** until a full paired A/B proves benefit.
- **Remaining limitation:** sub-pixel correction cannot recover pairs whose candidate set never contains the correct basin.

### 3. Set C rejection — blind-set threshold risk
- **Observed:** the completed Set-C fine-tune improved rejection F1 from **0.9078 → 0.9198** at the shipped threshold `0.18`, while total measured score moved about **+0.33/85** on the 2,250-pair self-generated holdout.
- **Cause:** absent references remain periodically plausible; threshold-only rejection is fragile when degraded present pairs overlap absent confidence.
- **Mitigation:** ship the completed Set-C checkpoint and the validated fixed threshold; keep nonlinear/post-hoc rejector experiments out because their measured gains did not justify the trade-off.
- **Remaining limitation:** organizer scoring uses a 200-pair blind set, so the measured F1 margin above the +4 gate is not guaranteed to reproduce.

### 3b. Calibration ranking — scalar confidence wastes fusion signal (new checkpoint)
- **Observed:** on a fresh 500-pair holdout decoded with the Set-C checkpoint (2026-09-03 campaign rebase), the legacy scalar `min(net, zncc)` ranks per-pair correctness at **AUC 0.9689** while the shipped 6-feature fusion reaches **0.9927** (same pairs) — the new checkpoint sharpened the network signal and the fusion exploits it (raw ZNCC alone: 0.7750). CV tooling (`scripts/fit_calibration.py`) reproduces the family ordering; a derived gap-feature variant reads 0.9931 but is 500-pair noise-adjacent.
- **Cause:** peak *heights* (score, zncc) alone cannot separate a confident wrong lock-on from a true match; peak-quality statistics (peak_ratio, pose_peak, psr, apce) carry the missing information (Bolme et al., MOSSE 2010; fusion grounded against monotone-map AUC invariance, Guo et al. arXiv:1706.04599).
- **Mitigation attempted and rejected:** a fused 6-feature confidence (`driftsense/calibration.py`) was fitted and measured against the incumbent `min(network score, native ZNCC)`. On the 2,500-pair pool its constants were fitted on it gained +0.18; on an untouched 500-pair holdout it lost **0.43** (paired bootstrap P(better) = 0.011) and moved rejection F1 0.8958 → 0.8663, i.e. away from the +4 bonus gate at F1 ≥ 0.90. It is **not shipped**. The implementation is retained behind `driftsense.config.SHIPPED_CONFIDENCE`, which now selects `"min_med3"` at threshold 0.55 for the v2 extension (see section 7); `"legacy_min"` at 0.18 was the Phase 2 pairing. Evidence: `.agents/B_CALIBRATION_REPORT.md`.
- **Remaining limitation:** frozen constants were fit on the pre-Set-C checkpoint's feature distributions and re-validated (not re-fit) on the new one; a full 2,250-pair re-decode + refit is the follow-up. Official-20 AUC remains non-estimable (single correctness class).

### 4. CPU runtime / timeout risk
- **Observed:** CPU `channels_last` reduced measured median network-stack runtime from **4.97 s → 1.82 s** on the profiled setup. Independently (2026-09-03 campaign), `register.py` never capped thread pools: the same 20 pairs read **2.98 s/pair untuned vs 1.58 s tuned** on one machine, explaining roughly half of a foreign-harness 7.08 s/pair reading as thread oversubscription. End to end on 600 internal pairs at the shipped 4-thread cap, per-pair latency is **median 0.96 s, mean 1.00 s, p90 1.34 s, max 1.64 s, 0 pairs over 20 s** (Apple M4, arm64, 4P+6E cores — *not* the judge's 4-core x86 box; see `.agents/PR51_CAMPAIGN.md`).
- **Cause:** NCHW caused repeated oneDNN activation reorders; the network, not the coarse sweep, was the dominant graded-CPU cost. Thread pools default to every physical core (torch intra-op + OpenCV), oversubscribing a 4-core grader box; on macOS GCD ignores `cv2.setNumThreads` (no-op) while Linux pthreads/TBB honors it. Beyond that, every pose hypothesis paid a full network forward even when the first one was already uncontested, and hypotheses landing in the same scale/rotation basin were evaluated twice.
- **Mitigation:** channels-last on CPU with a safe fallback; `register.py cap_threads()` caps torch+OpenCV to `min(4, cores)` at process start (`--threads` overrides), with `torch.set_flush_denormal` best-effort; Conv+BatchNorm folding (an eval-mode algebraic identity); an uncontested-hypothesis early exit whose gates live in `driftsense.config.EARLY_EXIT_GATES` and which is **measured bit-identical** to evaluating every hypothesis (0/200 found flips, 0.0 max score delta, 1.18x faster); per-pair timings emitted for audit (stderr when redirected, a `<output>.timing` sidecar when stderr is a terminal). Late sub-pixel refinement is exception-guarded so a refinement failure cannot zero a whole otherwise-correct row.
- **Measured out and removed:** same-basin candidate deduplication skipped hypotheses whose pose lay inside a kept candidate's polish window. That reasoning does not hold where it ran: dedup precedes neural localisation, and `polish_pose` only re-fits pose around an already-chosen `(x, y)`, so nearby hypotheses can still land on different periodic repeats. Measured over 600 pairs it moved **123 localisation tier crossings**, cost **0.12 points** on one set (81.30 -> 81.42 with it off) and saved **no time at all** (median 0.964 s with, 0.960 s without). Off by default (`DRIFTSENSE_DEDUP=0`); the code is retained for a future set with genuinely clustered candidates, which would require re-running the A/B rather than assuming these numbers transfer.
- **Measured, not assumed:** the reduced golden-section polish budget (`_refine_pose_local` 1x4, `polish_pose` 1x6) was A/B'd against the previous 2x8 / 2x7 on three independent 200-pair sets: 81.45 vs 81.39 mean subtotal, i.e. a wash against a 0.4 per-set spread, for 1.55x the speed (median 0.915 s vs 1.421 s on one set, idle machine). A single-set A/B said the opposite (+0.21 for the deeper budget) and was wrong — the same one-sample error the `fused6` result exposed. Evidence: `.agents/PR51_CAMPAIGN.md`.
- **x86 validation (2026-09-03, post-merge):** the M4 figures above are ARM; a fresh 60-pair Phase-2-style set decoded end-to-end on real x86 (AMD Ryzen AI 7 350, 8c/16t, `--threads 4` -- the shipped cap) read **median 2.66 s, mean 3.48 s, p90 6.16 s, max 7.16 s, 0 pairs over 20 s**, found-rate 47/60 consistent with the ~80% present rate. Real x86 is roughly **2.8x slower** than the M4 number. Still comfortably inside the 5 s median target and the 20 s hard timeout, but every runtime figure quoted anywhere in this repo before this run was ARM, not x86 -- treat the M4 numbers above as directional, not as evidence of meeting the graded-hardware budget.
- **Remaining limitation:** the x86 run above used a locally-generated set, not the organizer's own blind set, and ran on a machine well above the reference 4-core/8GB spec (16 logical CPUs, ~15.5GB RAM) with only thread count capped to emulate the constraint -- a true 4-core-only box could read slower still. The coarse sweep's remaining FFT-immune cost is template construction (~47% of coarse time).

### 5. Generator / label fidelity
- **Observed:** the Issue 45 audit measured a maximum realised-raster label correction of **4.903 px**. All 16 present pairs passed raw-intensity and independent gradient verification; Set-C same-family absent NCC scores ranged **0.3254–0.8658**.
- **Mitigation:** fixed pose geometry, post-write dual verification, explicit semantic absence labels, Set-C similarity auditing, and supersampled anti-aliasing comparisons are integrated in `generator/`.
- **Remaining limitation:** the coarse NCC baseline's error is not monotone at severity level 4 because periodic structure can create a harder wrong basin at a lower nominal degradation level. This is retained in the report rather than hidden by relabelling.

## Phase 2 v2 extension (measured 2026-09-18)

The extension re-runs the same SEM-to-SEM task on a harder generator: a severity ladder with per-row raster shear (1–3 px across the frame) and white per-row jitter (sd up to 1.05 px), charging streaks, heavier impulse/speckle noise — and **pixel-centre labels**. The shipped Phase 2 decoder scored **76.71/85** on the mentor's 25-pair set. Every number below is measured on seed-disjoint 500-pair dev/holdout splits drawn from that same generator (`scripts/gen_phase2_v2_val.py`), with each choice made on dev alone and confirmed on the untouched holdout. Per-pair tables for the confidential mentor set stay local.

### 6. Coordinate convention — the extension's labels are pixel-centre (#86)
- **Observed:** a y error of **+0.49 ± 0.14 px** across the present pairs, 18 of 20 of them inside +0.32 to +0.65. That is a constant, not scatter. Once removed, every remaining localisation error was horizontal.
- **Cause:** the v2 generator labels `M @ (x0 + 499.5, y0 + 499.5)`; our decoder reports top-left + tw/2, i.e. pixel-edge. The same point, named half a pixel apart. The convention also decides **which scan row** a label's raster-drift sample is read from — `round(y_centre)` against `round(y_edge)` — a different row on about half of all pairs.
- **Mitigation:** `SHIPPED_LABEL_CONVENTION`, applied as the last geometric step so every stage above keeps one internal convention. Mentor set 76.71 → **81.73**; fresh 48-pair set 77.03 → 79.34.
- **Remaining limitation:** this is a property of the dataset, so it ships as a flag, not a new default — the original generator's own data moves the other way by the same half pixel (82.75 → 79.93).

### 7. Rejection on v2 frames — the statistic, not the threshold (#87)
- **Observed:** on the v2 dev split the historical `min(network, native ZNCC)` cannot separate present from absent at **any** threshold: absent max 0.529 against present min 0.370.
- **Cause:** the failure is on the *present* side. Impulse, speckle and shot noise drag a true match's ZNCC to 0.37 while the network — trained on noisy frames — stays at or above 0.69. A threshold cannot fix an overlapping distribution.
- **Mitigation:** measure the ZNCC term on a 3×3-median copy of the frame (`SHIPPED_CONFIDENCE="min_med3"`). A median restores correlation where there is structure to restore and cannot invent it where there is none, so present pairs lift and absent ones do not: dev present min 0.592 against absent max 0.529. Gate 0.55, fixed inside that empty band before any held-out split was scored. Holdout 80.23 → **82.39**; absent pairs accepted 25 → 0.
- **Remaining limitation:** the band was measured on one generator's noise ladder. The statistic is what generalises; the exact gate is the part a different blind set could move.

### 8. Rotation — raster drift corrupts half of the pose signal (#88)
- **Observed:** rotation credit 0.889 on the mentor set, with the errors growing with severity. Started **at the ground-truth pose**, `polish_pose` still walked ~0.2 deg away on Set B, so the objective is biased rather than under-searched.
- **Cause:** a rotation error `d` displaces template point (u, v) by (d·v, −d·u). The horizontal half varies along the row axis, which is exactly the axis raster drift acts on — a 1–3 px shear impersonates 0.06–0.17 deg of rotation. A 2-D correlation fit uses both halves, so it inherits the drift.
- **Mitigation:** re-measure rotation from the **vertical** offsets of vertical template strips, where drift, shear, scale error and barrel distortion contribute nothing, on a de-streaked and median-filtered correlation copy; blend it with the polish estimate by inverse variance (`driftsense.matching.strip_rotation`). Dev rotation credit 0.895 → **0.958**, holdout 0.879 → 0.954; totals +0.61 [+0.43, +0.80] and +0.79 [+0.61, +1.00]. The original generator's own audit set gains too (rotation 9.21 → 9.43), so this is an estimator fix, not a v2 convention fix. Cost +3 ms median per pair.
- **Remaining limitation:** adopting the strip regression whole gives back half the gain, so the prior in the blend is a fitted quantity (flat from 0.10 to 0.22 on dev). A set whose drift model differs would want it re-measured.

### 9. Drift-row re-placement was trusted too much (#89)
- **Observed:** on the dev split the stage **declines on 35% of present pairs**, and the declining guard is always the same one — the label row's own correlation below the floor, never the row count and never the runaway clamp. Worse, where it did fire it was actively harmful at both ends of the severity ladder: mean |x error| at severity 0 went 0.233 px (no correction) → 0.251 px (corrected), and at severity 4 0.773 → 0.839.
- **Cause:** the measured row offset is `s + e` — the row's own drift sample plus measurement noise. On a quiet frame there is almost no `s` to recover, and at severity 4 the row is measured badly, so in both regimes the correction is mostly `e`. Taking it whole is the wrong estimator. Bucketing by the row's correlation shows it directly: below 0.5 the correction helped only ~35% of the time and raised the mean error.
- **Mitigation, measured but NOT shipped:** scale the correction by its own signal-to-noise, `1 − σ_m²/σ_resid²`, with `σ_resid` already measured per pair (the scatter of row offsets about their smooth trend) and `σ_m` modelled from the row's correlation peak. Both ends are repaired (0.207 and 0.754) while the middle keeps its gain: dev localisation 38.90 → **39.07**/40, holdout 38.86 → 38.98, and the severity-4 within-1 px rate 0.650 → **0.717**. It is worth +0.17 and +0.12 on the two 500-pair splits — below this repo's +0.35 promotion gate, with the holdout CI including zero — so it sits in a **draft PR** pending a decision rather than on `main`. A row-preserving 1×3 median, the obvious alternative, measured **negative** (38.92 → 38.61).
- **Remaining limitation:** until that decision is taken, the shipped stage still applies each row correction at full weight, so the two ends of the severity ladder keep the harm described above. One severity-4 mentor pair remains 1.85 px out: its label row correlates at 0.23, so there is no measurement to trust at any weight. That is the information floor of a single-row estimator, and beating it needs a different measurement, not a better guard.

### Where the extension stands
| step | issue | mentor 25-pair | dev (500) | holdout (500) |
|---|---|---|---|---|
| shipped Phase 2 | | 76.71 | | |
| pixel-centre coordinates + drift row | #86 | 81.73 | | |
| median-ZNCC confidence, gate 0.55 | #87 | 83.40 | 82.80 | 82.39 |
| drift-immune rotation | #88 | 83.58 | 83.41 | 83.18 |
| drift-row shrinkage (measured, in draft — not on `main`) | #89 | *84.07* | *83.57* | *83.31* |

The shipped total is the #88 row; the #89 row is italicised because it is measured but not merged. Localisation is now near-saturated on nominal pairs and the remaining loss is concentrated in severity 3–4 horizontal residuals; rotation carries most of the rest.


## Release rule

Only measured failures and validated mitigations belong here. Keep exact experiment/PR references when available; remove or revise a statement when newer evidence invalidates it. The final PDF should be compiled from this file, not maintained separately.

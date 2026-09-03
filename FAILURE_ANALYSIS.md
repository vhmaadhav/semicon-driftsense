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
- **Mitigation attempted and rejected:** a fused 6-feature confidence (`driftsense/calibration.py`) was fitted and measured against the incumbent `min(network score, native ZNCC)`. On the 2,500-pair pool its constants were fitted on it gained +0.18; on an untouched 500-pair holdout it lost **0.43** (paired bootstrap P(better) = 0.011) and moved rejection F1 0.8958 → 0.8663, i.e. away from the +4 bonus gate at F1 ≥ 0.90. It is **not shipped**: `SHIPPED_CONFIDENCE="legacy_min"` at threshold 0.18. The implementation is retained behind that constant. Evidence: `.agents/B_CALIBRATION_REPORT.md`.
- **Remaining limitation:** frozen constants were fit on the pre-Set-C checkpoint's feature distributions and re-validated (not re-fit) on the new one; a full 2,250-pair re-decode + refit is the follow-up. Official-20 AUC remains non-estimable (single correctness class).

### 4. CPU runtime / timeout risk
- **Observed:** CPU `channels_last` reduced measured median network-stack runtime from **4.97 s → 1.82 s** on the profiled setup. Independently (2026-09-03 campaign), `register.py` never capped thread pools: the same 20 pairs read **2.98 s/pair untuned vs 1.58 s tuned** on one machine, explaining roughly half of a foreign-harness 7.08 s/pair reading as thread oversubscription. End to end on 600 internal pairs at the shipped 4-thread cap, per-pair latency is **median 0.96 s, mean 1.00 s, p90 1.34 s, max 1.64 s, 0 pairs over 20 s** (Apple M4, arm64, 4P+6E cores — *not* the judge's 4-core x86 box; see `.agents/PR51_CAMPAIGN.md`).
- **Cause:** NCHW caused repeated oneDNN activation reorders; the network, not the coarse sweep, was the dominant graded-CPU cost. Thread pools default to every physical core (torch intra-op + OpenCV), oversubscribing a 4-core grader box; on macOS GCD ignores `cv2.setNumThreads` (no-op) while Linux pthreads/TBB honors it. Beyond that, every pose hypothesis paid a full network forward even when the first one was already uncontested, and hypotheses landing in the same scale/rotation basin were evaluated twice.
- **Mitigation:** channels-last on CPU with a safe fallback; `register.py cap_threads()` caps torch+OpenCV to `min(4, cores)` at process start (`--threads` overrides), with `torch.set_flush_denormal` best-effort; Conv+BatchNorm folding (an eval-mode algebraic identity); an uncontested-hypothesis early exit whose gates live in `driftsense.config.EARLY_EXIT_GATES` and which is **measured bit-identical** to evaluating every hypothesis (0/200 found flips, 0.0 max score delta, 1.18x faster); per-pair timings emitted for audit (stderr when redirected, a `<output>.timing` sidecar when stderr is a terminal). Late sub-pixel refinement is exception-guarded so a refinement failure cannot zero a whole otherwise-correct row.
- **Measured out and removed:** same-basin candidate deduplication skipped hypotheses whose pose lay inside a kept candidate's polish window. That reasoning does not hold where it ran: dedup precedes neural localisation, and `polish_pose` only re-fits pose around an already-chosen `(x, y)`, so nearby hypotheses can still land on different periodic repeats. Measured over 600 pairs it moved **123 localisation tier crossings**, cost **0.12 points** on one set (81.30 -> 81.42 with it off) and saved **no time at all** (median 0.964 s with, 0.960 s without). Deleted.
- **Measured, not assumed:** the reduced golden-section polish budget (`_refine_pose_local` 1x4, `polish_pose` 1x6) was A/B'd against the previous 2x8 / 2x7 on three independent 200-pair sets: 81.45 vs 81.39 mean subtotal, i.e. a wash against a 0.4 per-set spread, for 1.55x the speed (median 0.915 s vs 1.421 s on one set, idle machine). A single-set A/B said the opposite (+0.21 for the deeper budget) and was wrong — the same one-sample error the `fused6` result exposed. Evidence: `.agents/PR51_CAMPAIGN.md`.
- **Remaining limitation:** organizer hardware differs and nothing here ran on x86; the final package still requires a clean 4-core x86 timing pass with p50/p90/max recorded. The coarse sweep's remaining FFT-immune cost is template construction (~47% of coarse time).

### 5. Generator / label fidelity
- **Observed:** Phase 2 requires labels to survive all geometric operations and absent decoys to remain same-family but genuinely absent. Earlier generator variants could create overly similar Set-C decoys and post-pose geometry can shift labels by scoring-relevant pixels.
- **Mitigation:** decoy-pitch fidelity fix, explicit pose geometry, post-write verification, and official-material traceability are part of the release path.
- **Remaining limitation:** the repository still needs the final Phase-2 generator-assignment package and its `REPORT.md`/verification evidence integrated into the release tree.

## Release rule

Only measured failures and validated mitigations belong here. Keep exact experiment/PR references when available; remove or revise a statement when newer evidence invalidates it. The final PDF should be compiled from this file, not maintained separately.
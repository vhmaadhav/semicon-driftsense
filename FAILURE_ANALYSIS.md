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

### 4. CPU runtime / timeout risk
- **Observed:** CPU `channels_last` reduced measured median network-stack runtime from **4.97 s → 1.82 s** on the profiled setup, with no found/tier decision changes in the validation set.
- **Cause:** NCHW caused repeated oneDNN activation reorders; the network, not the coarse sweep, was the dominant graded-CPU cost.
- **Mitigation:** use channels-last on CPU with a safe fallback; late sub-pixel refinement is exception-guarded so a refinement failure cannot zero a whole otherwise-correct row.
- **Remaining limitation:** organizer hardware differs; final package still requires a clean 4-core CPU timing pass with p50/p90/max recorded.

### 5. Generator / label fidelity
- **Observed:** Phase 2 requires labels to survive all geometric operations and absent decoys to remain same-family but genuinely absent. Earlier generator variants could create overly similar Set-C decoys and post-pose geometry can shift labels by scoring-relevant pixels.
- **Mitigation:** decoy-pitch fidelity fix, explicit pose geometry, post-write verification, and official-material traceability are part of the release path.
- **Remaining limitation:** the repository still needs the final Phase-2 generator-assignment package and its `REPORT.md`/verification evidence integrated into the release tree.

## Release rule

Only measured failures and validated mitigations belong here. Keep exact experiment/PR references when available; remove or revise a statement when newer evidence invalidates it. The final PDF should be compiled from this file, not maintained separately.
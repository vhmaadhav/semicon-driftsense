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
- **Observed:** on a fresh 500-pair holdout decoded with the Set-C checkpoint (2026-09-03 campaign rebase), the legacy scalar `min(net, zncc)` ranks per-pair correctness at **AUC 0.9689** while the *experimental* 6-feature fusion (`fused6`, **not shipped** — see the mitigation line below) reaches **0.9927** on the same pairs under the lenient present-only convention — the new checkpoint sharpened the network signal and the fusion exploits it (raw ZNCC alone: 0.7750). CV tooling (`scripts/fit_calibration.py`) reproduces the family ordering; a derived gap-feature variant reads 0.9931 but is 500-pair noise-adjacent. Both figures are subsample AUCs under the present-only convention, **not** the rubric definition (score column against per-pair correctness), which `scripts/eval_ext.py` reports separately at 0.76–0.81 — and against the 10-point AUC component the shipped `legacy_min` already reads 0.9882 on the full 2,500, so the ranking gain was worth about **+0.05 points**.
- **Cause:** peak *heights* (score, zncc) alone cannot separate a confident wrong lock-on from a true match; peak-quality statistics (peak_ratio, pose_peak, psr, apce) carry the missing information (Bolme et al., MOSSE 2010; fusion grounded against monotone-map AUC invariance, Guo et al. arXiv:1706.04599).
- **Mitigation attempted and rejected:** a fused 6-feature confidence (`driftsense/calibration.py`) was fitted and measured against the incumbent `min(network score, native ZNCC)`. On the 2,500-pair pool its constants were fitted on it gained +0.18; on an untouched 500-pair holdout it lost **0.43** (paired bootstrap P(better) = 0.011) and moved rejection F1 0.8958 → 0.8663, i.e. away from the +4 bonus gate at F1 ≥ 0.90. It is **not shipped**: `SHIPPED_CONFIDENCE="legacy_min"` at threshold 0.18. The implementation is retained behind that constant. Evidence: `.agents/B_CALIBRATION_REPORT.md`.
- **Remaining limitation:** frozen constants were fit on the pre-Set-C checkpoint's feature distributions and re-validated (not re-fit) on the new one; a full 2,250-pair re-decode + refit is the follow-up. Official-20 AUC remains non-estimable (single correctness class).

### 4. CPU runtime / timeout risk
- **Observed:** CPU `channels_last` reduced measured median network-stack runtime from **4.97 s → 1.82 s** on the profiled setup, with no found/tier decision changes in the validation set. Independently (2026-09-03 campaign), `register.py` never capped thread pools: the same 20 pairs read **2.98 s/pair untuned vs 1.58 s tuned** on one machine, explaining roughly half of a foreign-harness 7.08 s/pair reading as thread oversubscription.
- **Cause:** NCHW caused repeated oneDNN activation reorders; the network, not the coarse sweep, was the dominant graded-CPU cost. Thread pools default to every physical core (torch intra-op + OpenCV), oversubscribing a 4-core grader box; on macOS GCD ignores `cv2.setNumThreads` (no-op) while Linux pthreads/TBB honors it.
- **Mitigation:** channels-last on CPU with a safe fallback, plus Conv+BatchNorm folding (an eval-mode algebraic identity, 17 BatchNorms removed; paired pipeline median 2.04 → 1.91 s, 1.07×, 0 found flips / 0 tier changes over 252 pairs). The two are independent switches (`DRIFTSENSE_CHANNELS_LAST`, `DRIFTSENSE_FUSE_BN`), each degrading to the unoptimised path on failure; `register.py cap_threads()` caps torch+OpenCV to `min(4, cores)` at process start (`--threads` overrides), with `torch.set_flush_denormal` best-effort; per-pair timings emitted to stderr (`# t,<pair_id>,<seconds>` + summary) so per-pair runtime is now provable per pair. Late sub-pixel refinement is exception-guarded so a refinement failure cannot zero a whole otherwise-correct row.
- **Remaining limitation:** organizer hardware differs; final package still requires a clean 4-core CPU timing pass with p50/p90/max recorded. The coarse sweep's remaining FFT-immune cost is template construction (~47% of coarse time; search-side DFT reuse measured at only ~1.5% net and its instrumentation module was deleted rather than shipped — measurement retained in `.agents/A_EFFICIENCY_REPORT.md`).

### 5. Generator / label fidelity
- **Observed:** Phase 2 requires labels to survive all geometric operations and absent decoys to remain same-family but genuinely absent. Earlier generator variants could create overly similar Set-C decoys and post-pose geometry can shift labels by scoring-relevant pixels.
- **Mitigation:** decoy-pitch fidelity fix, explicit pose geometry, post-write verification, and official-material traceability are part of the release path.
- **Remaining limitation:** the repository still needs the final Phase-2 generator-assignment package and its `REPORT.md`/verification evidence integrated into the release tree.

## Regenerating `failure_analysis.pdf`

The PDF is the submission artifact and must be rebuilt whenever this file or the shipped configuration changes — it is generated from a measured `eval_ext.py` results CSV, not from this markdown, so the two are kept in step by hand.

`.agents/pr48_full.csv` is the 2,500-pair decode; its `score` column was recorded under the *experimental* `fused6` statistic, so re-derive the shipped one (`legacy_min` = `min(net_score, zncc)`, `driftsense/matching.py`) before plotting. Coordinates are unaffected — one decode, one statistic recomputed:

```bash
python - <<'EOF'
import numpy as np, pandas as pd
d = pd.read_csv(".agents/pr48_full.csv")
z = d["zncc"].where(np.isfinite(d["zncc"]), d["net_score"])
d["score"] = np.minimum(d["net_score"].astype(float), z.astype(float))
d.to_csv("/tmp/shipped_legacy_full.csv", index=False)
EOF
python scripts/eval_ext.py unused --rescore /tmp/shipped_legacy_full.csv --threshold 0.18
python scripts/failure_analysis.py /tmp/shipped_legacy_full.csv -o failure_analysis.pdf --threshold 0.18
```

The rescore step is the check, not decoration: it must print `SUBTOTAL 77.91`, `F1(reject) 0.9198`, `AUC 0.9882` — the same figures as the shipped-config decode in `.agents/pr48_legacy_full.log`. If it does not, the CSV is not the shipped configuration and the PDF must not be built from it.

## Release rule

Only measured failures and validated mitigations belong here. Keep exact experiment/PR references when available; remove or revise a statement when newer evidence invalidates it. The PDF's figures are generated from a measured results CSV rather than from this markdown (see above), so the two are separate artifacts that must move together: a change here that reflects a change in the shipped configuration requires a rebuild of the PDF in the same commit, and `scripts/build_submission_zip.py` ships whichever PDF is on disk.

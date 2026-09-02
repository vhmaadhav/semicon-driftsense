# Full-rubric rescore at the shipped threshold (0.18) — PR #34 headline CSVs

Answers review 5086839305 (CHANGES_REQUESTED): the committed headline logs
(`eval_base_nb.log`, `eval_setcfull.log`) that produce the 77.15 -> 77.54
claim were scored `@fix 0.202`, while `driftsense.config.SHIPPED_THRESHOLD =
0.18` is what ships. This document rescores **the same two feature CSVs**
through the shipped evaluator's own `score()` at exactly 0.18, and adds the
paired bootstrap the +0.35 promotion gate is defined on
(`RUBRIC_ROADMAP.md`: "promote only on paired Δ ≥ +0.35").

## Method

- `scripts/eval_ext.py --rescore <csv> --threshold 0.18` on
  `.agents/feat_base_nb.csv` (shipped weights) and
  `.agents/feat_setcfull.csv` (the candidate) — the identical CSVs the
  headline logs were generated from (2,250 pairs, paired by `pair_id`).
  Rescore applies the same threshold, submission masking, tier tables and
  weighted set A/B aggregation as a real decode; no inference re-run is
  needed because the threshold acts at scoring time.
- Paired bootstrap of the 85-pt subtotal delta (candidate − shipped), 10,000
  resamples, seed 0, resampling pair indices with replacement and
  differencing the two checkpoints on the identical resample. Both frames
  are aligned row-for-row by `pair_id` before resampling (set equality
  proves membership only, not row order). The resamples evaluate through a
  multiplicity-weighted fast path (every graded quantity in the subtotal is
  linear in per-pair draw counts), cross-validated against independent
  implementations before use — see below.
- **v2 correction (review 5087501487):** the first committed bootstrap
  computed `np.bincount(take)[take]` — the weight of the *drawn* pair at
  each draw-order position, not the multiplicity of original row i — which
  produced an invalid CI ([+0.1177, +0.5434], too narrow). The v2 driver
  uses the multiplicity vector `np.bincount(take, minlength=n)` directly and
  replaces the per-resample Python loop with a vectorised, block-bounded
  bootstrap (no per-resample loop; per-block peak ~100 MB including the
  weight-transport matrices).
- The threshold is **bound to the config**, not hard-coded: `T =
  ee.SHIPPED_THRESHOLD` (the totals assert in `main()` additionally pins the
  audited value), so a config bump cannot silently rescore a different
  point.
- **v3 (review 5089846291 point 3):** `bootstrap_vectorised` seeded each
  block `seed + s`, so the advertised seed's Monte Carlo sample depended on
  the memory-block size. v3 uses one `RandomState(seed)` consumed
  sequentially across blocks; C3b now asserts block sizes
  67/200/500/1000 produce identical deltas, and C5 asserts the
  per-component deltas sum to the subtotal deltas row-for-row.
- **Labelling (review 5089846291 point 2):** the staged-200 statistics in
  `.agents/STAGED_BOOTSTRAP_T018.md` are a **model-risk bootstrap over our
  own generated pool, NOT the probability of clearing the organizer's fixed
  blind stage** (A70/B70/C40/D20 drawn from their set, not sampled with
  replacement from our 2,250). All staged claims are relabelled
  accordingly.
- Cross-validation suite (any mismatch aborts before CIs are printed):
  **C1** full-frame fast path, **per component**, == `score()` (1e-9);
  **C2** six random resamples, per component, == `score()` on the expanded
  frames (1e-9) — per-component asserts so a compensating error (e.g.
  +loc/−pose) cannot survive; **C3** vectorised bootstrap == an explicit
  per-resample reference loop over 200 identical draws (1e-9); **C3b**
  block sizes 67/200/500/1000 produce identical deltas (block is a memory
  parameter only) and the production entry point matches independently
  recomposed blocks; **C5** per-component deltas sum to subtotal deltas
  row-for-row; **C4** weighted AUC == brute-force pairwise AUC on 30
  tie-forcing cases (1e-9).
- Driver (committed): `.agents/rescore018_driver.py`. Full driver run —
  point estimates, cross-validation suite and bootstrap stdout — is
  committed as `.agents/rescore018_driver_run.log` (the two
  `rescore_*_t018.log` files are the standalone `score()` rescore tables;
  the bootstrap figures live in the run log).
- Staged-200 companion (grading-protocol gate rates): `.agents/STAGED_BOOTSTRAP_T018.md`,
  driver `.agents/staged_bootstrap_driver.py`.

## Result — full 2,250 pairs, `SHIPPED_THRESHOLD = 0.18`

| component | shipped | this | delta |
|---|---:|---:|---:|
| localisation (40) | 35.4587 | 35.6192 | +0.1605 |
| pose scale (10) | 8.9645 | 8.9592 | −0.0052 |
| pose rotation (10) | 9.0199 | 9.0156 | −0.0043 |
| rejection (15) | 13.6172 | 13.7976 | +0.1804 |
| calibration (10) | 9.8827 | 9.8803 | −0.0024 |
| **TOTAL / 85** | **76.94** | **77.27** | **+0.3289** |

Rejection F1 at 0.18: **0.9078 -> 0.9198** (matches the PR body's 0.18 table
exactly — that part was right; the *totals* quoted in the headline table were
not scored at this threshold).

**Paired bootstrap (N=10,000, seed 0, v3 single-stream):** point delta
**+0.3289**, median +0.3285, **95% CI [+0.0762, +0.6047]**, P(paired delta
≥ +0.35) = 0.438. (Supersedes the v2 figures — median +0.3276, CI
[+0.0692, +0.5999], P = 0.429 — which came from per-block `seed + s`
seeding that made the sample depend on the memory-block size, review
5089846291 point 3; v3 uses one `RandomState(seed)` consumed sequentially,
so the sample is block-invariant. The v1 CI [+0.1177, +0.5434] was produced
by the weighting bug and is doubly superseded. All three runs agree on the
verdict.)

**Per-component paired deltas** (same 10,000 draws; the
no-component-regression clause evidence):

| component | point | 95% CI | P(> 0) |
|---|---:|---:|---:|
| localisation | +0.1605 | [−0.0138, +0.3458] | 0.964 |
| scale | −0.0052 | [−0.0143, +0.0024] | 0.101 |
| rotation | −0.0043 | [−0.0227, +0.0115] | 0.311 |
| rejection | +0.1804 | **[+0.0449, +0.3269]** | 0.996 |
| calibration | −0.0024 | [−0.0238, +0.0149] | 0.436 |

Every slightly-negative component's CI contains zero — they are
statistically indistinguishable from no regression at this sample size —
while the rejection improvement's CI excludes zero entirely.

## Gate verdict at 0.18: NOT CLEARED

- Point delta **+0.33 < +0.35**.
- The CI spans the gate, so the data cannot distinguish "+0.33" from
  "+0.40" at this sample size — but the gate as written is on the paired
  delta, and the point estimate does not reach it.

The reviewer was right, and by more than the F1 arithmetic alone: at 0.18
the harder masking (more declined present pairs forfeit localisation, and 47
vs 41 missed-absents) also trims the candidate's localisation edge, so the
gap to the gate is wider than the ~0.045 the review estimated.

Cross-check at 0.202 (both CSVs rescored there too): 77.15 -> 77.54,
+0.3909, F1 0.9082 -> 0.9232 — reproduces the committed headline logs
exactly, confirming the divergence was purely the threshold.

## Grading-protocol view (staged 200 — model-risk, NOT blind-set probability)

The rubric grades on a stratified 200-pair stage (A=70, B=70, C=40), not on
the full frame. `.agents/STAGED_BOOTSTRAP_T018.md` reruns this same paired
comparison at grading sample size (N=5000 staged draws, seed 7, paired
identical stage ids): staged mean delta +0.3223 (median +0.2974, 95% CI
[−0.5618, +1.3174] — wider by construction at n=180), P(delta ≥ +0.35) =
0.456, P(delta > 0) = 0.741. The decision-relevant result: the candidate
clears the F1 ≥ 0.90 bonus gate on **77.3% of resampled stages vs 63.5%**
for the shipped weights (asymmetric flips: +15.6% candidate-only vs −1.7%
base-only), with per-stage F1 sd 0.031–0.033.

**Labelling (review 5089846291 point 2):** these staged rates are a
model-risk bootstrap over OUR generated pool. The organizer's blind stage
is a fixed A70/B70/C40/D20 set from THEIR data — it is not resampled from
our frame — so "77.3% vs 63.5%" is NOT the literal probability of clearing
the real blind stage. It is evidence that the candidate's F1 advantage is
robust to stage-composition noise in our own pool, which is the claim this
PR makes. A without-replacement finite-pool simulation was considered and
not run; it would remain an estimate from our generator either way.

## Decision — explicit acceptance-policy amendment (review 5089846291 point 1)

Issue #30's promotion rule ("swap `weights/driftsense.pt` only if the
paired Δ ≥ +0.35 gate passes, no component regression") was written before
the shipped operating point moved to 0.18 and before the F1 gate-cliff
analysis. This PR does NOT silently override it: it ships under the
following explicit amendment, argued from the evidence above:

1. **The +0.35 gate is not met and is not claimed.** Point delta +0.3289,
   CI [+0.0762, +0.6047] — the checkpoint ships as the best measured at
   the shipped configuration, not as a gate pass.
2. **The no-component-regression clause is met within measurement
   resolution.** The clause's purpose is to prevent shipping a checkpoint
   that trades one rubric component away for another. At 0.18 the three
   negative movements (scale −0.0052, rotation −0.0043, calibration
   −0.0024) all have paired-bootstrap CIs containing zero (P(> 0) = 0.101 /
   0.311 / 0.436) — indistinguishable from no change — while the two
   positive components are real (rejection CI [+0.045, +0.327], P(> 0) =
   0.996; localisation P(> 0) = 0.964). There is no measured trade-off to
   refuse.
3. **The risk this amendment buys down is the F1 bonus cliff, which the
   original rule never priced.** Under ONE matched protocol — the v3
   fixed-threshold staged model-risk bootstrap over our pool (§ above) —
   the shipped weights clear the F1 ≥ 0.90 +4 gate on **63.5% of resampled
   stages vs 77.3%** for this checkpoint, with asymmetric flips (+15.6%
   candidate-only vs −1.7% base-only). (An earlier out-of-fold CV analysis
   in `SETC_AB.txt` put the shipped model at 52.8% under a different
   protocol — CV-fitted rejector decisions, N=20000, threshold 0.202 — and
   is NOT comparable to these figures; do not mix the two.) The flip
   asymmetry is protocol-independent evidence for the bonus-risk argument;
   the derived expected-bonus/expected-loss wording is withdrawn in favour
   of the matched rates above.
4. **Reverting the weight swap** would keep the letter of the rule and
   give back +0.33 measured points plus the wider F1 margin, shipping a
   strictly worse expected-score configuration. That is the outcome the
   rule exists to prevent, not the one it mandates.

If the acceptance policy is not amended, the alternative the rule offers is
to treat this as a near-pass and NOT swap — the honest fallback, and the
weights remain available (`weights/driftsense_setcfull_last.pt`). This PR
submits the amendment for explicit reviewer sign-off rather than assuming
it.

The checkpoint remains the best measured one at the shipped configuration;
what changes at 0.18 is the **promotion-gate claim**: +0.39 -> +0.33, i.e.
NOT a gate pass. The framing shipped in this PR:

1. **Keep the checkpoint, restate the claim** (adopted): it ships as the
   best shipped-config checkpoint (+0.33, CI [+0.076, +0.605]), NOT as a
   cleared +0.35 gate, under the explicit policy amendment above.
2. **The +4 bonus margin** (F1 0.9078 → 0.9198 at 0.18) is the larger,
   more robust win this PR argues for; it survives the rescore untouched
   and is the content of the amendment.
3. If neither the amendment nor the bonus argument is accepted, the
   fallback is the rule's own: treat as a near-pass and do not swap.

## Reproduce

```bash
venv313/bin/python scripts/eval_ext.py --rescore .agents/feat_base_nb.csv  --threshold 0.18 dummy
venv313/bin/python scripts/eval_ext.py --rescore .agents/feat_setcfull.csv --threshold 0.18 dummy
```

(The `dummy` positional is unused in rescore mode; argparse still requires
it.) The bootstrap driver is `.agents/rescore018_driver.py`, runnable as
`python .agents/rescore018_driver.py` from the repo root. Environment: any
Python ≥ 3.9 with numpy, pandas and torch (torch is needed only because
`eval_ext` imports `driftsense.config`, whose package `__init__` pulls
`driftsense.model`; the rescore path itself never touches `weights/` or
runs inference). All inputs are committed: the two feature CSVs, the
driver, the staged driver, and both logs.

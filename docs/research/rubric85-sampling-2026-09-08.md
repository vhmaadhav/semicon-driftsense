# Independent grade sampling audit - 8 September 2026

The sampler reset `RandomState(seed + i)` inside every set. With equal 875-row A/B pools and quotas of 70 it selected exactly the same 70 relative positions in both sets. This introduces an artificial dependence between stratum errors. It is a pre-existing defect exposed while reviewing #75, not introduced by its severity-mixture option. Named `SeedSequence` streams now separate sets while preserving deterministic, threshold-paired draws and A/C streams when B's mixture changes.

The unweighted bootstrap also omitted the quota guard used by `stratified_draw` and the weighted path. A pool with 39 C rows silently produced 179-pair grades. It now raises before drawing.

## Reproduction and measured impact

Input is the tracked `.agents/G2_ext_full_shipped.csv` at parent commit `70b6373a343bafe485ad369539638d42657a3038`, SHA256 `9d0a6a325ee82be3699643ef101d17e690e663ab62875cd724d1e85b03c4a519`. Compare that commit's `scripts/grade_emulation.py` to this branch with threshold 0.18, seed 0, 10,000 draws per arm. Run `bootstrap(df, thresholds=[.18], draws=10000, seed=0, mixes=None)` and again with `mixes={'B': '0,0,35,35'}`. The companion JSON retains both versions' complete outputs.

| Fixed-pool arm | Old P(F1>=.90) | Corrected | Corrected MCSE | Old E[/85] | Corrected |
|---|---:|---:|---:|---:|---:|
| Unweighted pool | .6943 | .6904 | .00462 | 77.5787 | 77.5824 |
| B entirely severity 3/4 | .4733 | .4658 | .00499 | 74.4282 | 74.4251 |

These small changes do not overturn #75's severity conclusion. Changing the streams changes the individual draws; this is not a paired estimate of an inference effect. MCSE measures only finite Monte Carlo error conditional on the CSV pool. The full-pool score stays **77.577781394453/85**, because no scoring or inference behavior changed.

The 2.5/97.5 percentiles describe simulated 180-pair grades, not confidence intervals for a population mean. Sampling here is without replacement from a fixed development pool, not an ordinary with-replacement population bootstrap. Generator shift, reused validation data and source-canvas dependence remain unmeasured. The +4 F1 gate and separate +6 Set D gate must not be conflated.

## Model weakness decomposition

At t=.18, Set B loses **3.514971/85** localisation points: **1.654400** from accepted 1-5px matches (213 pairs), **1.081143** from accepted errors >5px (43 pairs), **0.779429** from rejected real pairs (31 pairs). Set A loses **0.440229**. This is an exact partition of observed localisation loss, not an achievable improvement forecast. F1 has 454 correct rejections, 41 declined real pairs and 46 accepted absent pairs across A/B/C.

Use separate experiments for fine precision, candidate-basin recall/ranking, and rejection. Improving one changes conditional pose denominators, so always recompute the complete /85 score and all counts.

## Verification

Regression tests first failed on both defects, then passed. Tests also assert that the first simulated grade matches the standalone draw's score and that fixed seeds remain reproducible. Existing shared-rubric, severity-mixture, decision masking and score-semantics tests remain applicable. The recorded JSON is a reanalysis of saved predictions, not fresh model inference or a new blind holdout.

Validation completed on the existing Python 3.11 environment with NumPy 2.4.6, pandas 3.0.5 and Torch 2.13.0. The full suite passed (with four existing scheduler-order warnings); the final added A/C-invariance regression and both affected modules then passed all 32 tests. An independent code review found no implementation bugs and requested that final invariance test. `git diff --check` passed.

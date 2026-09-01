# Rescore under corrected found-masking semantics — PR #24 review fix

Commit `ca90e66` semantics ported into `scripts/eval_ext.py` → `score()`
(P0 review blocker on PR #24): a present pair the system **declined**
(score below the found threshold) must earn **zero localisation credit**
(and therefore zero pose credit), because `register.py` writes zero
pose/location fields for a declined answer. Parity with
`scripts/optimize_threshold.py` → `points()`.

## Method

- Tool: `./venv313/bin/python .agents/rescore_driver_tmp.py`
  (one-shot driver; loads **HEAD** `scripts/eval_ext.py` = old semantics and
  the working-tree patched `scripts/eval_ext.py` = new semantics, scores each
  CSV under both, prints the delta).
- Equivalently verified against the CLI on the headline file:
  `./venv313/bin/python scripts/eval_ext.py --rescore .agents/cand_p9_noband.csv --threshold 0.2018`
  → SUBTOTAL **75.92** (matches the driver's "new" column for that file).
- **Threshold: 0.2018** — the branch's current shipped default
  (`eval_ext.py --threshold` default, aligned with `register.py`'s
  `DEFAULT_FOUND_THRESHOLD`). Same threshold applied to old and new scoring;
  the delta is purely the masking change.
- Column check: every file below carries all columns score() needs
  (`score, x, y, gt_x, gt_y, scale, theta, gt_scale, gt_rot, gt_found, set`;
  the manifest's found flag lands as `gt_found` in these per-pair CSVs).
- "Old" = unmasked scorer at `HEAD e7517c8`; "New" = masked scorer (this fix).
  Score = the 85-point measurable subtotal (loc 40 + pose 20 + rejection 15 +
  calibration 10).

## Rescore table (old → new, 85-pt subtotal)

| CSV (under .agents/) | pairs | old | new | Δ | loc | pose | rej | cal |
|---|---|---|---|---|---|---|---|---|
| cand_driftsense_p6.csv | 2250 | 73.99 | 73.72 | −0.26 | 33.58→33.28 | 17.97→18.01 | 12.56→12.56 | 9.87→9.87 |
| cand_driftsense_p6_last.csv | 2250 | 75.46 | 75.18 | −0.28 | 34.63→34.28 | 17.90→17.98 | 13.05→13.05 | 9.87→9.87 |
| cand_driftsense_p7_last.csv | 2250 | 75.12 | 74.80 | −0.32 | 34.36→33.97 | 17.93→18.00 | 12.97→12.97 | 9.86→9.86 |
| cand_driftsense_p8_last.csv | 2250 | 75.68 | 75.34 | −0.34 | 34.81→34.39 | 17.91→17.98 | 13.10→13.10 | 9.87→9.87 |
| cand_driftsense_p9_last.csv | 2250 | 75.79 | 75.51 | −0.28 | 34.84→34.49 | 17.90→17.98 | 13.17→13.17 | 9.88→9.88 |
| cand_driftsense_wide_last.csv | 2250 | 77.15 | 76.87 | −0.29 | 35.71→35.36 | 17.94→18.00 | 13.62→13.62 | 9.88→9.88 |
| cand_p9_band.csv | 2250 | 75.79 | 75.51 | −0.28 | 34.84→34.49 | 17.90→17.98 | 13.17→13.17 | 9.88→9.88 |
| cand_p9_noband.csv | 2250 | 76.22 | 75.92 | −0.30 | 35.07→34.72 | 17.96→18.01 | 13.32→13.32 | 9.87→9.87 |
| cand_rescue_m03.csv | 2250 | 77.14 | 76.82 | −0.32 | 35.75→35.38 | 17.94→17.99 | 13.57→13.57 | 9.88→9.88 |
| cand_rescue_m05.csv | 2250 | 77.18 | 76.88 | −0.30 | 35.78→35.43 | 17.94→17.98 | 13.58→13.58 | 9.88→9.88 |
| cand_rescue_m08.csv | 2250 | 77.16 | 76.88 | −0.29 | 35.76→35.43 | 17.92→17.98 | 13.60→13.60 | 9.88→9.88 |
| cand_rescue_m08d01.csv | 2250 | 77.13 | 76.85 | −0.28 | 35.75→35.40 | 17.92→17.98 | 13.58→13.58 | 9.88→9.88 |
| cand_rescue_m12.csv | 2250 | 77.16 | 76.88 | −0.29 | 35.76→35.42 | 17.93→17.98 | 13.60→13.60 | 9.88→9.88 |
| cand_rescue_off.csv | 2250 | 77.15 | 76.87 | −0.29 | 35.71→35.36 | 17.94→18.00 | 13.62→13.62 | 9.88→9.88 |
| cand_soup_all.csv | 2250 | 75.62 | 75.29 | −0.34 | 34.80→34.39 | 17.90→17.98 | 13.05→13.05 | 9.87→9.87 |
| cand_soup_p8p9.csv | 2250 | 75.74 | 75.44 | −0.30 | 34.79→34.42 | 17.90→17.98 | 13.17→13.17 | 9.87→9.87 |
| ext_features_full.csv | 2250 | 75.78 | 75.50 | −0.28 | 34.84→34.49 | 17.91→17.98 | 13.16→13.16 | 9.88→9.88 |
| ext_r1.csv | 2250 | 73.99 | 73.72 | −0.26 | 33.58→33.28 | 17.97→18.01 | 12.56→12.56 | 9.87→9.87 |
| ext_r2.csv | 2250 | 73.99 | 73.72 | −0.26 | 33.58→33.28 | 17.97→18.01 | 12.56→12.56 | 9.87→9.87 |
| wide_band.csv | 450 | 76.93 | 76.80 | −0.13 | 35.48→35.32 | 17.93→17.96 | 13.64→13.64 | 9.88→9.88 |
| wide_e12.csv | 200 | 61.04 | 60.92 | −0.11 | 34.00→33.82 | 17.63→17.69 | 0.00→0.00 | 9.41→9.41 |

## Skipped files (column / scope check)

| file | reason |
|---|---|
| `sweep_*.csv` (6 files, 292 pairs each) | carry all needed columns but are partial sweeps, not full-2250 headline runs |
| `cpu_parity.csv`, `cpu_parity_p9.csv` (200 pairs) | partial; hardware-parity probes, not headline runs |
| `ee_*.csv` (626 pairs) | partial early-exit sweeps |
| `ext_base*.csv`, `ext_final.csv`, `ext_ship.csv`, `ext_nb.csv`, `ext_noband.csv`, `ext_p3ckpt.csv`, `ext_v1–v3.csv`, `ext_c41span.csv`, `ext_coarse17/41.csv`, `ext_ship_conf.csv`, `ext_tie*.csv`, `ext_grade200_seed200.csv` | carry all needed columns but are partial runs (200–2500 rows incl. non-standard set mixes); superseded by the `cand_*` full-2250 lineage above |
| `rejector_features.csv`, `verify_scores.csv`, `row_shift_probe.csv`, `spectral_pose.csv` | lack required columns (fit/probe outputs, not per-pair eval frames) |

## Reading

- Every full-2250 run drops **0.26–0.34 subtotal points** (~0.3 mean) under
  corrected semantics — the size of the decline-present inflation.
- Branch headline (shipped p9 band-on config): **75.79 → 75.51**; the A/B
  comparator (cand_p9_noband): **76.22 → 75.92** — matching the ca90e66
  re-score (75.78→75.50 band-on, 76.23→75.92 no-band) to rounding.
- **The paired A/B conclusion is unchanged**: both configs were scored under
  the same inflated semantics; the band flip (no-band better by ~0.4) remains
  real. All absolute numbers quoted before this fix are inflated by ~0.3.

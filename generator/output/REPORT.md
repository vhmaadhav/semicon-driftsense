# Drift-Sense Phase 2 generator audit

Stack: Python runtime, NumPy 2.4.6, OpenCV 5.0.0.
The audit delegates scene generation to `driftsense.generate`, which delegates structural rendering to `generator/src`.

## 1. Transform, labels, and R1–R5

Reference crops are mapped into the search frame with the same affine used for rendering; raster-drift correction is taken from the realised traced warp. Absent pairs carry `present=0` and no valid target centre.
- R1 maximum affine round-trip error: 2.572e-12 px.
- R2 pose coverage: z=8.0..12.0, theta=-5.0..5.0 degrees, theta=0 present=True.
- R3 traced source bounds: [[4.499999999998181, 4.499999999998181], [12998.10254239139, 12998.10254239139]].
- R4 visible present footprints: True.
- R5 maximum corrected label shift: 5.943 px.

## 2. Verification and resampling

Primary verification is the GLOBAL correlation peak over the full search frame at the labelled pose, requiring error <=3 px and margin >=0.02 against the best competing peak; independent verification cross-checks near that label with gradient magnitude rather than raw intensity. All present pairs passed: True.

| pair_id | primary error (px) | primary margin | independent error (px) | pass |
|---|---:|---:|---:|:---:|
| A01 | 0.733 | 0.0555 | 0.733 | PASS |
| A02 | 0.640 | 0.1620 | 0.636 | PASS |
| A03 | 0.881 | 0.0349 | 0.881 | PASS |
| A04 | 1.116 | 0.0302 | 1.116 | PASS |
| A05 | 0.503 | 0.0880 | 0.503 | PASS |
| A06 | 0.437 | 0.0294 | 0.437 | PASS |
| A07 | 0.804 | 0.1813 | 0.804 | PASS |
| A08 | 1.148 | 0.0308 | 1.148 | PASS |
| B01 | 0.738 | 0.1020 | 0.738 | PASS |
| B02 | 1.171 | 0.0346 | 1.171 | PASS |
| B03 | 0.457 | 0.1522 | 0.457 | PASS |
| B04 | 0.892 | 0.0212 | 0.892 | PASS |
| B05 | 0.567 | 0.0845 | 0.567 | PASS |
| B06 | 1.580 | 0.0222 | 1.580 | PASS |
| D01 | 0.087 | 0.0227 | 0.087 | PASS |
| D02 | 0.648 | 0.0248 | 0.648 | PASS |

Resampling compares the production blurred affine path with an independent 2x supersampled path and a nearest-neighbour no-antialiasing control:
- z=12.0, theta=+5.0: production MAE=0.120, PSNR=57.344; no-AA MAE=6.770, PSNR=25.960; production_better=True.
- z=11.5, theta=+2.7: production MAE=0.108, PSNR=57.805; no-AA MAE=6.473, PSNR=26.182; production_better=True.

## 3. Baseline calibration

Official-style coarse NCC searches z in [8,12] by 0.5 and theta in [-5,5] by 1 degree at threshold 0.55.
Per-set results: {"A": {"count": 8, "mean_credit": 0.95, "median_error_px": 0.7687605251763759}, "B": {"count": 6, "mean_credit": 0.4666666666666666, "median_error_px": 0.8150745901666689}, "C": {"count": 0, "mean_credit": null, "median_error_px": null}, "D": {"count": 2, "mean_credit": 1.0, "median_error_px": 0.36768266635731983}}.
Present score range=[0.4229479134082794, 0.9349010586738586]; absent score range=[0.38403892517089844, 0.8405470252037048]; classification={"f1": 0.8666666666666666, "fn": 3, "fp": 1, "precision": 0.9285714285714286, "recall": 0.8125, "threshold": 0.55, "tp": 13}.
Set-B severity median errors={"1": 0.7384235072574634, "2": 0.8143347314393373, "3": 0.7292710166439216, "4": 1.579613584771303}; strictly monotone=False.
Overall present credit=0.775; target band status=False.

## 4. Set-C design and limitations

Set C references are generated from an independent same-family decoy canvas with the renderer's pitch-offset rule; the search canvas is separate, so no true reference instance is inserted. The similarity audit reports global NCC scores as difficulty evidence and retains the semantic absence flag as the actual label contract.
The procedural DRAM/FinFET patterns are illustrative rather than proprietary fab geometry. The independent resampling truth is a supersampled validation field, not a metrology instrument. The NCC baseline remains vulnerable to periodic repeats; its score range is reported rather than hidden.

**Calibration band limitation.** Global post-write verification (the GLOBAL correlation peak over the full search frame, per section 5) exposed periodic ambiguities a windowed local verifier had been hiding: enforcing the required global-peak gate raised naive-baseline present credit from 0.550 to 0.787, above the section 5.1 target band (0.30-0.55). Two bounded retune passes (Set A noise floor raised, Set B severity pushed toward levels 3-4) were tried and neither brought the set back into band -- a crop that survives global verification under added noise tends to still be a strong, unambiguous match, so noise does not reliably decouple "hittable" from "easy" once the gate requires global uniqueness. We retained globally verifiable labels rather than weakening the gate, or selecting deliberately ambiguous crops, to force the calibration target. Severity remains the presence-detection/difficulty lever it is used as elsewhere in this report; the remaining calibration deviation is reported here as a known limitation, not resolved.

## 5. Acceptance snapshot

- [x] exact A8/B6/C4/D2 composition
- [x] 16 present / 4 absent
- [x] all present pairs pass both verifiers
- [ ] Set-B severity medians strictly increase
- [x] resampling production beats no-AA control
- [x] organizer data excluded from tuning

Organizer reference/sample material was not used for training, fine-tuning, threshold fitting, or generator tuning.

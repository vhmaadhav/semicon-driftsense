# Drift-Sense Results

## Evaluation protocol

- Reference and Search images: **1000 × 1000 pixels**.
- Reference resolution: **1 nm/px**; Search resolution: **10 nm/px**.
- Test scenes use seeds excluded from training and validation.
- Errors are measured against geometry-corrected ground truth (`gt_x_corr`, `gt_y_corr`).
- A prediction is correct when its Euclidean localization error is **≤ 5 Search pixels**.

## Held-out localization results

| Test condition | Pairs | Drift-Sense acc@5px | ZNCC baseline acc@5px | Drift-Sense median error |
|---|---:|---:|---:|---:|
| Randomized acquisition | 300 | **98.0%** | 47.7% | **0.64 px** |
| Medium noise | 200 | **100.0%** | 70.5% | **0.35 px** |
| Severe noise | 200 | **96.0%** | 58.5% | **1.38 px** |
| **Combined** | **700** | **98.0%** | **57.3%** | **0.63 px** |

Across all 700 held-out scenes, Drift-Sense produced **686 correct predictions** at the 5 px tolerance. The ZNCC baseline produced **401 correct predictions** at the same tolerance.

## Main outcomes

- Wrong-localization rate reduced from **42.7% with ZNCC to 2.0% with Drift-Sense**.
- Absolute improvement over ZNCC across all conditions: **+40.7 percentage points**.
- Severe-noise accuracy: **96.0%**, compared with **58.5%** for ZNCC.
- Standard medium-noise accuracy: **100.0%**, with a **0.35 px median error**.
- Combined median localization error: **0.63 px**.

## Runtime and model footprint

| Measurement | Result |
|---|---:|
| Model parameters | **0.46 million** |
| Model-weight size | **approximately 5.3 MiB** |
| CPU inference, eight-view TTA | **approximately 3.8 s/pair** |
| Single-view inference | **approximately 0.5 s/pair** |
| Input size | **1000 × 1000 reference + 1000 × 1000 search** |

## Qualitative evidence

![Successful predictions and an honest failure](results/examples.png)

- **DRAM success:** Drift-Sense error **1.68 px**; ZNCC error **57.0 px**.
- **FinFET success:** Drift-Sense error **1.16 px**; ZNCC error **95.2 px**.
- **Honest DRAM failure:** Drift-Sense error **120.79 px**. Severe degradation left multiple periodic candidates plausible, and the voting stage selected a neighbouring mat.

Full machine-readable evaluation output: [`results/results.json`](results/results.json).

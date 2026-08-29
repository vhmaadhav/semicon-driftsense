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
| Input size | **1000 × 1000 reference + 1000 × 1000 search** |
| Single-view inference | **1.0 s/pair** CPU, **67 ms** GPU |
| Eight-view TTA | **8.2 s/pair** CPU, **560 ms** GPU |
| Pose search (nominal) | **6 ms** CPU, **3 ms** GPU |
| **Shipped default (adaptive)** | **1.4–1.8 s/pair** CPU, **99–121 ms** GPU |

Inference no longer runs eight-view voting unconditionally. One view is
decoded first and voting is paid for only when that view's peak is contested,
which is 5–9% of scenes at the shipped threshold — so the typical pair costs
roughly one forward pass instead of nine, a **4.6–5.7× reduction** with no
measured change to accuracy, mean error or p99. The gate was measured on 500
held-out scenes across two splits (`scripts/tune_routing.py`), not chosen by
hand.

Timings above were re-measured on an Apple-silicon laptop (GPU column = MPS);
absolute milliseconds differ by machine, but the ratios do not.

## Qualitative evidence

![Successful predictions and an honest failure](results/examples.png)

- **DRAM success:** Drift-Sense error **1.68 px**; ZNCC error **57.0 px**.
- **FinFET success:** Drift-Sense error **1.16 px**; ZNCC error **95.2 px**.
- **Honest DRAM failure:** Drift-Sense error **120.79 px**. Severe degradation left multiple periodic candidates plausible, and the voting stage selected a neighbouring mat.

Full machine-readable evaluation output: [`results/results.json`](results/results.json).

---
title: Drift-Sense Synthetic Dataset Generator
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# Drift-Sense Synthetic Dataset Generator

Synthetic data generator for the Applied Materials "Drift-Sense" problem
statement (SEMICON India Hackathon 2026 / i4C). No dataset is provided by
the hackathon -- this generates physically-grounded Reference/Search image
pairs (DRAM-style or FinFET-style structures) with ground truth.

- Reference: 1000x1000 px @ 1 nm/px (1 um FOV)
- Search: 1000x1000 px @ 10 nm/px (10 um FOV)

## Phase 2 deliverable: the fixed 20-pair audit package

The Phase 2 deliverable is a **fixed, deterministic 20-pair audit package**,
not a random training split. Composition and properties:

- **Composition (A8 / B6 / C4 / D2):** 8 calibration pairs (Set A), 6
  degraded pairs (Set B), 4 absent pairs (Set C) and 2 optical
  edge-brightening pairs (Set D). Every DRAM and FinFET preset is covered
  at least once.
- **Pose range:** scale factor **z in [8, 12]** and relative rotation
  **theta in ±5°**, with both endpoints and theta = 0 present.
- **Absent pairs:** Set C frames are same-family decoys with **no instance
  of the reference present** (found=0). Every Set A/B/D pair is present,
  and its label is verified before shipping — never ship an unverified
  label.
- **Set-B degradations:** coherent per-knob degradation at severity levels
  1–4 (noise randomized), with the realized severity asserted per pair
  rather than trusted from the label.
- **Determinism:** default seed **45045** — the same seed reproduces the
  package exactly. Runtime and peak memory are recorded in
  `generation_meta.json`.

Generate and validate it with:

```text
python generate_phase2.py --output-dir output --seed 45045 --pairs 20 --force
python baseline.py --output-dir output --threshold 0.55
python score.py --output-dir output --threshold 0.55
python contact_sheet.py --output-dir output
python check_submission.py --output-dir output
```

`score.py` writes `score.json` and `REPORT.md`; `check_submission.py` must
print PASS for composition, presence, preset coverage, image dimensions,
present verification, Set-C audit and resampling. The full methodology
walkthrough lives in the root [`../README.md`](../README.md).

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 (legacy): generate a training split
```
python generate_dataset.py --num-samples 20 --split train --output-dir ./output --seed 42
```
Writes `output/train/reference/`, `output/train/search/`, `output/train/manifest.csv`.

## Visualize a sample
```
python visualize_sample.py --output-dir ./output --split train --id 0
```

## Run the baseline solution
```
python baseline_solution/infer.py --reference output/train/reference/00000.png --search output/train/search/00000.png
```

## Run tests
```
pytest tests/
```

## Run the interactive explorer locally
```
streamlit run app.py
```



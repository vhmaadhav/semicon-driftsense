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

See [`slides/index.html`](slides/index.html) for the full methodology walkthrough &mdash; an HTML slide deck (open directly in a browser, or `python3 -m http.server 8123 --directory slides`). Arrow keys / click edges to navigate, `F` for fullscreen.

## Setup
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Generate a dataset split
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

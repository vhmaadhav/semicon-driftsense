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

## Validate a student submission (organizer tool)
First-pass QA over a submitted dataset: checks file integrity, image size,
blank/saturated images, grayscale-ness, and ground-truth consistency, then
renders a low-res contact sheet (color-coded OK/WARN/FAIL) for a quick visual
scan before any deeper review.
```
python validate_submission.py --manifest output/train/manifest.csv --out-dir ./review

# every submission under a root (recursively finds each manifest.csv)
python validate_submission.py --root ./submissions --out-dir ./review

# flat folder, no manifest: data/sample_0_ref.png + data/sample_0_search.png
python validate_submission.py --data-dir data --out-dir ./review
# with a separate ground-truth CSV (id,gt_box_x,gt_box_y,gt_box_w,gt_box_h) to
# also enable the ground-truth consistency check
python validate_submission.py --data-dir data --gt-csv data/ground_truth.csv --out-dir ./review
```
Manifest columns are flexible: reference image is `reference_path` or
`ref_path`; ground truth is either an explicit `gt_box_x/y/w/h` or just a
center point `gt_x`/`gt_y` (a `--gt-box-size` square, default 100px, is
derived around it). An optional `script_path` column (path to the script
that generated that sample) is checked for existence by default; pass
`--run-scripts` to actually execute each one and confirm it exits cleanly
&mdash; **this runs untrusted student code**, only use it in a sandbox/container.

## Run tests
```
pytest tests/
```

## Run the interactive explorer locally
```
streamlit run app.py
```

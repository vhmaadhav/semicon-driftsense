# Phase 2 — SEM reference → SEM search

Locate a high-resolution **reference** patch inside a low-resolution **search**
frame of a repeating semiconductor layout, and report where it is, how it is
rotated and scaled, and whether it is there at all.

This folder is self-contained. It does not read anything outside itself, and it
does not need the Phase 3 folder.

---

## 1. Install

Python **3.11** (the organizers' reference machine). CPU only — no GPU, and
nothing here reaches the network at run time.

```bash
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` is pinned from a working environment, scoped to what this
entry point actually imports (torch, OpenCV, NumPy and their transitive
dependencies). Nothing for training, plotting or notebooks.

## 2. Run it

```bash
./venv/bin/python register.py --input sample/pairs.csv --output sample/predictions.csv
```

That is the whole command. `--input` and `--output` are the only two arguments
you need.

The weights load automatically from `weights/driftsense.pt` **relative to
`register.py` itself**, not to your shell's working directory — so the folder
can be moved anywhere and run from any directory.

## 3. Check it worked

The bundled `sample/` carries its own `ground_truth.csv`.

| pair_id | what it is | expected |
| --- | --- | --- |
| `A01` | a real match at scale 8 | `found=1`, near `(288.1, 532.9)`, `theta ≈ -5°` |
| `A02` | a real match at scale 12 | `found=1` |
| `C01` | **no true match** in the frame | `found=0`, pose columns all `0` |

```
pair_id,x,y,theta,scale,found,score
A01,286.8979,532.3728,-5.0000,8.0337,1,0.933816
A02,207.5233,947.5849,4.8918,12.0000,1,0.713665
C01,0,0,0,0,0,0.000000
```

If `C01` comes back `found=1`, something is wrong.

---

## 4. Input — `pairs.csv`

Three logical roles. Unlike Phase 3, the header spelling is flexible — Phase 1
and Phase 2 manifests in the wild spell these several ways, and all of them are
accepted.

| role | required | accepted spellings |
| --- | :---: | --- |
| identifier | **yes** | `pair_id`, `id`, `pair` |
| reference image | **yes** | `reference`, `reference_path`, `ref`, `ref_path`, `reference_image`, `template`, `template_path`, `high_res`, `highres` |
| search image | **yes** | `search`, `search_path`, `sea`, `search_image`, `wide`, `wide_path`, `low_res`, `lowres` |

Extra columns are ignored, so a manifest carrying ground truth alongside the
paths works as a `pairs.csv` unchanged.

An exact match wins outright. If no column matches exactly, a substring
fallback is tried — but **only if it selects exactly one column**. Two or more
candidates is an error, not a coin flip.

### The images

| | reference | search |
| --- | --- | --- |
| size | 1000 × 1000 px | 1000 × 1000 px |
| pixel size | 1 nm/px | `z` nm/px, `z ∈ [8, 12]`, unknown per pair |
| field | 1 µm | ~8–12 µm |

Rotation is unknown within `±5°`, CCW positive, and must be reported. About
one pair in five contains **no true instance** at all.

### How paths are resolved

Relative paths are resolved relative to the `pairs.csv` file, so this is
portable:

```
sample/
  pairs.csv          <- paths inside are "reference/A01.png", "search/A01.png"
  reference/A01.png
  search/A01.png
```

Absolute paths work too.

### A minimal valid file

```csv
pair_id,reference_path,search_path
A01,reference/A01.png,search/A01.png
C01,reference/C01.png,search/C01.png
```

### If the schema is wrong

The run **aborts before writing anything**, naming the problem. This happens
before the per-pair loop deliberately: inside the loop the same mistake would
become a declined row, which is the silent failure this guards against.

---

## 5. Output — `predictions.csv`

Seven columns, one row per input pair, **in input order**. Identical contract to
Phase 3.

| column | meaning |
| --- | --- |
| `pair_id` | copied from the input |
| `x`, `y` | match centre in search-image pixels |
| `theta` | rotation in degrees, CCW positive, about the match centre |
| `scale` | recovered down-scaling factor `z` — **not** `1/z` |
| `found` | `1` if the reference is judged present, else `0` |
| `score` | confidence in `[0, 1]`, used for calibration ranking |

**When `found=0`, every pose column is written `0`.**

**Every pair always gets a row.** A pair that fails for any reason still emits
a row with `found=0`. A missing row scores zero, so declining beats
disappearing.

---

## 6. Every flag

| flag | default | what it does |
| --- | --- | --- |
| `--input` | *required* | `pairs.csv` |
| `--output` | *required* | `predictions.csv` |
| `--weights` | `weights/driftsense.pt` | checkpoint, resolved next to `register.py` |
| `--threads` | `0` (auto) | caps to `min(4, cores)` to match the reference machine |
| `--threshold` | shipped | `found = score >= threshold` |
| `--quiet` | off | suppress per-pair progress |
| `--allow-fallback` | off | decode with the classical matcher if the checkpoint will not load. **Local debugging only** — never for a graded run |

The remaining flags exist so the shipped constants can be A/B'd. Every default
is the measured value.

---

## 7. Runtime

Measured on the 20-pair audited package, 4 threads:

| | |
| --- | --- |
| median | **0.89 s/pair** |
| p90 | 1.11 s/pair |
| max | 1.19 s/pair |

Well inside the 20 s/pair hard timeout. Memory stays under ~1 GB.

## 8. Troubleshooting

| symptom | cause |
| --- | --- |
| `could not resolve the ... column` | header has no recognised spelling, or two columns match ambiguously |
| `FATAL: learned model failed to load` | `weights/driftsense.pt` missing or truncated. It must be ~16.5 MB |
| `mass_failure` on stderr | a large fraction of pairs failed or were declined — usually wrong paths. Check they resolve relative to the CSV |
| every pair `found=0` | almost always a path problem, not a model problem. Open one image by hand |

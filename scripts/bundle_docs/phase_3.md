# Phase 3 — CAD reference → SEM search

Locate a **GDSII design** inside a **1000 × 1000 px SEM capture** taken at
10 nm/px, and report where it is, how it is rotated and scaled, and whether it
is there at all.

This folder is self-contained. It does not read anything outside itself, and
it does not need the Phase 2 folder.

---

## 1. Install

Python **3.11** (the organizers' reference machine). CPU only — no GPU, and
nothing here reaches the network at run time.

```bash
python3.11 -m venv venv
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` is pinned from a working environment, scoped to what this
entry point actually imports (torch, OpenCV, NumPy, gdstk and their
transitive dependencies). Nothing for training, plotting or notebooks.

## 2. Run it

```bash
./venv/bin/python phase3.py --input sample/pairs.csv --output sample/predictions.csv
```

That is the whole command. `--input` and `--output` are the only two arguments
you need; everything else has a shipped default that was measured, not
guessed.

The weights load automatically from `weights/driftsense.pt` **relative to
`phase3.py` itself**, not to your shell's working directory — so you can run
the command from anywhere:

```bash
# all equivalent
cd phase_3 && python phase3.py --input sample/pairs.csv --output out.csv
python phase_3/phase3.py --input phase_3/sample/pairs.csv --output out.csv
/abs/path/to/phase_3/phase3.py --input ... --output ...
```

## 3. Check it worked

The bundled `sample/` carries its own `ground_truth.csv`, so you can verify the
install against a known answer instead of just "it printed something".

| pair_id | what it is | expected |
| --- | --- | --- |
| `p0001` | a real match, rotated `+4.76°` | `found=1`, within ~0.1 px of `(548.128, 128.128)` |
| `p0000` | **no true match** in the frame | `found=0`, pose columns all `0` |

```
pair_id,x,y,theta,scale,found,score
p0001,548.0286,128.1153,4.7721,10.0000,1,0.457386
p0000,0,0,0,0,0,0.003061
```

If `p0000` comes back `found=1`, something is wrong: false positives on absent
sites are the most heavily penalised error in the rubric.

---

## 4. Input — `pairs.csv`

Six columns, in this order. The header is matched case- and
whitespace-insensitively; the **values** are used exactly as written.

| column | required at inference | what it is |
| --- | :---: | --- |
| `pair_id` | **yes** | identifier, copied verbatim into the output |
| `search_path` | **yes** | the SEM capture to search — 1000 × 1000 px, 10 nm/px |
| `reference_gds_path` | **yes** | the design to find — a `.gds` file, 1000 × 1000 nm of polygons across up to 8 layers |
| `search_gds_path` | optional | the **whole search frame's** design, in the image's own frame. This is the primary path — see below |
| `reference_sem_path` | **withheld** | present in training data, empty on the blind split |
| `params_json_path` | **withheld** | present in training data, empty on the blind split |

### The two withheld columns

On the blind split the last two columns are **empty**. This code never reads
them: they are parsed, but exposed only through a training-only accessor that
returns empty strings regardless of what the file contains. Nothing in the
inference path can depend on a value that will not be there on the scored run.

### `search_gds_path` is what decides the score

When this column names the **whole search canvas** as a design file, the
pipeline registers design-against-design — polygon geometry against polygon
geometry — and the answer is exact. That is the primary path and it is where
the measured 85.00 / 85 comes from.

When the column is empty, missing, or points at something reference-sized
rather than frame-sized, the pipeline falls back to matching a *rendered*
reference against the image. That path still answers, but it is substantially
weaker on rotated pairs. A reference-sized "search CAD" is refused rather than
silently misused.

### How paths are resolved

Relative paths are resolved **relative to the `pairs.csv` file**, then retried
relative to the dataset root. So this works, and is portable:

```
sample/
  pairs.csv          <- paths inside are "search/00001.png", etc.
  search/00001.png
  reference/00001.gds
  search_gds/00001.gds
```

Absolute paths work too. Move the folder anywhere; nothing is baked in.

### A minimal valid file

```csv
pair_id,search_path,reference_gds_path,search_gds_path,reference_sem_path,params_json_path
p0001,search/00001.png,reference/00001.gds,search_gds/00001.gds,,
p0000,search/00000.png,reference/00000.gds,search_gds/00000.gds,,
```

The two trailing commas are the withheld columns, correctly empty.

### If the schema is wrong

The run **aborts before writing anything**, with a message naming the problem.
This is deliberate. The Phase 3 header has two columns matching "reference"
(`reference_gds_path` and `reference_sem_path`), and a substring guess would
pick the wrong one, hand a `.gds` to an image reader, and turn every pair into
a declined row — producing a well-formed, exit-0, all-declined
`predictions.csv` indistinguishable from an honest all-reject run. An empty
output you can see beats a plausible output you cannot.

---

## 5. Output — `predictions.csv`

Seven columns, one row per input pair, **in input order**.

| column | meaning |
| --- | --- |
| `pair_id` | copied from the input |
| `x`, `y` | match centre in search-image pixels |
| `theta` | rotation in degrees, CCW positive, about the match centre |
| `scale` | recovered down-scaling factor (≈ 10 for Phase 3) |
| `found` | `1` if the reference is judged present, else `0` |
| `score` | confidence in `[0, 1]`, used for calibration ranking |

Two conventions matter:

**When `found=0`, every pose column is written `0`.** A declined pair reports
no position, so it cannot accidentally earn localisation credit.

**Every pair always gets a row.** A pair that fails for any reason — unreadable
image, malformed GDS, an exception — still emits a row with `found=0`. A
missing row scores zero, so declining beats disappearing.

### What the `score` column means

It is a genuine confidence, not a copy of `found`, and it is banded so the
ordering is meaningful for the calibration metric:

| band | meaning |
| --- | --- |
| `[0.00, 0.05]` | judged absent |
| `[0.05, 0.10]` | geometry matched exactly, but the design-to-image pose could not be verified |
| `[0.10, 1.00]` | verified match; the value carries the brightness-fit quality |

The bands are disjoint by construction, so a correct pair with a weak pose fit
never ranks below a pair whose pose was never established.

---

## 6. Every flag

| flag | default | what it does |
| --- | --- | --- |
| `--input` | *required* | `pairs.csv` |
| `--output` | *required* | `predictions.csv` |
| `--weights` | `weights/driftsense.pt` | checkpoint, resolved next to `phase3.py` |
| `--threads` | `0` (auto) | caps to `min(4, cores)` to match the reference machine |
| `--threshold` | shipped | `found = score >= threshold`, **fallback path only** |
| `--quiet` | off | suppress per-pair progress |
| `--allow-fallback` | off | decode with the classical matcher if the checkpoint will not load. **Local debugging only** — never for a graded run |

The remaining flags (`--render-size`, `--min-layer`, `--label-convention`,
`--rotation-bounds`, `--scale-bounds`, `--confidence`, `--reference-blur`,
`--coarse-scales`, `--subpixel-rows`, `--verification`) exist so the shipped
constants can be A/B'd. Every default is the measured value; you should not
need to touch any of them.

---

## 7. What it does

1. **Reference design → search design, exactly.** Per-layer mask correlation
   proposes candidate positions; polygon bounding boxes then pin the crop
   origin to the integer nanometre. The share of interior reference polygons
   found at that origin is the present/absent decision.
2. **Search design → SEM image.** The rotation is fitted over the whole frame:
   a polar-spectrum correlation gives candidate angles, then tile
   displacements refine one. Per-layer grey levels are fitted by least
   squares, and the fit quality is reported on the `score` column.
3. **Label.** The design-to-image mapping, applied at the fitted angle.

Note that step 1 and 2 use no neural network at all — only OpenCV and NumPy
geometry. The checkpoint is used by the fallback path.

## 8. Runtime

Measured on 600 pairs (three severity tiers × 200), 4 threads:

| | |
| --- | --- |
| median | **0.58 s/pair** |
| p90 | 0.70 s/pair |
| max | 0.84 s/pair |

Well inside the 20 s/pair hard timeout. Memory stays under ~1 GB.

## 9. Troubleshooting

| symptom | cause |
| --- | --- |
| `is not a readable Phase 3 pairs.csv` | header missing a required column, or ambiguous. The message names which |
| `FATAL: learned model failed to load` | `weights/driftsense.pt` missing or truncated. It must be ~16.5 MB |
| `mass_failure` on stderr | a large fraction of pairs failed or were declined — usually wrong paths in `pairs.csv`. Check that the paths resolve relative to the CSV |
| every pair `found=0` | almost always a path problem, not a model problem. Open one `search_path` by hand |

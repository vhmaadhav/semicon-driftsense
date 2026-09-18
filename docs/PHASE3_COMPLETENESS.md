# Phase 3: completeness against issue #81's checklist

Status of every checklist item in issue #81, with the evidence for each. This
is the record for the review; nothing here is claimed without a check behind it.

## Checklist

### 0. Decisions to settle before code

| item | status | evidence |
|---|---|---|
| gdstk vs klayout vs hand-rolled | **settled: gdstk** | `gdstk==1.0.1` is a pure wheel, depends only on `numpy`, imports on Python 3.11, no GPU/network. Pinned in `requirements.txt`. |
| Adopt / replace / build on the shipped edge call | **answered by measurement: do not adopt** | Edge matching loses 5.89 of 85 held-out points to plain intensity ZNCC, and loses again with the edge-brightening physics the brief describes added to the forward model. See `PHASE3_MEASUREMENT.md`. The organizer's call is still not in the material supplied; that part stays open. |
| Same coordinate frame for reference and search CAD | **resolved in the generator** | The generator writes `search_gds_path = reference_gds_path` (one site frame) and `ground_truth.csv` is derived from the same crop. The organizer's own convention is still unconfirmed. |
| Absent-pair zero-fill under `found=0` | **implemented** | `generate_phase3_dataset.py` writes `present=0` with zeros in the pose columns; pinned by `test_absent_pairs_are_zero_filled_in_ground_truth`. |
| What "knew your yield fit was wrong" measures | **still open** | Not answerable from the material; needs the clarification window. |

### 1. Data layer

| item | status |
|---|---|
| Phase 3 reader with asserted column resolution | done — `driftsense/pairs3.py`, 25 tests |
| Tolerate the blind split's empty trailing fields | done — tested |
| Inference never dereferences the withheld columns | done — `test_blind_split_withheld_fields_are_never_read` uses poisoned paths |
| `predictions.csv` header pinned | done — `test_output_header_is_the_phase2_contract` |
| Params-JSON schema for training-time use | done — `driftsense/params3.py`, 12 tests |

### 2. GDS → geometry

| item | status |
|---|---|
| Read `.gds` into per-layer geometry with correct layer/**datatype** handling | done — only datatype 0 is drawn geometry; non-zero datatypes are skipped and reported |
| Rasterize at reference **and** search resolution | done — `render_reference_at(nm_per_px=...)` re-samples coordinates rather than downsampling a raster |
| Extend the stack from 4 to 8 layers | done — `driftsense/gds_layers.py`, both architectures |
| Replace hard-coded per-layer greys with derived brightness | done — brightness comes from stack position; emission never assigns a grey level |
| Layer order matches the deck's staged view | done and pinned — DRAM layers 1/3/5 are `word_line`/`bit_line_metal`/`storage_capacitor` |
| Pose mapping applied to geometry identically | not exercised — the generated set has no rotation; see the gap below |
| Round-trip tests with layer identity preserved | done — `tests/test_phase3_layers.py` |

### 3. Yield-raster and per-layer brightness

| item | status |
|---|---|
| Forward chain CAD → yield raster → SEM | done — the render side derives brightness per layer, then the existing `generator/src/sem_imaging.py` chain is used, not a second one |
| Per-layer brightness sampled, not fixed | done — sampled per site and recorded |
| Brightness recorded in params JSON | done — plus `observable_layers`, so a supervision target is never built on a layer the image cannot show |
| Generator emits GDS for reference and search, plus params | done — five artefacts per site |
| Absent rate ≈ 1 in 12, not Phase 2's 20% | done — defaults to 0.08 |

**Known limit:** two DRAM layers (4 storage contact, 6 via1) are 100% occluded
under painter's-algorithm compositing — the capacitor is a strict superset of
the storage contact, and via1 is a strict subset of the metal2 strap. Their
brightness is recorded but is not recoverable from the image, so
`observable_layers` excludes them. This is measured, not assumed.

### 4. Edge-based registration

**Not implemented, and deliberately so.** Measured negative before any code was
written: edge matching loses 5.89/85 held-out, converted-then-edged loses
11.39/85, and a brightness-invariant *and* structure-preserving representation
(rank) loses 16.7/85. Regenerating the search images with edge brightening in
the forward model does not change the ranking. Shipping an edge matcher would
cost points on evidence we already have.

What replaced it: the confidence margin (best peak minus best competing peak)
separates good pose basins (+0.1666 median) from bad ones (+0.0148) at
**AUC 0.948**, which targets the 15-point rejection and 10-point calibration
blocks.

### 5. `phase3.py` entry point

| item | status |
|---|---|
| `--input pairs.csv --output predictions.csv` | done |
| One row per `pair_id`, in input order | done — tested |
| Decline rather than omit; zero-fill on `found=0` | done — tested |
| Fail closed if weights/registration will not load | done — reuses `register.py`'s loader |
| Mass-failure detection | done — early alarm plus end-of-run `# mass_failure:` marker |
| Thread capping | done — reuses `cap_threads` |
| `register.py` keeps working | done — untouched |
| Refuse a bad CSV with an actionable message | done — refuses before writing anything; verified by mutation (reverting the guard fails 11 of 16 entry-point tests) |
| No network at run time | `gdstk` is a local import; nothing added fetches |

### 6. Scoring and evaluation

| item | status |
|---|---|
| Point the Phase 2 scorer at the new root, unmodified | done — verified by vendoring `driftsense/rubric.py` byte-identically and scoring with it |
| Report the new calibration signal alongside AUC | done — margin AUC 0.948 measured |
| Median time per pair for quartile ranking | reported by `phase3.py`'s runtime summary |

## Measured dataset quality, stated honestly

200 pairs generated in **2 m 32 s**:

```
total 200 pairs, 186 present, 14 absent   (7.0% absent; brief says ~8.3%)
```

Localisation with the classical baseline (`zncc_only`):

```
median 1.36 px
<=1 px 38%   <=2 px 57%   <=5 px 57%
```

**A 40% rate of >20 px errors. That is not a generator defect, and it is
important not to read it as one.** Checked directly: for every failing pair the
reference IS present at the ground truth (NCC 0.81–0.91 at the recorded crop;
ZNCC at GT ≥ 0.5 in 74/74 cases, mean 0.844). The failures are **lattice
ambiguity** — a periodic array matches its own aliases almost exactly:

```
ZNCC(best) - ZNCC(at ground truth):  median +0.042
the ground truth is the global maximum in only 8 of 55 pairs
```

At 10 nm/px a 1000 nm reference spans 4–16 pitches of a dense array, so the
true peak is routinely beaten by a few thousandths by an equally valid alias.
`FAILURE_ANALYSIS.md` already documents this mechanism for Phase 2.

This is a property of the **task**, not of the data pipeline: the generator's
job is to place the reference truthfully, which it does. It does mean the
generated set is not a good discriminator for a *localisation* claim on its own
— a model that lands on an alias scores identically to one that lands on the
true site. Any Phase 3 accuracy claim must be made on the organizer's set, or
against a metric that is alias-aware.

## Fixes found by testing, not by reading

Four defects were caught only because the invariant was tested directly. They
are recorded because each was silent:

| defect | symptom | found by |
|---|---|---|
| `datatype` ignored on read | a `datatype=7` polygon invisible to the reference renderer was painted by ours | synthetic GDS with a non-zero datatype |
| Reference clipped from one mat only | a window straddling a mat/strip boundary was missing the neighbour; 17.8% of pixels differed at the true match | comparing the reference raster against the canvas window |
| Non-square mats rasterized square then sliced | polygons past `w` were dropped from the search but kept in the design | mat dimensions 1812×3175 in the failing case |
| Strip routing painted to the raster only | a boundary reference contained none of the routing the search showed | value histogram: the canvas window had grey 115, the reference could not |

The first three share a root cause — **the reference and the search must be
derived from one geometry source**. The invariant test
(`test_reference_matches_the_search_canvas_at_the_crop`) now holds the line, and
reverting the cell-clipping fix fails it at 14.37% differing pixels.

## Remaining gaps, stated plainly

1. **Rotation is not exercised.** The generator pins `search_rotation_deg` at 0
   and `scale` at the nominal 10, so the 20 pose points are constant and
   untested. The brief promises "a wider pose range than Phase 2". This is the
   single largest gap against the brief.
2. **The organizer's edge-registration call is still unseen**, so the "adopt /
   replace" decision is answered only in the negative direction (replacing is
   measured worse).
3. **The CAD2SEM set has not been seen.** The edge A/B was run on locally
   generated data with the edge-brightening physics added by hand.
4. **Lattice ambiguity limits what this dataset can prove** about localisation
   accuracy, as set out above.

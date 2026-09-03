# Drift-Sense — Phase 2 SEM Registration

Drift-Sense recovers a reference pattern inside a repeating semiconductor SEM
layout when **magnification is unknown, rotation is unknown, and the target may
be absent**.

Phase 2 is the canonical project. The original Phase 1 fixed-pose localiser is
kept only as historical/compatibility code.

## Submission entry point

```bash
python register.py --input pairs.csv --output predictions.csv
```

The output contains exactly one row per input pair, in input order:

```text
pair_id,x,y,theta,scale,found,score
```

- `x, y` — recovered centre in Search-image pixels.
- `theta` — recovered rotation in degrees, CCW-positive.
- `scale` — recovered down-scaling factor `z`, nominally in `[8, 12]`.
- `found` — `1` when the target is accepted, `0` when declined.
- `score` — confidence in `[0, 1]`; higher means more confident.

A bad/unreadable pair never removes a row. It is zero-filled with `found=0`,
so one failure cannot invalidate the rest of the batch.

The submission is designed to run **offline**. It does not download models or
make network calls; the shipped checkpoint is `weights/driftsense.pt`.

## Phase 2 rubric alignment

The official scoring (`Applied Materials_Phase 2_Task.pptx`, slide 6) weighs
six areas. **The numbers below are our own held-out synthetic measurements**
(full 2,250-pair internal set, shipped configuration — `threshold=0.18`,
`band=False`) — **not the organizer's blind score**, which is unreleased.

| Scored area | Pts | Our measurement |
| --- | ---: | --- |
| Localisation (Set A 0.45 + Set B 0.55, tiered 1/2/3/5 px) | 40 | 36.05 |
| Pose — scale | 10 | 8.96 |
| Pose — rotation | 10 | 9.02 |
| Rejection (F1 on `found`, reject-positive) | 15 | 13.62 (F1 ≈0.91) |
| Confidence calibration (AUC of `score`) | 10 | 9.88 (AUC ≈0.99) |
| Efficiency (median wall-clock/pair) | 5 | 1.82 s median (CPU, 4 threads; 5 s budget) |
| **Measurable subtotal** | **85** | **77.53** |

Set D (optical, bonus-only) credit: 0.938 — clears the `+6` bonus gate
(`Set D ≥ 0.40` with `Sets A-C ≥ 0.50`). Sources: `.agents/RESCORE_SHIPPED_T018.md`,
`.agents/SUBPIXEL_DRIFT.md`, `.agents/CPU_RUNTIME.md`.

### Submission-surface compliance (slide 5)

| Requirement | Where |
| --- | --- |
| Entry point `python register.py --input pairs.csv --output predictions.csv` | `register.py` |
| Output columns `pair_id,x,y,theta,scale,found,score`, one row per pair | `register.py` (never drops a row) |
| Python 3.11, pinned dependencies | `requirements.txt` (`pip freeze`, tested in CI) |
| Offline: no network, no downloads | shipped checkpoint loads from disk only |
| Weights ship inside the ZIP | `weights/driftsense.pt` |
| Documented generator | `generate_dataset.py` |
| Failure analysis (≤2 pages) | `failure_analysis.pdf` |

## Phase 2 problem

Compared with the original fixed-pose task, Phase 2 removes three assumptions:

1. magnification varies over `[8×, 12×]`;
2. rotation varies over `[-5°, +5°]`;
3. some Reference/Search pairs contain **no true instance**.

The hard part remains the periodic semiconductor layout: many local patches can
look almost identical, so the system must identify the correct repeat before
sub-pixel localisation and pose refinement are useful.

## Method

The shipped pipeline keeps the Siamese network as the learned repeat selector
and wraps it with explicit pose search and native-resolution verification:

1. **Pose hypotheses** — search multiple scale/rotation basins instead of
   trusting a single coarse correlation maximum.
2. **Canonicalisation** — undo each pose hypothesis so the Siamese network sees
   the nominal distribution it was trained on.
3. **Repeat selection** — the Siamese correlation model decides which periodic
   candidate is the intended location.
4. **Native-resolution ZNCC verification** — reject wrong pose basins and refine
   the match without inheriting canonicalisation blur.
5. **Pose polish** — refine centre, scale and rotation around the selected
   location.
6. **Rejection** — combine learned and native-resolution confidence and decline
   pairs below the shipped threshold.

The current shipped operating point is defined once in `driftsense/config.py`:

```text
threshold    = 0.18
band filter  = false
verification = zncc
```

`register.py` and `scripts/eval_ext.py` consume the same configuration, and
`tests/test_submission_parity.py` checks the two paths end to end.

## Confidence semantics

`score` is monotonic confidence, not a calibrated probability. The Phase 2
pipeline combines two signals that fail differently:

- the **network score**, which represents confidence in the selected periodic
  repeat;
- **native-resolution ZNCC**, which checks whether the recovered pose actually
  matches the Reference at the reported location.

The shipped decision is:

```text
found = score >= 0.18
```

The threshold is chosen against the competition rubric rather than F1 alone,
because incorrectly declining a present pair also forfeits localisation and
pose credit.

## Quick validation

Run the full test suite:

```bash
python -m pytest -q
```

Audit the exact candidate ZIP, not only the checkout:

```bash
python scripts/check_submission_zip.py submission.zip
```

The artifact audit verifies required files, pinned requirements, checkpoint
loading/instantiation, entry-point smoke tests, offline import closure and the
failure-analysis PDF constraint.

For Phase 2 scoring on generated hold-out data:

```bash
python scripts/eval_ext.py --help
python scripts/grade_emulation.py --help
```

The evaluator masks localisation/pose credit when `found=0` before splitting
sets, and the grade emulator uses **rejection-positive F1** for the rejection
component and bonus gate.

## Generate Phase 2 synthetic data

The project generator supports Phase 2 pose variation and absent pairs:

```bash
python generate_dataset.py \
  --phase2 \
  --num-pairs 100 \
  --output-dir data/phase2_sample
```

For explicit ranges:

```bash
python generate_dataset.py \
  --magnification-range 8 12 \
  --rotation-range -5 5 \
  --absent-frac 0.2 \
  --num-pairs 100 \
  --output-dir data/phase2_sample
```

Synthetic data is for training/validation only; organizer data is not used for
training.

## Repository layout

```text
register.py                 Phase 2 submission CLI (canonical)
driftsense/
  config.py                 shipped Phase 2 operating point
  runtime.py                shared checkpoint / image / fallback runtime
  matching.py               pose search, localisation and refinement
  model.py                  Siamese network
  verification.py           verification utilities
  generate.py               synthetic generation internals
weights/driftsense.pt        shipped checkpoint
scripts/eval_ext.py          Phase 2 evaluator
scripts/grade_emulation.py   rubric / blind-composition emulator
scripts/check_submission_zip.py
                             artifact-level submission audit
generate_dataset.py          synthetic data CLI
failure_analysis.pdf         required failure analysis
```

Training and research utilities remain under `train.py`, `scripts/` and
`generator/`; they are not part of the runtime contract.

## Legacy Phase 1 compatibility

`infer.py` is the historical single-pair Phase 1 interface that prints `x,y`.
It is **not** the Phase 2 submission entry point. Shared runtime helpers were
moved to `driftsense/runtime.py`, so `register.py` no longer depends on the
legacy CLI.

Phase 1 remains useful for regression tests and for understanding how the
Siamese repeat selector was developed, but new usage should start from
`register.py`.

## Documentation

- `TRAINING.md` — training workflow and checkpoint provenance.
- `CITATIONS.md` — methods and literature references.
- `.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md` — Phase 2 contract notes derived
  from the official materials.
- `.agents/PHASE2_COMPLIANCE_ISSUES.md` — compliance review history.
- `.agents/VERIFICATION_REPORT.md` — validation notes and evidence.

## Reproducibility / safety rules

- no organizer data is committed or used for training;
- the submission runtime makes no network calls;
- `weights/driftsense.pt` is the only checkpoint required by the shipped path;
- every input pair produces exactly one prediction row;
- Phase 2 runtime/evaluator defaults are centralized in `driftsense/config.py`.

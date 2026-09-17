# AP-25 — evaluation of the shipped submission on the organizers' 25-pair set

The organizers' coverage-preserving 25-pair cut of the Phase 2 v2 dataset,
scored against the published rubric with the frozen shipped submission.

**Result: 82.71 / 100** (85-point subtotal 76.71, +6 Set D bonus, +4 F1 bonus
missed). Full write-up: [`AP25_EVALUATION.pdf`](AP25_EVALUATION.pdf).

| | Naive ZNCC baseline | **This submission** |
|---|---|---|
| Set A localisation credit | 0.9778 | 0.9111 |
| Set B localisation credit | 0.4444 | **0.8222** |
| Rejection F1 | 0.5714 | **0.8889** |
| **Total** | 59.56 | **82.71** |

## What is here

| File | Contents |
|---|---|
| `pairs.csv` | The 25 pairs, as published to participants |
| `ground_truth.csv` | Scoring key (withheld from participants) |
| `manifest_jury.csv` | Jury-only: set, architecture, severity, rivals, margins |
| `predictions.csv` | This submission's output from the required entry point |
| `register.stderr.log` | Decode provenance: threshold, threads, per-pair timings |
| `ap25_per_pair.csv` | Per-pair rubric record (error, credit, pose, score) |
| `ap25_components.csv` | Component ledger |
| `ap25_baseline_zncc.csv` | Naive ZNCC baseline over the same 25 pairs |
| `ap25_rowrefine_ab.csv` | Paired A/B of `SHIPPED_SUBPIXEL_ROWS` |

## Reproduce

```bash
python register.py --input evaluation/ap25/pairs.csv --output predictions.csv

python /path/to/judging/score_rubric.py \
    --pred predictions.csv \
    --gt evaluation/ap25/ground_truth.csv \
    --manifest evaluation/ap25/manifest_jury.csv
```

The scorer is the one that produced the Phase 2 campaign's ranking table.
`tests/test_eval_report_parity.py` pins this repo's recomputation of the rubric
to it; the test skips cleanly where `judging/` is not present.

```bash
python -m pytest tests/test_eval_report_parity.py -q   # arithmetic parity
python evaluation/make_eval_report.py                  # rebuild the PDF
```

The PDF needs `reportlab`, `matplotlib` and `pandas`; the parity test needs only
`pandas` and `numpy`, so it runs in CI.

## Headline findings

1. **The cut is harder than the parent set and the intended way.** Set B credit
   drops 0.587 → 0.444 for the naive baseline and rejection F1 drops to 0.833.
   Our Set B credit of 0.8222 against the baseline's 0.4444 is where the +23.15
   total margin comes from.

2. **Localisation is the only component with real mass left** — 34.49 of 40.
   Nine of twenty present pairs land above the 1 px tier, and the naive baseline
   clears that tier more often than we do (16/20 vs 11/20). On Set A the
   baseline is genuinely ahead (0.9778 vs 0.9111).

3. **The error is not isotropic.** Signed error averages dx = +0.610 px
   (std 0.984) and dy = +0.486 px (std 0.159) — tight on y, a fingerprint of a
   constant convention offset, and the organizers' own baseline shows no such
   shift (dx = +0.112, dy = −0.086). This localises the defect to sub-pixel
   placement. It is a diagnostic, not a free fix: the blind set has no ground
   truth to subtract against.

4. **`SHIPPED_SUBPIXEL_ROWS` does not reproduce here.** Its promotion record
   cites +0.589 localisation points on 2,500 pairs; on these 25 it is **−0.98**.
   It moves x on 10 of 20 pairs (y never moves), hurting 4 and helping 5. The
   drift row is recovered from the search trace, so the correction is a property
   of each generator's raster-drift model — a generalisation gap across
   generator builds, worth up to +1.0 point.

5. **The +4 F1 bonus is one pair away.** F1 is 0.8889 (tp/fp/fn = 4/0/1) against
   the 0.90 gate; a single false negative at p035 — a severity-0 absent decoy
   scored 0.2208 against the 0.18 gate — forfeits it. Declining it correctly
   alone would take F1 to 1.0000 and the total to 86.71.

6. **Severity is not the driver.** Level 4 pairs average a *lower* error
   (1.065 px, 3/4 inside 1 px) than level 2 (1.472 px) or level 3 (1.578 px,
   0/3 inside 1 px). Severity compresses confidence, not position — the
   organizers predicted this and the numbers reproduce it. The middle of the
   severity range is where the loss actually sits.

7. **Scale is solved, rotation is not.** All 20 credited pairs land inside the
   1% scale tier (median 0.199%). Six miss the 0.25° rotation tier, five of them
   at severity 3–4, costing 1.11 of the 10 rotation points.

## Provenance

- Commit `33841c5`, checkpoint `weights/driftsense.pt`
  (sha256 `e6506b7c3b2ccfd47f9337e9b95df20d0fb2b14a6d45b0a1c892d60464a71ca8`)
- Decode: `SHIPPED_THRESHOLD = 0.18`, band off, ZNCC verification, sub-pixel
  rows on, 4 threads
- Runtime: 1.89 s median, 4.73 s max — inside the 5 s median / 20 s hard budget
- Harness validated by reproducing the organizers' published baseline numbers
  (Set A 0.9778 vs 0.978, Set B 0.4444 vs 0.444, `core_credit` 0.6844 vs 0.684)

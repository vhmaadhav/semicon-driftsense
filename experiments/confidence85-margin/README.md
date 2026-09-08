# Confidence-model experiment against the complete /85 rubric

This is an **unshipped research candidate**, not a change to the submission's confidence policy. It uses the current checkpoint's existing decoder outputs, without changing candidate selection, coordinates or pose. The fit/tune/assessment protocol and stop rule were written before fitting in `spec.md`.

## Why another confidence experiment

Earlier raw-feature fusion failed independent validation. This one uses raw network score (not the already-min-combined score), ZNCC, peak ratio, coarse pose peak, log1p(PSR), log1p(APCE), and winner margin. A separate indicator distinguishes unavailable margins from real zero-margin ties: 975/2500 rows in the input lack that signal. Six base signals must be finite; no unavailable rank/band features are silently inserted.

The head is an eight-input ridge logistic presence model. Parameters and standardisation are learned only on the fit partition, the full /85 threshold is selected only on tuning data, and the artifact is saved before assessment. Groups use generator bundle/shard/canvas identity. A stable hash assigns approximately 60/20/20%; row order cannot change membership.

## Development results

Input: PR #75 head `70b6373`, tracked G2 CSV SHA256 `9d0a6a325ee82be3699643ef101d17e690e663ab62875cd724d1e85b03c4a519`. The 2,250 grayscale rows have 2,250 distinct source groups in available metadata. These partitions are **development assessment**: the original pool has already been reused, and the two seeded splits overlap.

| split seed | fit/tune/assess | fixed baseline /85 | candidate /85 | delta | rejection F1 before -> after |
|---|---|---:|---:|---:|---|
| 20260908 | 1346 / 434 / 470 | 76.8382 | 77.5262 | +0.6880 | .89583 -> .92708 |
| 20260909 | 1318 / 493 / 439 | 76.9476 | 77.6180 | +0.6704 | .88398 -> .91620 |

The threshold-only control scores 76.7515 / 76.9476 respectively. Candidate thresholds are .48650493125056 and .3712103681232539. There were 6/12 changed assessment decisions. The second split newly declines one real pair despite improving aggregate F1; inspect all rubric components, not F1 alone.

Both predefined development gates (+.35) passed, so the FIRST seed's frozen head proceeded to confirmation (result below: gate failed). It is not selected because it happened to have the larger delta. No full-pool refit or assessment-driven hyperparameter revision was made.

## Reproduction

Copy the public G2 CSV to `input/development.csv`. Build `input/source_groups.csv` from corresponding shard manifests with columns `pair_id,source_group`; define source_group as `generator_bundle_sha256 + '/' + shard_directory_name + '/' + canvas_id`. All grayscale IDs must match one-to-one. Keep generator source/data access under their original sharing constraints; do not publish organiser materials through this experiment.

```bash
python scripts/experiment_confidence85.py \
  --csv experiments/confidence85-margin/input/development.csv \
  --groups experiments/confidence85-margin/input/source_groups.csv \
  --output experiments/confidence85-margin/working
```

The saved model is label-free at inference: `predict(feature_frame, artifact)`. It is deliberately absent from the `register.py` import path. A compact model copy removes per-split group lists, retaining their hashes; its learned fields are identical to the tested artifact.

For a separately generated 200-pair confirmation set in `pairs.csv / ground_truth.csv / manifest_jury.csv / reference / search` layout:

```bash
python scripts/confirm_confidence85.py \
  --directory experiments/confidence85-margin/input/confirmation \
  --weights weights/driftsense.pt \
  --head experiments/confidence85-margin/candidate.json \
  --output experiments/confidence85-margin/working
```

This explicitly uses CPU, two threads, fused BN and channels-last. Both policies consume the same decoded predictions. The fresh-set paired bootstrap preserves row pairing and A/B/C strata and recomputes the nonlinear rubric. It assumes an independent generated canvas per row; the runner's unique-seed check does not alone establish independence from training sources.

## Confirmation provenance

Generation seed 73120260908, fresh per-pair RNG and newly constructed canvases. Source inspection confirms no development canvases are loaded. 400 emitted PNG hashes are distinct, with zero exact hash overlap with available development images; no nominal seed overlaps the development sample-entropy IDs. This is procedural/data provenance, not proof of semantic pattern uniqueness. Generator source hashes and original input artifacts remain local. The generator's stored verification errors are <=3px for present pairs; this does not substitute for a new independent PNG-label audit. Set D images were decoded and verified as three-channel RGB.

The source is a local organizer-generator reconstruction, not the undisclosed blind set. Its different distribution is an intentional transfer check. Timings during concurrent machine use must not be quoted as judge efficiency evidence.

## Fresh confirmation: retain shipped policy

The frozen first-seed head was evaluated once on 200 newly generated pairs (A70/B70/C40/D20), with no refit or threshold change. Baseline and candidate both scored **80.246058/85: delta 0.000000**. Both rejected all 40 absent pairs and declined one real pair; rejection F1 was .987654 and historical-label AUC was 1.0. Set D credit was .94 for both. The paired, within-stratum 2,000-draw bootstrap interval was [0, 0]; this describes this finite pool, not universal equivalence.

The alternative submitted-correctness AUC decreased from .793296 to .782123. The predeclared confirmation gate (delta >= .35 and lower interval >= 0) failed. **Do not promote this head.** The development improvements did not demonstrate transfer; the fresh proxy also offers little remaining rejection headroom. Keep the shipped policy and checkpoint unchanged. This experiment does not test the candidate-bank-plus-none architecture proposed in #76.

See `confirmation_results.json` for all rubric components and provenance hashes. The generated proxy is not the official blind set. Validation: 39 targeted tests passed, including fit/tune isolation, label-free inference and paired rubric parity.

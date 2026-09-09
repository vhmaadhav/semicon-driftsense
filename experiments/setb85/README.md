# Work toward 85% Set B accuracy

Target: strict radial error **<1 px on all present B pairs**, including rejected and unsupported cases as failures. **The target has not been reached.** These follow-up results are from the historically reused development pool, not fresh confirmation.

| Experiment | Validation B accuracy | Decision |
|---|---:|---|
| Shipped baseline | 68.75% | Control |
| Wider native-strip CNN, distribution or direct <1px objective | Best trial 68.18% | No model selected |
| Image-only distortion hypotheses + learned selector | 69.32% | One additional validation success; insufficient evidence for promotion |

The distortion selector changes full development B accuracy from 67.20% to 67.54%, and the separate development assessment remains unchanged. All candidates preserve the existing y, pose, confidence and acceptance decisions. The selector cannot recover candidates that its hypothesis bank does not contain: choosing with labels from this bank reaches only **78.86%** on all B. This is an oracle diagnostic, not deployable accuracy or a general information-theoretic limit.

## Diagnostic findings

An earlier oracle over the original position and top-three native/aligned centre-row correlation peaks reaches **676/853 = 79.25%** on supported B. More ranking capacity alone cannot yield 85% from that particular finite candidate set.

With **true centre, true pose and true radial coefficient supplied**, native centre-row matching reaches **643/875 = 73.49%**; known-coefficient undistortion increases it to **714/875 = 81.60%**. At severity 4 the latter is **145/218 = 66.51%**. This is a label-assisted controlled diagnostic, not a deployed result. Undistorting the reference assumes its coefficient is 0.3 times the search coefficient, following the local generator; the exact external generator correspondence has not been independently proved. Inverse interpolation can itself lose information, so this diagnostic is not an achievable-accuracy bound.

The implemented image-only bank tries nine radial coefficients, uses first-order scale compensation, and computes row refinements on locally remapped patches. Its generator accepts only images and baseline predictions. Labels enter selector training and diagnostics only. The difference between the oracle and selected result remains substantial, but even the bank's oracle is below the requested target.

## Next controlled experiment

`scripts/train_fresh_rows.py` trains the wider feature model on up to 512 fresh procedural scenes with 32 crops each, retaining original training crops as an anchor. Crop and pose errors are sampled only from the original training partition, stratified by severity. Validation rows never enter the fitting pool. This tests data diversity after small heads trained on the fixed pool stalled; synthetic candidate errors may still differ from actual end-to-end errors. Completed training used 16,181 fresh crops (17,324 with original training anchors). Epoch 20 was selected: validation B 121/176 → 123/176, assessment B 50/86 → 58/86, and all reused development B 588/875 → 625/875 (71.43%). All development A improved 94.63% → 95.43%. These are development gains, not fresh confirmation. Saved-checkpoint inference reproduces all 2,250 cached predictions within 1.14e-13 px. A six-shard frozen confirmation (420 B plus A/C/D controls) is running; its policy and hashes are in `fresh_confirmation/freeze.json`.

The checkpoint must be frozen before any new confirmation data is generated. Previous confirmation seed 9120260909 remains seen data. Any future 85% claim requires fresh B evaluation with severity breakdown, explicit denominators and uncertainty, plus Set A and rejection controls.

## Reproduction and evidence

```sh
python scripts/train_setb85.py --data /path/to/ext_p2 --predictions /path/to/G2_ext_full_shipped.csv --output /tmp/context
python scripts/diagnose_scan_geometry.py --data /path/to/ext_p2 --output /tmp/geometry
python scripts/train_scan_bank.py --data /path/to/ext_p2 --predictions /path/to/G2_ext_full_shipped.csv --output /tmp/scan-bank
python scripts/train_fresh_rows.py --development /tmp/context --output /tmp/fresh-rows
```

Use fresh output directories when inputs change; caches are reused by these experimental scripts. Source images and large caches are not committed. Included summaries: `ceiling.json`, `context/results.json`, `geometry/results.json`, and `scan_bank/results.json`. Shipped inference and backbone weights remain unchanged.

## Fresh confirmation outcome

Fresh 1,200-pair confirmation: B **262/420 → 264/420 (62.86%)**, paired delta 95% CI **[-4.76, +5.71] percentage points**. A **328/420 → 366/420**. Severity4 B **48/90 → 39/90**. Promotion fails; target85 remains unmet. All six confirmation seeds are now seen. Residual diagnosis finds 155/156 B failures x-dominant and only three errors over5px, so gross pose recovery is not the main next lever on this proxy. Investigate generator/training scanline mismatch and centre-row representation before another training run. Aggregation was repaired to namespace shard-local pair IDs and score absent C as rejection; original freeze and amendment retained, predictions/model unchanged.

## Pixel-centre training ablation

Code comparison identifies a label convention mismatch: local in-memory generation maps crop centre500 and adds the area offset; the confirmation generator maps499.5 without that offset. Across100,000 random poses this changes the selected independent jitter row in50.01% of cases. The optional `make_pairs(pixel_center_labels=True)` maps the physical centre before drift; defaults remain unchanged. A matched512-scene training arm is running with this flag. Images/rendering are unchanged; Gaussian-versus-box search prefilter mismatch remains. Historical validation uses legacy labels, so this arm requires cautious interpretation and fresh confirmation before promotion. Fifteen generator/label tests passed.

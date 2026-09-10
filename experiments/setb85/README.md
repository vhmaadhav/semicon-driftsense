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

Pixel-centre label ablation completed on512scenes/16,175fresh crops: no checkpoint passed the historical validation gate (`selected=null`); reported candidate columns therefore equal baseline and are not learned-model gains. This gate uses a different label convention. The next arm explicitly converts previously consumed1,200-pair confirmation into development, split by source group, and reuses the pixel-centre synthetic crops. These data are now seen; any eventual candidate needs a new frozen holdout.

## Aligned-development result

Aligned-development epoch39: B **262/420 → 326/420 (77.62%)**; group validation **49/83 → 61/83 (73.49%)**, assessment **34/43 → 40/43 (93.02%)**. A **328/420 → 411/420**. These1200pairs were previously consumed confirmation and are now development, not fresh evidence. A new420B confirmation is frozen in `aligned_confirmation/freeze.json`; outcome pending. The small assessment result does not establish85%.

## Aligned-model fresh confirmation

Frozen aligned-model confirmation on1,200 new organizer-proxy pairs improves strict B **262/420 → 302/420 (62.38% → 71.90%)**, paired95% delta **[+4.76,+14.52] percentage points**. A improves **321/420 → 408/420 (97.14%)**; all240absent controls remain rejected. B severity1/2/3/4 accuracy:88.33/75.93/64.71/53.33%. The declared improvement gate passes on this proxy, but85% is not reached and cross-generator safety is unproven; shipped inference stays unchanged. These confirmation seeds are now seen. Next isolated training arm replaces Gaussian prefilter with the organizer-style box filter; same512 scene seeds, pixel-centre labels and prior seen_context train/validation, without incorporating this new confirmation.

## Box-prefilter development result

Box-prefilter epoch39 reaches337/420=80.24% B on reused development. Validation64/83=77.11% versus61/83 for prior aligned model; assessment37/43=86.05% versus40/43. A414/420=98.57%. Saved checkpoint reproduces1,200 cached predictions within1.14e-13px. No fresh accuracy claim: a new420B confirmation is frozen in box_confirmation/freeze.json.

## Box-model fresh confirmation

Box-model frozen fresh confirmation: B249/420→307/420 (**59.29%→73.10%**), paired95% delta **[+8.33,+19.29] percentage points**. A311/420→414/420; absent rejection239/240 unchanged. B severity1/2/3/4:93.33/74.07/68.63/50.00%. This exceeds its paired baseline but does not establish superiority to the prior model tested on different samples.85% remains unmet. Of113 B failures,85 are1–2px,27 are2–5px,one≥5px; none has |dy|≥1. A fixed maximum-success probability decoder was tested only on seen development and rejected: validationB77.11%→71.08%, A100%→97.70%.

## Row reliability experiment

Seen-development texture diagnostic: B accuracy by centre-row gradient-texture quartile is72/105,80/105,90/105,95/105. A label-assisted choice among17 row peaks covers414/420, but is not deployable accuracy. A reliability-conditioned residual now tests whether explicit reference/search texture and correlation peak/margin help combine neighbouring rows. The prior box model encoder/head are frozen; only the new zero-initialized residual and reliability weights train, reusing existing synthetic crops. Outcome pending.

## Reliability outcome and strip geometry

Reliability residual development: B validation68/83=81.93% versus64/83 for box model, assessment36/43 versus37/43, all340/420=80.95%. A validation86/87 versus87/87 (1.15pp decline), so no new confirmation spent yet. Geometry diagnosis then found template-strip vertical centre h/2 while x uses(w-1)/2. For a physical pixel-centre label this samples0.5px too low. An exact embedded-row test proves the offset; optional patch_y_offset=-0.5 corrects it while defaults remain unchanged. Next isolated arm trains with corrected patch sampling on the same seen-development split and fresh synthetic crops; five context/reliability tests pass.

## Centered-strip outcome

Centered-strip epoch15: validation B66/83=79.52% but A85/87 versus87/87 for box model; all B338/420=80.48%. Not promoted. Fixed equal averaging with box model reduced validationB to63/83=75.90%, rejected. Next experiment tests early spatial features (3x5 instead of1x5 convolutions), using neighbouring image rows before correlation reduction rather than only combining row correlation curves. Same cached box training crops and seen_context split; pending, no gains claimed.

## Spatial-feature development result

SpatialContextRow epoch5 improves reused B development to349/420=83.10%. Validation67/83=80.72% versus64/83 for box; assessment39/43=90.70% versus37/43. A validation87/87 and assessment44/44. These are seen-development results, not85% confirmation. A new420B evaluation is frozen in spatial_confirmation/freeze.json; evaluator restores the matching1-row or3-row kernel architecture from checkpoint shape with strict state loading.

## Spatial-model confirmation

Spatial model fresh confirmation: B269/420→332/420 (**64.05%→79.05%**), paired95% delta **[+9.29,+20.71] percentage points**. A324/420→416/420=99.05%; all240absent controls rejected. B severity1/2/3/4:92.50/87.04/76.47/54.44%; severity4 has zero net gain.85% remains unmet. Candidate comparisons across different holdouts are not paired. Training code samples barrel k up to±.02, whereas this organizer proxy uses at most.005. Next isolated arm caps training |k| at.005, preserving architecture, other degradation settings and seen-development split; it targets this proxy and may reduce transfer to wider distortion ranges.

## Original-population transfer failure

Narrow-barrel training did not improve validationB over spatial model (both67/83=80.72%); no fresh confirmation spent. Crucial original-data transfer check: spatial model B degrades67.20%→55.20%, A94.63%→82.17%; narrow-barrel B56.23%, A85.83%. Thus organizer-proxy gains do not improve the original scoring population. These models must not be promoted. The next arm uses spatial features with ORIGINAL context development and legacy-label/Gaussian512-scene caches, restoring the original target convention.

## Original spatial outcome

Original-label spatial arm: selected epoch0, validation B125/176=71.02% vs121/176 baseline; assessment57/86=66.28%; full B616/875=70.40%. A remains96.47% validation and95.43% full. Prior original horizontal CNN fullB625/875=71.43% is higher, so no fresh confirmation spent. Saved spatial checkpoint reproduces2250 cached outputs. Next original-population test adds reliability conditioning to that stronger original horizontal CNN, freezing its features and training only the residual on cached original-label data. Proxy-trained reliability results do not establish performance on this target.

## Original reliability outcome

Original reliability residual fails to improve B over its base: validation123/176 unchanged, assessment58/86 unchanged, all624/875=71.31% versus625/875. No fresh confirmation spent. Next bounded diagnostic tests full-width search scanline motion minus local-window motion as a fixed horizontal correction; uses image pixels only, no labels in proposals. This tests evidence outside the limited reference crop. Two tests cover displacement sign and unsupported flat rows.

## Global scanline result

Global scanline correction failed: original validation B68.75%→67.61%, A96.47%→92.94%; full B67.20%→65.83%. Rejected without threshold tuning. Work now verifies the original notebook generator by replaying one recorded seed/acquisition profile before spending a fresh evaluation. Its embedded bundle hash differs from the dataset manifest, so compatibility must be checked rather than assumed.

# Set B horizontal refinement experiments

The frozen correlation refiner improved one fresh 200-pair proxy evaluation from **81.707143 to 82.010000 /85** (+0.302857). Set B accepted ≤1 px matches increased **46/70 → 50/70** (65.71% → 71.43%). This is a measured preliminary gain, **not a validated production improvement**: the paired 95% interval is **[-0.171571, +0.823000]**, and the predeclared +0.35/lower-bound≥0 gate failed. Shipped inference, confidence policy and backbone weights are unchanged.

## What was implemented

`driftsense/row_refiner.py` extracts native and fractionally aligned row-correlation curves, including the centre row and adjacent-row evidence. A small trained MLP predicts 33 horizontal shift bins over ±4 px. Its decoder chooses one mode and interpolates only within its immediate neighbourhood. Unsupported crops retain the incumbent coordinate. The experiment updates accepted rows only at the frozen .18 threshold; y, scale, rotation and confidence are preserved.

Training compares distributional cross-entropy with/without an auxiliary smooth approximation to the 1/2/3/5 px localisation tiers. This surrogate uses horizontal error; the final evaluation uses the full radial error and complete rubric. A separate translated-window ablation encourages translation equivariance. `fine_row.py` implements a second, native-strip CNN with horizontal convolutions, rowwise feature correlation and local decoding. These modules are experimental and are not called by the shipped registration path.

The coarse backbone was frozen; this is training of new small refinement models, not a backbone retrain. Structural candidate reranking and wrong-basin rescue remain future work.

## Development and model selection

Development uses PR #75's saved predictions and the available `ext_p2` grayscale images. Source groups are shard name plus canvas ID; SHA-256 buckets allocate 70% train, 20% validation and 10% assessment. Epoch/arm selection uses validation only. The pool has been repeatedly examined historically; none of these partitions is claimed as untouched blind data. The assessment partition was reported after each experiment and is development evidence.

| Model | Validation delta /85 | Assessment delta /85 | Full historical 2,250-row delta /85 |
|---|---:|---:|---:|
| Correlation MLP, selected epoch 50, tier weight 1 | +0.196238 | +0.037500 | +0.107429 |
| Translated-window correlation MLP | -0.004678 | +0.052268 | +0.218119 |
| Native-strip CNN | -0.181443 | +0.125736 | +0.038226 |

The first two models' split metrics use 1,907 supported crops (856 A, 853 B, 198 C). Full historical deltas restore all omitted predictions as unchanged no-ops. CNN metrics retain every row, including unsupported crops. The largest full-pool delta was not used to select the confirmation candidate; the validation winner was frozen instead. First-model training was rerun and all exported weight arrays were exactly identical.

Simple direct-row, fractional-alignment and winsorised-row ablations on 240 historical development rows also failed to demonstrate a promotion-sized benefit. Their best measured subtotal delta was +0.135/85. See `append_dev/probe.json`; these are post-shipped-refinement probes, not replacements of the original pre-refinement stage.

## Fresh paired confirmation

The candidate and backbone hashes were recorded in `freeze.json` before generating the confirmation set, seed **9120260909**. The local organizer-generator reconstruction emitted A70/B70/C40/D20, with one newly generated canvas per pair. There are 400 unique image hashes and zero exact overlaps with available development images. This does not prove semantic pattern uniqueness or official blind-generator equivalence.

| Metric | Baseline | Candidate |
|---|---:|---:|
| Total /85 | 81.707143 | 82.010000 |
| Set A weighted localisation credit | .951429 | .954286 |
| Set B weighted localisation credit | .922857 | .934286 |
| Set A accepted ≤1 px /70 | 53 | 54 |
| Set B accepted ≤1 px /70 | 46 | 50 |
| Rejection F1 | 1.0 | 1.0 |
| Historical-label correctness AUC | 1.0 | 1.0 |
| Set D credit | .95 | .95 |

Scale/rotation credits are identical. Submitted-answer correctness AUC is undefined because that target contains only one class. The paired bootstrap resamples identical baseline/candidate row indices within A/B/C for 2,000 draws (seed 20260909). The scorer requires unique source groups; repeated canvases would require a clustered bootstrap. This finite proxy sample does not establish the population gain.

Median incremental refiner time across all 200 rows was approximately **10 ms** on the local M4 CPU. This includes no-ops for rejected pairs and is not a judge-x86 latency claim. No confirmation labels were used to alter the checkpoint or policy.

## Reproduction

Use the repository Python environment with NumPy, pandas, PyTorch and OpenCV. The large source image pool and generator reconstruction are not included in this PR; supply their paths explicitly. The original prediction CSV SHA-256 is `9d0a6a325ee82be3699643ef101d17e690e663ab62875cd724d1e85b03c4a519`.

```sh
python scripts/train_row_refiner.py --data /path/to/ext_p2 --predictions /path/to/G2_ext_full_shipped.csv --output /tmp/row-training
python scripts/train_row_refiner.py --data /path/to/ext_p2 --predictions /path/to/G2_ext_full_shipped.csv --translation-augmentation --output /tmp/row-augmented
python scripts/train_fine_row.py --data /path/to/ext_p2 --predictions /path/to/G2_ext_full_shipped.csv --output /tmp/fine-row-training

python /path/to/organizer_generator/gen_200.py --set-id row-confirm-20260909 --seed 9120260909 --output-dir /tmp/row-confirm-data
python scripts/eval_row_refinement.py --data /tmp/row-confirm-data --weights weights/driftsense.pt --row-refiner experiments/setb_rows/learned/row_refiner.npz --output /tmp/row-confirm-results
python scripts/score_row_refinement.py /tmp/row-confirm-results
```

The cached feature-building stages reuse existing caches when present; use a new output directory for a new dataset. For scoring the committed confirmation, run `python scripts/score_row_refinement.py experiments/setb_rows/confirmation`. Formatting after confirmation changed the module's text hash but not its AST; `frozen_row_refiner.py` preserves the exact evaluated source and `validation.json` records the check. Model weights were not altered.

## Evidence and limits

Included: trained artifacts and selection histories for all three arms; frozen provenance; paired confirmation predictions, image hashes and score; reproducibility checks. Large feature/crop caches and local absolute-path input manifests are intentionally excluded. Tests cover native geometry, independent centre-row evidence, translation extraction, local-mode decoding, NumPy/PyTorch agreement, pair alignment, invariant outputs and rejection of invalid bootstrap inputs.

The methods draw on distributional localisation ([Generalized Focal Loss, 2020](https://arxiv.org/abs/2006.04388)) and local-mode fine matching ([Efficient LoFTR, 2024, §3.4](https://arxiv.org/html/2403.04765v2)). Neither paper establishes SEM gains; this PR's numbers come only from the recorded experiments. The result supports further independent confirmation, not deployment or another parameter sweep on this confirmation set.

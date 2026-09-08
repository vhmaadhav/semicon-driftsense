# confidence85-margin: frozen seven-feature confidence experiment

## Goal
Train and evaluate an optional confidence model from existing decoder outputs; preserve shipped weights/threshold and leave work in a draft PR.

## Data
Copied input/development.csv: public PR75 G2 saved predictions, 2,500 pairs (2,250 gray A/B/C + 250 D). This pool has been used for previous development and is NOT an untouched test, even when split again. Source CSV SHA256 9d0a6a325ee82be3699643ef101d17e690e663ab62875cd724d1e85b03c4a519. Attach source-group metadata from the corresponding local shard manifests; exclude D from fitting and /85. Group by generator bundle, shard and canvas ID to keep related source renderings together.

## Evaluation
Primary metric: total measurable /85, maximise, historical correctness-AUC definition. Report alternative submitted correctness AUC separately. Fit on ~60% of source groups, choose threshold on ~20%, assess on remaining ~20%, assigned by stable SHA256 hash with seeds 20260908 and 20260909. Fixed shipped baseline t=.18. Also report a tune-only baseline threshold control to distinguish changed features from threshold tuning.

One predefined candidate: logistic P(present) on raw network score, ZNCC, peak ratio, coarse pose peak, log1p(PSR), log1p(APCE), winner margin plus a margin-missing indicator. Missing margins are zero-filled only alongside that indicator; the six base signals must be finite. Standardise using fit partition only; ridge .001, 4,000 GD iterations, lr .5, no hyperparameter search. Margin existed in earlier work; the experimental change is the joint seven-feature fit with log-transformed peak-shape tails and clean fit/tune/assessment separation. Do not claim the candidate bank plus none experiment is implemented here.

## Constraints
CPU-only offline fit, <=10 minutes per run, NumPy/pandas only. One candidate family; two split seeds as predeclared robustness checks. No GPU run concurrently with the separate TTA worktree. Final outputs: working/results.json, working/assessment_predictions.csv and working/candidate.json, plus a journal. Generated caches/input copy remain untracked.

## Success and stop rule
Candidate must exceed fixed baseline by .35/85 on BOTH development assessment splits before spending compute on independent generated confirmation. If it fails either, stop the candidate and publish the negative result plus reproducible experiment code. No threshold/feature retuning based on assessment results. Even if it passes, no promotion without a fresh source-canvas-disjoint confirmation set and paired full-rubric uncertainty. Existing-pool partitions do not undo historical selection bias.

## Pre-fit data inspection amendment
Before any fitting or assessment, 975/2500 winner margins were missing. The decoder explicitly returns NaN when no competing evidence exists; n_hyp in this evidence file is not a reliable evaluated-candidate count. Therefore the predeclared model uses seven signals plus one availability indicator (eight numeric inputs), not silent zero imputation. No labels or assessment performance were used to choose this handling. All six base signals are finite, PSR/APCE nonnegative. All 2250 grayscale pair IDs match metadata; each has a unique source group in this pool.

## Confirmation protocol frozen after both development gates passed
The predefined candidate passed both development gates. Confirm the FIRST seed's frozen artifact (20260908), not whichever seed scored best. No refit on all development data. Generate a fresh 200-pair A70/B70/C40/D20 set with the local organizer-generator reconstruction at seed 73120260908, a new directory, then decode once with the pinned e6506b7c checkpoint. Apply baseline .18 and candidate .48650493125056 to the SAME decoder outputs. The new set is a proxy-domain confirmation, not the official blind set. Do not use confirmation labels to modify the head/threshold. Stop and keep unshipped if delta < .35 or paired uncertainty includes a material regression. Generator source, image and model hashes remain in local provenance; report metrics without republishing organiser source materials.

## Recorded stopping decision
Fresh confirmation delta was 0.0/85; the predefined gate failed. Stop this candidate without refitting or tuning on confirmation. Retain the shipped model.

# Memory
- Fixed log-peak-shape + margin-availability head: development delta +.688 and +.670 /85 for predeclared group-split seeds; proceeds to new-domain confirmation with first seed's frozen model. Historical pool reuse limits these results.

## node-01: predefined logistic head - OK
Plan: seven inference signals, log1p PSR/APCE, missing-margin indicator, fixed regularisation; source-group fit/tune/assessment separation.
Result: seed20260908 n1346/434/470, 76.8382 ->77.5262 /85, F1 .89583->.92708; seed20260909 n1318/493/439,76.9476->77.6180, F1 .88398->.91620. Gains exceed the predeclared .35 development gate and beat the threshold-only controls (+.7747 / +.6704).
Caveat: these are partitions of reused development evidence, not two independent unseen tests. They overlap and do not undo earlier selection bias. Model artifacts were saved before assessment scoring. No assessment-driven revision was made.
Code: scripts/experiment_confidence85.py; outputs working/results.json and candidate-*.json. Next: fixed first-seed model on fresh generated proxy set, no refit.

## node-02: frozen confirmation - STOP / NOT ADOPTED

The frozen first-seed head was evaluated once on 200 newly generated pairs (A70/B70/C40/D20), with no refit or threshold change. Baseline and candidate both scored **80.246058/85: delta 0.000000**. Both rejected all 40 absent pairs and declined one real pair; rejection F1 was .987654 and historical-label AUC was 1.0. Set D credit was .94 for both. The paired, within-stratum 2,000-draw bootstrap interval was [0, 0]; this describes this finite pool, not universal equivalence.

The alternative submitted-correctness AUC decreased from .793296 to .782123. The predeclared confirmation gate (delta >= .35 and lower interval >= 0) failed. **Do not promote this head.** The development improvements did not demonstrate transfer; the fresh proxy also offers little remaining rejection headroom. Keep the shipped policy and checkpoint unchanged. This experiment does not test the candidate-bank-plus-none architecture proposed in #76.

See `confirmation_results.json` for all rubric components and provenance hashes. The generated proxy is not the official blind set. Validation: 39 targeted tests passed, including fit/tune isolation, label-free inference and paired rubric parity.

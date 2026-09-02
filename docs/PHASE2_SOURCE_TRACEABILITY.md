# Phase 2 official-source traceability

This file is the source map for the Phase 2 submission requirements. It does not replace `.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md`; it points each requirement family back to the organizer material used to ground that document.

## Authoritative source

Applied Materials — Problem Statement 2, Phase 2 task materials folder:

https://interinstitutional-my.sharepoint.com/:f:/g/personal/sourabh_i4c_in/IgAcHvZyl5QISJT0RIfvgFOcAc7Il3foDNtUgRe9BI6F_ps?e=UrYBJn

## File → requirement map

- `Applied Materials_Phase 2_Task.pptx`
  - slide 4: blind-set composition (A/B/C/D)
  - slide 5: output contract, `register.py` entry point, reference machine, runtime, ZIP contents
  - slide 6: scoring weights and +6/+4 bonus conditions
  - slide 7: localisation and pose credit tiers
  - slide 8: rejection semantics
  - slide 9: allowed/disqualified methods and data-use rules
  - slide 10: timeline and freeze point
  - slide 11: theta definition, score-column semantics, nearest-to-centre rule, tie-breakers

- `Applied Materials_Prompt for phase 2 dataset.docx`
  - §2.2: theta sign convention
  - §2.3: scale = `z` semantics and absent-row zeroing
  - §3: geometry requirements R1–R5
  - §3.1: resampling-quality evidence
  - §4: absent-pair/decoy design
  - §5: post-write label-verification gate and margin floor (`>=0.02`, prefer `>=0.12`)
  - §5.1: naive-baseline calibration target (`0.30–0.55` mean present-pair credit)
  - §6: determinism and `REPORT.md`
  - §7: generator deliverables
  - §8: generator grading weights

- `AMP_Phase 2 material/README.md`
  - worked-set calibration guidance and reference-generator notes
  - per-pair verification/calibration evidence for the organizer example set

- `AMP_Phase 2 material/ground_truth.csv`, `manifest_jury.csv`, `baseline_calibration.txt`
  - machine-readable worked-set ground truth, generation parameters and baseline calibration evidence

## Repository rule

When a repository document conflicts with these official materials, `.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md` and the organizer source above govern. Do not infer new rubric gates from historical experiment notes.

Organizer reference/sample data must remain validation-only; do not train or tune on it.
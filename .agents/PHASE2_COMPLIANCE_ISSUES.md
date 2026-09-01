# Phase 2 compliance issues - grounded in the official materials

Companion to ORGANIZER_PHASE2_GROUND_TRUTH.md (authoritative source list there).
Every issue cites the official material that proves it. Verified 2026-09-01 on
branch phase2-pose-accuracy @ a10c351. Severity: BLOCKER = would lose points or
disqualify; GAP = would lose points; MINOR = polish. None of these are invented;
each is a factual mismatch between the repo and the official spec/materials.

## BLOCKERS

### B1. The +6 Set D bonus was wrongly written off in our docs
Official: slide 6 - '+6 Set D credit >= 0.40 with Sets A-C >= 0.50'. Our
PHASE2_STATE.md:152-154 claims it 'requires Sets A-C above 95' - no such
condition exists in any official material, and our measured numbers clear the
actual bar component-wise (A loc 0.97, B loc 0.82, F1 0.91, D 0.938). The bonus
also acts as the second tie-breaker after Set B credit (slide 11), which makes
claiming it strategically relevant even beyond the +6.
Fix: docs corrected by this audit; nothing to change in code.

### B2. failure_analysis.pdf page limit — RESOLVED: COMPLIANT (2 pages exactly; two independent probes agree). Keep the limit if regenerated.
Verified 2026-09-01 (local branch phase2-compliance-fixes): two independent
probes agree the PDF is EXACTLY 2 pages (page-object count 2, /Count 2). No fix
needed; keep the limit in mind if the document is ever regenerated.
Official: slide 5 - 'failure_analysis.pdf, max 2 pages'. Repo: the file exists
but is ~2 pages by the /Count probe only - an exact page count could not be
verified in-session (no pdf tooling). MUST be confirmed and, if over, re-cut
before shipping the zip.
Fix: verify page count; rewrite to 2 pages max if needed.

### B3. Official sample pairs + Set-A-like unscored samples not ingested — RESOLVED (validation pass done; see evidence below)
Official: slide 10 (T+2: 'Sample pairs published: pairs.csv format plus three
unscored sample pairs with full ground truth') and slide 5 ('Three sample pairs
with full ground truth ship with this addendum'). The SharePoint folder's
Dataset_AMP_Phase 2 subfolder (20 items) was downloaded but not yet inventoried
in-session. The three official sample pairs are the only pairs with real
organizer GT we are allowed to validate the I/O contract against.
Fix: inventory Dataset_AMP_Phase 2; run register.py on the three sample pairs;
assert output format matches the contract exactly.

RESOLUTION EVIDENCE (2026-09-01, branch phase2-compliance-fixes, local run only —
no organizer data committed, per slide 9):
- Dataset_AMP_Phase 2 inventoried: 20 search-side PNGs (p001-p018 grayscale
  1000x1000 uint8, p019/p020 3-channel Set D) — matches the spec exactly.
- register.py ran END-TO-END on all 20 jury pairs (official pairs.csv format,
  absolute paths, --threads 4, shipped weights, threshold 0.18, band=False):
  every pair_id exactly once, found/zero-fill contract respected.
- Correctly DECLINED all 4 absent pairs (p015-p018 found=0), zero false alarms;
  absent-pair confidence 0.000-0.061 vs present >= 0.479 — the separation the
  naive baseline cannot achieve (official gap -0.055).
- Scored against the official ground_truth.csv with the official rubric
  (18 grayscale pairs, sample-sized): localisation 39.27/40, pose 19.75/20,
  rejection F1 1.000 -> 15/15, calibration AUC 1.000 -> 10/10, Set D credit
  0.90 -> +6 bonus condition (>=0.40 with A-C >= 0.50) satisfied on this set.
  SUBTOTAL 84.02/85 on the jury worked set (naive baseline: 0.800 mean credit).
- Runtime: median 3.04 s, p90 3.34 s, max 6.26 s (local Mac, 4 threads) — inside
  the <=5 s median budget; reference-machine (4-core x86) number still unknown,
  treat local figures as optimistic.
- CAVEAT recorded: this is a 20-pair WORKED jury set built 'easy on purpose'
  (README 5: Set A naive-baseline credit 1.000) for I/O validation — it is NOT
  the 200-pair blind set, and the severity-skew recommendation for the real set
  (README 5) means these numbers are an UPPER BOUND, not a prediction. No
  threshold or hyperparameter was tuned on this data (single pass, shipped
  config), keeping the run inside the slide 9 rules.

## GAPS

### G1. Blind-set composition is FIXED A70/B70/C40/D20 - our sampling docs must say so
Official: slide 4. Our docs describe the grade as '200 pairs' without the fixed
composition (PHASE2_STATE.md 173+ bootstraps unstratified 200-pair draws; PR #24
reviews independently demanded the stratified draw). The composition is now
OFFICIAL, not inferred: A=70, B=70, C=40, D=20, F1 over the 180 grayscale pairs.
Fix: eval --sample stratified 70/70/40 (already PR #24 review item), plus doc
update citing slide 4.

### G2. Set B severity is 4 UNDISCLOSED levels, real set skewed harder than our ext_p2
Official: slide 4 (four undisclosed severity levels; polygon scaling +/-20%),
README 5 (jury recommends the real 200-pair set shift B toward severity 3-4 and
raise A's floor). Our ext_p2 B distribution is our own choice and skews easy.
Implication: our severity-robustness margin is thinner than our self-scores
suggest.
Fix: weight severity 3-4 in the next generator run; re-test rejection F1 under a
B-heavy severity mix before quoting P(bonus).

### G3. Absent pairs: official Set C is 'plausible and periodically similar'
Official: slide 4 - C is 'a different die region of the same architecture'.
DOCX section 4 warns the naive decoy (same zone geometry) scores HIGHER peaks
than present pairs and requires the decoy to carry large-scale structure the
search canvas lacks, with an honest audit of the residual signature. Our
ext_p2 C-set decoys follow our own design; our docs do not record the decoy
signature audit the DOCX demands.
Fix: document ext_p2's decoy design + signature (or run the audit) so the
generator component's 15 absent-pair points are defensible.

### G4. Efficiency is a RELATIVE QUARTILE ranking, not an absolute bar
Official: slide 6 - efficiency 5 pts is 'relative quartile ranking on median
wall clock'. Our docs treat <=5 s median as the target (correct for the budget,
slide 5) but the 5 POINTS come from the field's quartiles. At ~5.0-5.2 s the
wide model risks the bottom quartile if the field ships leaner decodes.
Fix: no code change; strategy note - the coarse-sweep elimination (#7) is the
lever, as already planned.

### G5. Calibration AUC definition — PARTIALLY RESOLVED: submitted-output variant fixed (issue #27); organizer definition still ambiguous, question reserved for the T+3 window
Official: slide 6 - 'AUC of your score column against per-pair correctness on
the blind set'. The organizer definition of per-pair 'correctness' for absent
pairs is still not specifiable from the materials. eval_ext.py now computes
two separately named variants (issue #27):
- `calibration` — present-only, err <= 5 px (historical, unchanged);
- `calibration_submitted` — submitted-output correctness: a present pair is
  correct only if the score clears the found threshold AND localises within
  5 px (a declined present pair forfeits its measurement and is NOT correct);
  an absent pair is correct only if it was actually rejected (found=0
  submitted). The earlier `calibration_all_pairs` variant was WRONG
  (issue #27): it labelled every absent pair correct regardless of the
  submitted found decision — it was removed, not replaced, so no historical
  number is silently redefined.
Fix (remaining): prepare the T+3 question to the organizers asking which
correctness binary the blind-set calibration AUC uses; implement the
organizer's answer separately with a citation when it arrives.

### G6. Submission zip checklist — RESOLVED: scripts/check_submission_zip.py now audits an
actual ZIP (artifact-level); repo-tree runs are --preflight only
Official: slide 5. Usage: `python scripts/check_submission_zip.py
dist/submission.zip` extracts the artifact to a temp dir and audits ONLY the
extraction: required root layout (register.py, infer.py, requirements.txt,
generate_dataset.py, failure_analysis.pdf, weights/driftsense.pt -- the layout
the organizer command `python register.py --input pairs.csv --output
predictions.csv` needs, since register.py resolves weights relative to its own
location via infer.DEFAULT_WEIGHTS); an actual torch.load(weights_only=True)
of the shipped checkpoint asserting a 'model' key (mirrors
tests/test_checkpoint_safety.py; SKIPs honestly, never false-PASSES, when the
auditing python has no torch); `python register.py --help` and
`python generate_dataset.py --help` smoke tests executed from the extraction
dir (outside the checkout); a network-marker scan over register.py, infer.py
AND every local .py they transitively import within the extraction; PDF page
count (max of /Type /Page objects and /Count, fail over 2); requirements pin
check read from the ZIP. Running with no argument or --preflight audits the
repository tree instead, labels every line and the summary PREFLIGHT, and is
explicitly NOT evidence that the submitted artifact is compliant. The
documentation predicate parses the AST for a real module docstring that names
its arguments -- a shebang or a --help string literal cannot pass it.
Honest limits: (a) the generate_dataset.py --help smoke test additionally
requires generator/src/ inside the ZIP (driftsense/generate.py and
driftsense/presets.py import src.* from REPO_ROOT/generator), so a ZIP
omitting generator/ fails the smoke test -- ship it; (b) offline-runtime
evidence is limited to the --help smoke tests; no network-isolated execution
test is claimed. ACTION before shipping: run the artifact audit on the final
ZIP with ./venv313/bin/python and confirm 0 FAILED.

### G7. gt_scale semantics — RESOLVED: tests/test_scale_semantics.py pins z-semantics (doc + decode path)
Official: slide 5 scale 'nominally in [8, 12]' (= z); DOCX 2.3 fixes the
absent-row reading; jury ground_truth.csv shows scale in [8,12] on present rows.
Our decode emits z-semantics scale (verified in code and CSVs). Add a unit test
pinning this so a future refactor cannot flip to 1/z silently.

## MINOR

### M1. theta convention - verified MATCH (proof recorded)
Official: DOCX 2.2 R(theta) = [[cos, sin], [-sin, cos]] on y-down coords, CCW
positive as displayed. Ours: cv2.getRotationMatrix2D (driftsense/generate.py:284).
PROVEN numerically identical this session (both matrices equal at theta=30 deg;
official R maps +x to (0.866, -0.5) = upward = CCW as displayed). No action; keep
the proof in ORGANIZER_PHASE2_GROUND_TRUTH.md section 5.

### M2. found=0 zero-fill contract - verified compliant
Official: slide 5 ('When 0, write 0 in the pose columns'). Ours: register.py
zero-fills all pose fields when found=0 (verified lines 108-142). The eval-side
masking fix (PR #24 4dbf171) makes our scorer match this contract. No action.

### M3. Tie-breaker order differs from our assumptions
Official: Set B credit -> rejection F1 -> median error -> median runtime
(slide 6), and the bonus is the second tie-breaker after Set B credit (slide 11).
Our docs never recorded an order. Strategy note only.

### M4. 'Nearest-to-centre' rule confirmed as the official ambiguity policy
Official: slide 11 FAQ. Our decode already breaks ties toward centre; document
this alignment in TRAINING.md/policy docs when touched next.

## Cross-reference

- Score provenance issue (75.92 vs 76.23 etc.): tracked in PR #24 reviews and
  .agents/RESCORE_MASKING.md - not duplicated here.
- EMA/resume and defaults-parity blockers: tracked in PR #24 reviews.
- Nothing in this register is speculative; items without a code fix say so.
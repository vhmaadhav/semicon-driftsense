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

### B2. failure_analysis.pdf exceeds the 2-page limit
Official: slide 5 - 'failure_analysis.pdf, max 2 pages'. Repo: the file exists
but is ~2 pages by the /Count probe only - an exact page count could not be
verified in-session (no pdf tooling). MUST be confirmed and, if over, re-cut
before shipping the zip.
Fix: verify page count; rewrite to 2 pages max if needed.

### B3. Official sample pairs + Set-A-like unscored samples not ingested
Official: slide 10 (T+2: 'Sample pairs published: pairs.csv format plus three
unscored sample pairs with full ground truth') and slide 5 ('Three sample pairs
with full ground truth ship with this addendum'). The SharePoint folder's
Dataset_AMP_Phase 2 subfolder (20 items) was downloaded but not yet inventoried
in-session. The three official sample pairs are the only pairs with real
organizer GT we are allowed to validate the I/O contract against.
Fix: inventory Dataset_AMP_Phase 2; run register.py on the three sample pairs;
assert output format matches the contract exactly.

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

### G5. Calibration AUC is computed against 'per-pair correctness' - define ours identically
Official: slide 6 - 'AUC of your score column against per-pair correctness on
the blind set'. Our eval uses err <= 5 px as 'correct' for present pairs and
excludes absent pairs from AUC (eval_ext.py). The official correctness binary
could plausibly include absent pairs (correct = correctly rejected). Not
specifiable further from the materials; flag for the organizers' T+3 question
window.
Fix: prepare the question; optionally compute both variants locally.

### G6. Submission zip must contain requirements.txt from pip freeze + documented generate_dataset.py
Official: slide 5. Repo: requirements.txt exists (pinned, pip-freeze format).
generate_dataset.py exists with --help docs. ACTION before shipping: regenerate
requirements.txt from the EXACT ship venv and confirm the zip contents checklist
(register.py, weights/, requirements.txt, generate_dataset.py,
failure_analysis.pdf).

### G7. gt_scale semantics confirmed correct - lock it with a test
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
# ORGANIZER PHASE 2 GROUND TRUTH - Applied Materials, Problem Statement 2

Status: AUTHORITATIVE. Compiled 2026-09-01 from the official Phase 2 task
materials (organizer SharePoint folder, downloaded via browser same day).
Every claim below cites its source file. Where this file disagrees with any
other doc in this repo, THIS FILE WINS and the other doc must be corrected.

## 0. Materials inventory (proof of access)

Source: OneDrive folder 'Applied Materials_Problem Statement 2_Phase 2'
(share link provided by organizers), downloaded as OneDrive_1_9-1-2026.zip
(36.6 MB, 62 files). Contents verified:

- 'Applied Materials_Phase 2_Task.pptx' (517 KB, 12 slides)
- 'Applied Materials_Prompt for phase 2 dataset.docx' (28.9 KB, 8 sections)
- 'AMP_Phase 2 material/': the jury's WORKED 20-pair reference implementation -
  pairs.csv, ground_truth.csv (WITHHOLD-grade scoring key), manifest_jury.csv
  (32 generation-parameter columns), baseline_calibration.txt, README.md,
  contact_sheet.png, reference/p001-p020.png (1000x1000, 1 nm/px),
  search/p001-p020.png (1000x1000, z nm/px), generator/ (transcribed upstream
  src/ + new src/phase2_pipeline.py), requirements.txt
- 'Dataset_AMP_Phase 2/': 20-item dataset folder

## 1. The blind set (slide 4) - what is actually graded

200 organizer-generated pairs, teams never see the images, only their scores.
Same geometry as Phase 1: grayscale 1000x1000, [0,0] top-left. Composition is
FIXED, not sampled:

| Set | Pairs | Content | Feeds |
|---|---|---|---|
| A | 70 | Nominal pose, reference present, noise like Phase 1 sample prompt, full [8,12]x and +/-5 deg range | localisation + pose |
| B | 70 | Degraded, present, charging/scan-distortion/defocus/elevated shot noise/polygon scaling +/-20%, FOUR UNDISCLOSED severity levels | localisation + pose |
| C | 40 | Absent - a different die region of the SAME architecture ('plausible and periodically similar'). found=0 | rejection F1 |
| D | 20 | Optical, RGB 3-channel, reference present. Bonus only | +6 bonus |

Slide 4 also states: noise models from public SEM literature; the CATEGORIES are
disclosed but the parameters and the severity ladder are NOT.

Jury README corroboration (20-pair worked set): sets were A8/B6/C4/D2 = 20 with
16/20 = 20% present, zoom list hits BOTH endpoints 8.00 and 12.00, theta spans
-4.9..+4.9 with three pairs at exactly 0.00, 9 architecture presets across both
families. The 200-pair set is built to the same recipe (README section 5).

## 2. Output contract (slide 5)

- ONE entry point: python register.py --input pairs.csv --output predictions.csv
- One row per pair: pair_id (as supplied), x, y (match centre, wide-search
  coords, float, subpixel allowed), theta (degrees, CCW positive, about the
  match centre), scale (recovered down-scaling factor, nominally in [8,12]),
  found (1/0 - when 0, write 0 in the pose columns), score (own confidence, any
  monotonic scale). Every pair_id EXACTLY once; a missing row scores zero.
- Reference machine: 4-core x86 CPU, 8 GB RAM, NO GPU, NO NETWORK, Python 3.11.
- Weights ship inside the zip - nothing downloads at run time.
- Runtime budget: MEDIAN <= 5 s per pair; hard timeout 20 s - that pair scores
  zero (slide 10 wording: code frozen at T+7, organizers run everything T+8..9).
- Also in the zip: requirements.txt from pip freeze; generate_dataset.py
  documented; failure_analysis.pdf max 2 pages.

## 3. Scoring (slide 6) - exact weights and bonus conditions

| Component | Pts | Definition (slide 6/7 verbatim semantics) |
|---|---|---|
| Localisation | 40 | Sets A and B, PRESENT pairs. Tiered: <=1 px 1.00, <=2 px 0.80, <=3 px 0.60, <=5 px 0.40, >5 px 0.00. Total = (0.45*A + 0.55*B) * 40 |
| Pose | 20 | Scale 10 + rotation 10. 'Scored only where the location was already correct' (slide 7). Scale tiers: <=1% 1.00, <=2% 0.60, <=5% 0.30; rotation: <=0.25 deg 1.00, <=0.5 deg 0.60, <=1.0 deg 0.30 |
| Rejection | 15 | F1 on the found flag across ALL 180 grayscale pairs (A+B+C). Slide 8: 'A team that never rejects anything scores zero here' - that is only true with REJECT as the positive class. Slide 8 also says FP and FN 'weigh equally in F1' and the jury pack breaks them out separately |
| Calibration | 10 | AUC of the score column against per-pair correctness on the blind set |
| Efficiency | 5 | RELATIVE QUARTILE ranking on median wall clock per pair |
| Generator, citations, failure analysis | 10 | Carried forward from Phase 1, re-judged under Phase 2 conditions |
| BONUS | +10 | +6 IF Set D credit >= 0.40 AND Sets A-C >= 0.50. +4 IF rejection F1 >= 0.90. Tie-breakers in order: Set B credit -> rejection F1 -> median error -> median runtime (slide 6). Bonus cannot lift ranking above 100 but is the SECOND tie-breaker after Set B credit (slide 11 FAQ) |

### Reading the +6 condition (grounded in the official materials)

Slide 6 (task deck): '+6 Set D credit >= 0.40 with Sets A-C >= 0.50'. The
natural reading, parallel to how Set D credit is elsewhere a per-set mean
credit, is per-set mean credits: our Set D measures 0.938 (PHASE2_STATE.md),
Set A loc ~0.97, Set B loc ~0.82, Set C (rejection F1) 0.91 - every one clears
its bar. Even the harshest component-points reading clears: A+B loc 35.5/40,
pose 18/20, F1 13.6/15. An earlier briefing-call recollection ('Sets A-C above
95' / 'scores must be extremely good') has NO support anywhere in the official
materials and is RETIRED from planning; the written rubric governs. The +6 is
REACHABLE. Disambiguation note: the official materials DO contain a '0.30-0.55'
band - that is the generator difficulty-calibration target (naive-baseline mean
credit on present pairs, DOCX section 5.1; jury README section 5), unrelated to
any bonus gate. A single confirming question in the T+3 window remains good
practice, but planning proceeds on the written rubric.

## 4. Rejection, ambiguity, and the score column (slides 8 and 11)

- Slide 8 defines the four found-flag outcomes and states most Phase 1 methods
  'return an unconditional argmax and will score near zero on 40 pairs'.
- Slide 11 FAQ: 'Is the score column compared between teams? No. It is only
  evaluated for internal monotonicity against your own correctness.'
- Slide 11 FAQ: 'What if two regions match equally well? Ground truth is the
  true placed instance, and the Phase 1 nearest-to-centre rule still decides.
  Set B deliberately includes pairs where the global-argmax answer differs from
  the nearest-to-centre answer.' (Our winner-margin instrumentation exists for
  exactly this; nearest-to-centre is already our decode's tie-break.)
- Slide 11 FAQ: theta is 'rotation of the reference pattern as it appears in the
  wide-search image. Counter-clockwise positive, measured about the match centre.'
- Slide 11 FAQ: 'Do we need to regenerate our dataset? You should' - fixed-10x
  always-present Phase 1 data validates nothing about rejection.

## 5. Ground-truth conventions fixed by the organizers (DOCX sections 2.2-2.3)

- theta: degrees, CCW positive as displayed, about the match centre. Mapping
  p_search = (1/z) R(theta) (p_canvas - c_canvas) + c_search with
  R(theta) = [[cos t, sin t], [-sin t, cos t]], t = radians(theta).
  PROOF our pipeline matches: cv2.getRotationMatrix2D's linear part is
  numerically IDENTICAL to this matrix (verified: both equal
  [[0.866, 0.5], [-0.5, 0.866]] at theta=30 deg; the official matrix maps +x to
  (0.866, -0.5), i.e. up = CCW as displayed). Our generator uses
  cv2.getRotationMatrix2D (driftsense/generate.py:284) - CONVENTION MATCHES.
- scale: 'the recovered down-scaling factor, nominally in [8, 12]' (slide 5) -
  i.e. z, the search nm/px, NOT 1/z. DOCX 2.3 fixes the absent-pair reading:
  for present=0 rows the scale column carries z 'NOT the reference-to-search
  linear factor 1/z. Both are defensible readings... they differ by a factor of
  ~100... fixed here by fiat.' Our decode and our ext_p2 CSVs both carry z-semantics
  (matching the official ground_truth.csv format where present pairs list
  scale in [8,12]).
- Absent rows: present=0 and ALL FOUR pose columns 0 (DOCX 2.3; confirmed in the
  jury's ground_truth.csv: p015-p018 are 0,0.0,0.0,0.0,0.0).

## 6. What the jury's own 20-pair calibration proves (baseline_calibration.txt)

Naive ZNCC baseline (brute force over a 0.5x / 1.0 deg grid), official numbers:

- Set A mean credit 1.000 (median 0.40 px) - the SAMPLE set is intentionally easy;
  the three participant sample pairs are 'Set-A-like and unscored' (README 5).
- Set B mean credit 0.467; severity 3-4 defeat it outright (p011/p012/p014 -> 0).
- Set D mean credit 1.000 (median 0.15 px) on the optical analogue.
- Present ZNCC peaks 0.338-0.956; absent peaks 0.279-0.393; separation gap
  -0.055 - NEGATIVE, deliberately: 'a naive matcher cannot distinguish too
  degraded to find from not there, which is exactly the discrimination the
  15-point rejection block and the 10-point calibration block are meant to
  reward' (README 5). IMPLICATION: threshold-on-ZNCC alone cannot win Set C;
  a learned/confidence signal that survives severity (ours does - see the
  eval tables) is the intended differentiator.
- Rejection @0.55: precision 1.00, recall 0.81, F1 0.897 (3 FN, all severity 3/4).
- Pose on the coarse grid: scale within 3.0% worst / 1.0% median; theta within
  1.10 deg worst / 0.35 deg median. 'Since the published tolerances are 1%/0.25 deg
  for full credit, a finer search or peak interpolation is required to earn top
  marks - which is the intended incentive' (README 5). Our subpixel refine +
  3-hypothesis decode already targets this.
- Label verification gate (README 6): every present pair rigid-template-verified
  at its own labelled pose, global peak within 3 px of the label - all 16 verify
  at 0.11-1.04 px, margins 0.118-0.468. The DOCX floor is margin >= 0.02 to ship,
  prefer >= 0.12 (section 5) - the jury set clears the PREFERRED bar everywhere.
  Our own ext_p2 generator ships the same concept; margins deserve a check.
- For the 200-pair set the jury recommends shifting Set B severity toward 3-4
  and raising Set A's floor (README 5) - expect the real set to be HARDER than
  ext_p2's B mix. Severity-forward robustness matters more than nominal-pair
  polish.

## 7. The generator component (DOCX) - what the 10 judged points actually grade

Weighting (DOCX section 8): geometry correct (R1-R5, labels exact + verified) 35,
verification gate implemented + reported 20, resampling quality (no aliasing at
z=12) 10, absent-pair design + honest signature audit 15, calibration in band
(0.30-0.55 overall mean credit) 10, determinism/contact sheet/REPORT.md/
limitations 10. Report: <= 3 pages, must contain the 6 listed items, written
EARLY and iterated ('an empty or cosmetic limitations list scores zero').
Deliverables (DOCX section 7): generate_phase2.py (--output-dir, --seed,
--pairs), baseline.py, score.py, contact_sheet.py, src/, REPORT.md, output/ with
pairs.csv + ground_truth.csv + manifest.csv + baseline_calibration.txt +
contact_sheet.png + reference/ + search/. Runtime target: 20 pairs < 5 min laptop
CPU; hold one z=12 canvas at a time (>13k x 13k px).

Also required to SHIP (slide 5): failure_analysis.pdf (max 2 pages) inside the
submission zip.

## 8. Allowed vs disqualified (slide 9)

ALLOWED: extending Phase 1 to search the disclosed ranges or invariant
formulations; regenerating data with z in [8,12], theta in +/-5, absent pairs;
HARD-CODING the disclosed bounds ('intended, not a loophole'); augmentation,
retraining, hyperparameter/threshold changes; classical, learned, or hybrid -
judged equally.

DISQUALIFIES (no appeal): network access during the scored run; hard-coding,
file-name fingerprinting, or reading outside supplied paths; 'a method materially
different from your Phase 1 declared approach - Phase 2 tests the evolved Phase 1
method, not a new one'; proprietary/non-public layout data in the generator;
'Mixing organizer test data into training, in either direction.' Phase 1 rules
stand: Python only, pip-freeze zip, documented generator with cited sources.

NOTE on the last clause: the jury's 20 reference pairs + ground truth are
organizer material. NOTHING from AMP_Phase 2 material/ (images, GT, manifest)
may be trained on or tuned against beyond what is publicly given to all teams.
Our ext_p2 hold-out is our own generator's test split with verified zero
pair_sha256 overlap - that discipline stands.

## 9. Timeline (slide 10)

T+0 addendum released; T+2 sample pairs published (3 unscored, with GT); T+3
I/O-contract questions close; T+7 submission due 23:59, code frozen; T+8..9
organizers run everything on the reference machine; T+10..11 top 10 announced.

## 10. Source files (local extraction)

- Extracted to /tmp/amp_p2 (session scratch): PPTX slide text, DOCX text,
  AMP_Phase 2 material/*. The zip remains at ~/Downloads/OneDrive_1_9-1-2026.zip.
- Slide references above are by slide number in 'Applied Materials_Phase 2_Task.pptx';
  DOCX references by section in 'Applied Materials_Prompt for phase 2 dataset.docx'.
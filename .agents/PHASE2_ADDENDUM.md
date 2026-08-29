# SEMICON India Hackathon 2026 / Drift-Sense — Project Source Addendum

**Prepared from:** the 10 screenshots supplied in this project, the supplied Drift-Sense repository/source files, and public SEMICON India / i4C information checked on 27 August 2026.

## Evidence policy

This file deliberately distinguishes three kinds of information:

1. **Screenshot-observed facts** — text visibly present in the supplied screenshots. These screenshots show a Microsoft Teams briefing deck marked **Applied Materials Confidential**.
2. **Project-file facts** — information present in the supplied repository/source files.
3. **Externally verified facts** — information checked against public SEMICON India and i4C pages.

No statement about the spoken content of the supplied MP3 is made unless it can be reliably transcribed. At present, only its file metadata is recorded (see “Audio source”).

---

# 1. Screenshot source — Applied Materials Phase 2 briefing

All ten screenshots are 1920×1200 PNG captures dated 27 August 2026. The visible slides form a coherent Phase 2 addendum for the Applied Materials Drift-Sense problem.

## Screenshot: `Screenshot 2026-08-27 190618.png`

### Visible title
**PHASE 2 — ADDENDUM: Registration under Unknown Pose**

### Visible statement
“Everything in the Phase 1 problem statement continues to apply. This section states only what is added or changed.”

### Interpretation limited to the slide
Phase 2 is presented as an extension of Phase 1 rather than a replacement problem. The new focus is registration when the relative pose is not fully known.

---

## Screenshot: `Screenshot 2026-08-27 190755.png`

### Visible title
**What Changes in Phase 2**

### Phase 1 vs Phase 2 changes shown on the slide

| Item | Phase 1 (as issued) | Phase 2 (this addendum) |
|---|---|---|
| Zoom ratio | Exactly 10×, given in the statement | **Unknown — uniform in [8×, 12×]** |
| Rotation | Injected as noise, 1–3° | **Unknown, ±5° — and must be reported** |
| Is the reference present? | Always present in the search image | **~20% of pairs contain no true instance** |
| Required output | x, y | **x, y, θ, s, found, score** |

The slide says the disclosed scale and rotation ranges are exact search bounds, not a loophole to hard-code a single pose.

### Consequence for Drift-Sense
A Phase 2 system must do more than localization. It must:
- localize the target when present;
- recover rotation;
- recover scale;
- detect when no true instance exists;
- output a confidence score.

---

## Screenshot: `Screenshot 2026-08-27 191159.png`

### Visible title
**Phase 2 Dataset — 200 Blind Pairs, Organizer-Generated**

The slide states:
- same geometry as Phase 1;
- grayscale;
- 1000×1000 px;
- coordinate origin `[0,0]` is top-left;
- teams do **not** see the blind images, only their scores.

### Dataset composition shown

| Set | Count | Description | Used for |
|---|---:|---|---|
| **Set A** | 70 pairs | Nominal pose; reference present; noise comparable to Phase 1 sample prompt; full `[8,12]×` and `±5°` range | Localization + pose |
| **Set B** | 70 pairs | Degraded; reference present; charging, scan distortion, defocus, elevated shot noise, polygon scaling ±20%, in four undisclosed severity levels | Localization + pose |
| **Set C** | 40 pairs | Absent; no true instance; a different die region of the same architecture, plausible and periodically similar; correct answer is `found = 0` | Rejection F1 |
| **Set D** | 20 pairs | Optical (bonus); RGB 3-channel optical-microscope analogue; reference present | Bonus only, +6 pts |

The footer says the noise models are drawn from public SEM literature; categories are disclosed but parameters and the severity ladder are not.

---

## Screenshot: `Screenshot 2026-08-27 191722.png`

### Visible title
**Output Contract and Run Environment**

### Required entry point shown

```bash
python register.py --input pairs.csv --output predictions.csv
```

### `predictions.csv` contract shown

One row per input pair:

| Field | Meaning on slide |
|---|---|
| `pair_id` | as supplied in `pairs.csv` |
| `x, y` | match centre in wide-search coordinates; float; subpixel allowed |
| `theta` | rotation in degrees, CCW positive, about the match centre |
| `scale` | recovered down-scaling factor, nominally in `[8,12]` |
| `found` | `1` or `0`; when `0`, write `0` in pose columns |
| `score` | team's own confidence, any monotonic scale |

The slide warns: **every `pair_id` exactly once; a missing row scores zero.**

### Reference machine shown
- **4-core x86 CPU**
- **8 GB RAM**
- **No GPU**
- **No network**
- **Python 3.11**
- Model weights must ship inside the ZIP; nothing may download at runtime.

### Runtime budget shown
- **Median ≤ 5 s per pair**
- **Hard timeout 20 s**; that pair scores zero.

### Additional files shown as required in ZIP
- `requirements.txt` from `pip freeze`
- documented `generate_dataset.py`
- `failure_analysis.pdf`, maximum 2 pages

The slide also says three unscored sample pairs with full ground truth ship with the addendum for validating the I/O contract.

---

## Screenshot: `Screenshot 2026-08-27 192410.png`

### Visible title
**Phase 2 Scoring — 100 Points, plus 10 Bonus**

The slide states that these weights replace the Phase 1 weights for Phase 2 only.

### Scoring categories shown

| Category | Points | Visible definition |
|---|---:|---|
| Localization | **40** | Sets A and B, present pairs; tiered credit at 1 / 2 / 3 / 5 px; weighted `0.45·A + 0.55·B` |
| Pose recovery | **20** | Scale 10 pts, rotation 10 pts; scored only where location was already correct |
| Rejection | **15** | F1 on `found` flag across all 180 grayscale pairs; never rejecting scores zero here |
| Confidence calibration | **10** | AUC of the submitted score column against per-pair correctness on the blind set |
| Efficiency | **5** | Relative quartile ranking on median wall-clock time per pair |
| Generator, citations & failure analysis | **10** | Carried forward from Phase 1 and re-judged under Phase 2 conditions |

### Bonus line shown
**BONUS +10**
- **+6** if Set D credit ≥ 0.40 with Sets A–C ≥ 0.50
- **+4** if rejection F1 ≥ 0.90

The slide lists tie-breakers in this order:
1. Set B credit
2. rejection F1
3. median error
4. median runtime

---

## Screenshot: `Screenshot 2026-08-27 192806.png`

### Visible title
**Credit Tiers: Localization and Pose**

### Localization credit shown
Euclidean error of the reported centre:

| Error | Credit |
|---|---:|
| ≤ 1 px | **1.00** |
| ≤ 2 px | **0.80** |
| ≤ 3 px | **0.60** |
| ≤ 5 px | **0.40** |
| > 5 px | **0.00** |

The slide says the Phase 2 localization total is `0.45·A + 0.55·B`, scaled to 40 points, so the degraded set carries more weight than the nominal set.

### Pose recovery credit shown
Scored only where localization credit is greater than zero.

| Pose component | Full credit 1.00 | Credit 0.60 | Credit 0.30 |
|---|---:|---:|---:|
| Scale relative error `|ŝ-s|/s` | ≤ 1% | ≤ 2% | ≤ 5% |
| Rotation error `|θ̂-θ|` | ≤ 0.25° | ≤ 0.5° | ≤ 1.0° |

---

## Screenshot: `Screenshot 2026-08-27 192936.png`

### Visible title
**Rejection: The Failure Mode That Actually Costs Money**

The slide says:
- Set C contains **40 pairs with no true instance**.
- The correct answer for those pairs is **`found = 0`**.

It visually distinguishes:
- correct grab: present + `found=1`;
- false positive: absent + `found=1`, described as a confident grab on the wrong site that can silently corrupt a measurement;
- false negative: present + `found=0`, which costs a re-scan;
- correct reject: absent + `found=0`, allowing the tool to re-scan.

### Scoring facts shown
- Rejection is worth **15 points**.
- F1 is calculated on the `found` flag across all **180 grayscale pairs**.
- False positives and false negatives both affect F1, although the slide notes they have different real-tool costs.
- A system that never rejects cannot score well on this component.
- Confidence calibration is separately checked for **10 points**.

The slide explicitly predicts this as a field-separating part of Phase 2 because many Phase 1 methods return an unconditional argmax.

---

## Screenshot: `Screenshot 2026-08-27 193119.png`

### Visible title
**What Is Allowed, and What Disqualifies**

### Allowed
The slide explicitly allows:
- extending the Phase 1 method to search disclosed scale/rotation ranges or using an invariant formulation;
- regenerating the team's own dataset with scale in `[8,12]`, rotation in `±5°`, and absent pairs;
- hard-coding the disclosed search bounds `[8,12]` and `±5°`;
- data augmentation, retraining, hyperparameter changes and threshold changes;
- classical, learned or hybrid localization.

### Disqualifies — no appeal
The slide lists:
- any network access during the scored run;
- hard-coding, file-name fingerprinting, or reading outside supplied paths;
- using a method materially different from the team's declared Phase 1 approach;
- proprietary fab or non-public layout data in the generator;
- mixing organizer test data into training in either direction.

The footer says Phase 1 rules still stand, including Python-only submission, a ZIP with a pip-freeze environment, and a documented generator with cited sources.

---

## Screenshot: `Screenshot 2026-08-27 193509.png`

### Visible title
**Phase 2 Timeline**

The slide defines **T** as the day Phase 1 results are announced and the 30 shortlisted teams are notified.

Relative schedule shown:
- **T+0:** addendum released
- **T+2:** `pairs.csv` format plus three unscored sample pairs with full ground truth published
- **T+3:** questions close for I/O-contract clarifications
- **T+7:** submission due at 23:59; code frozen, no resubmission
- **T+8–9:** organizers execute every submission on the reference machine
- **T+10–11:** Top 10 announced; Phase 3 begins

This is a screenshot-derived relative timeline. The public i4C roadmap has absolute dates (see Section 4); both are preserved separately rather than silently reconciled.

---

## Screenshot: `Screenshot 2026-08-27 193542.png`

### Visible title
**Phase 2 FAQ Addendum**

The slide answers:

**Do we rewrite our Phase 1 method?**  
No — extend it. Searching the disclosed scale/rotation ranges or moving to an invariant formulation is acceptable.

**Do we need to regenerate our dataset?**  
The slide says yes/“you should,” because a fixed-10×, always-present Phase 1 generator gives no way to validate rejection logic.

**How exactly is θ defined?**  
Rotation of the reference pattern as it appears in the wide-search image; counter-clockwise positive; measured about the match centre.

**What if two regions match equally well?**  
Ground truth is the true placed instance, and the Phase 1 nearest-to-centre rule still decides. Set B deliberately contains cases where the global argmax answer differs from the nearest-to-centre answer.

**Is the score column compared between teams?**  
No. It is evaluated for internal monotonicity against the team's own correctness; any scale works.

**Does the bonus change the ranking?**  
The slide says it cannot lift a team above 100 for ranking, but is the second tie-breaker after Set B credit.

---

# 2. What the screenshots change for the current Highest in the Room implementation

The supplied project README describes a Phase 1 Drift-Sense system built around:
- a spatial Siamese correlation network;
- long-range context;
- eight-way dihedral test-time augmentation;
- local full-resolution ZNCC refinement;
- confidence output;
- synthetic semiconductor data generation.

The project README states that, under its Phase 1-style evaluation, the system localizes a high-resolution Reference patch inside a low-resolution Search frame, originally under an approximately 10× magnification relationship, and outputs the search-image match centre.

## Phase 2 gaps that are directly implied by the briefing screenshots

The current Phase 1 interface is not sufficient by itself because Phase 2 visibly requires:

1. **Unknown magnification / scale**
   - Search must cover `[8×,12×]`.
   - The recovered scale must be reported.

2. **Unknown rotation**
   - Search must cover `±5°`.
   - Rotation must be reported in degrees, CCW positive.

3. **Absent-reference detection**
   - Approximately 20% of blind pairs are stated to have no true instance.
   - The system needs a calibrated `found` decision rather than unconditional localization.

4. **Confidence calibration**
   - The score is explicitly judged for monotonic relationship with correctness.

5. **CPU-only deployment**
   - The screenshot reference environment is 4-core x86 CPU / 8 GB RAM / Python 3.11 / no GPU / no network.
   - The stated median runtime target is ≤5 s per pair.

6. **Batch file contract**
   - Required interface is `register.py --input pairs.csv --output predictions.csv`.
   - Required fields include `pair_id, x, y, theta, scale, found, score`.

These are requirements visible in the supplied screenshots and should be treated as higher-priority Phase 2 implementation constraints than speculative architecture changes.

---

# 3. Audio source

## Supplied file
`Standard recording 19.mp3`

## Verified metadata
- Duration: **2880.329675 seconds**
- Equivalent duration: approximately **48 minutes 0.33 seconds**
- Format: MP3

## Spoken-content status
A reliable speech-to-text transcript could not be produced in the current tool environment. The available local environment did not contain a speech-recognition model, network installation was unavailable, and the connected media transcription service could not be authenticated.

Therefore **no spoken claims from the recording are included here**. This is intentional: inventing or guessing the meeting dialogue would violate the project requirement to use clear, non-fabricated data.

The ten screenshots captured during the same Microsoft Teams briefing are documented independently above and are usable evidence regardless of the unavailable audio transcript.

---

# 4. Publicly verified SEMICON India Hackathon 2026 details

The following details were checked against the public SEMICON India and i4C Hackathon 2026 pages on 27 August 2026.

## Event
**SEMICON India 2026**

Public SEMICON India information lists:
- dates: **17–19 September 2026**
- venue: **Yashobhoomi (IICC), New Delhi / Delhi, India**
- 2026 theme on the public event page: **“Silicon to Systems: Building the Ecosystem”**
- the hackathon is a special feature of SEMICON India 2026.

## Hackathon positioning
The public SEMICON India hackathon page describes the event as a national semiconductor hackathon aimed at solving real-world, industry-defined semiconductor problems and bridging academia with industry.

Challenge areas listed publicly include:
- Chip Design
- AI-enabled Semiconductor Manufacturing
- Yield Optimisation
- AI in EDA
- Advanced Verification

The public page says solutions are evaluated on:
- Innovation
- Technical Depth
- Feasibility
- Deployment Potential

## Participation
The public pages state eligibility for undergraduate, postgraduate and PhD/research students from recognised Indian institutions, with teams generally consisting of **2–4 members**. The i4C page lists B.E./B.Tech/M.Tech/MCA among eligible programmes and says no prior semiconductor-industry experience is required.

## Organisations and partners
Public i4C information identifies:
- **SEMI India** — organiser
- **India Electronics and Semiconductor Association (IESA)** — strategic partner
- **KLA** — industry partner
- **Applied Materials** — industry partner
- **Vellore Institute of Technology (VIT)** — academic partner
- **Inter Institutional Inclusive Innovations Center (i4C)** — hackathon implementation partner

## Public roadmap
The i4C public roadmap lists:
- **24 Jul 2026:** registration opens
- **16 Aug 2026:** registration and Phase 1 submission deadline
- **17–26 Aug 2026:** Round 1 evaluation
- **27 Aug 2026:** Top 30 teams announcement
- **28 Aug 2026:** Round 2 / semifinal begins
- **4 Sep 2026:** Round 2 submission deadline
- **5 Sep 2026:** semifinal evaluation
- **6 Sep 2026:** Top 10 finalists announcement
- **7–12 Sep 2026:** finalist mentoring
- **16–17 Sep 2026:** team arrival / venue onboarding
- **17 Sep 2026:** Grand Finale
- **18 Sep 2026:** winner announcement and awards ceremony

The public i4C page states a **total prize pool of ₹5,00,000**.

---

# 5. Publicly verified Applied Materials problem statement — Drift-Sense

The i4C page names Applied Materials Problem Statement 2:

**“Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools”**

## Problem background
The public explanation says wafer inspection tools must revisit the same site with very high positional accuracy. Thermal effects, vibration and mechanical effects can create navigation drift. Repeating semiconductor structures make incorrect sites look very similar to the intended site, so ordinary template matching can fail, especially in periodic DRAM and FinFET structures.

## Original public Phase 1 task
The public problem page describes:
- find where the Reference pattern appears inside the Search image;
- return the match centre `(x,y)`;
- when several matches exist, use the one closest to the Search-image centre;
- synthetic training data must be generated by teams;
- DRAM-style and FinFET-style structures are accepted;
- independent noise is required for Reference and Search;
- edge-brightening and realistic degradation should be modelled;
- ground-truth coordinates must be recorded;
- public citations are required to justify augmentation/noise choices.

The public page describes the Reference/Search synthetic setup as:
- Reference: high-resolution semiconductor patch;
- Search: lower-magnification wide field;
- both nominally 1000×1000 in the supplied starter setup;
- original nominal scale relationship around 10×.

The supplied Phase 2 screenshots add the stricter unknown-pose, absence/rejection and batch-evaluation requirements described in Section 1.

---

# 6. Highest in the Room — project facts already present in supplied source

The supplied project source identifies:

**Team:** Highest in the Room  
**Members:** Maadhav V H, Nakul T, Sachin A S, Nishanth R S  
**Repository:** `vhmaadhav/semicon-driftsense`

The supplied code audit ranks the repository first in its engineering audit, while explicitly stating that the audit is **not an official competition result**.

The audit describes the implementation as a hybrid of:
- spatial Siamese correlation;
- learned long-range context;
- heatmap + sub-cell offset prediction;
- eight-way dihedral test-time augmentation / cluster voting;
- full-resolution ZNCC verification/refinement;
- synthetic semiconductor data generation;
- extensive streaming/stress evaluation and diagnostic tooling.

The supplied README reports a 0.46M-parameter model and strong self-evaluation results, but those remain **repository-reported/self-generated measurements**, not official Phase 2 blind scores.

---

# 7. Other Phase 2 team/repository data already supplied in this project

The project source currently maps the following repositories:

| Team | Repository |
|---|---|
| Atlas | `kushal-script/drift-sense` |
| Bhoochadae | `avaramahmood/semicon_driftsense` |
| SUNRISE | `Suryooday/Driftsense` |
| The Learning Loop | `Jaswanthj006/Semicon-Submission` |
| Volt Visionaries | `RHUDHRESH/LatticeRank` |
| The T guys | `itsAryan-devop/drift-sense` |
| TECHTONICS | `DK-A/Techtonics_Drift-Sense_Wafer_Inspection_PS2` |
| Highest in the Room | `vhmaadhav/semicon-driftsense` |
| Silicon Stars | `icy-chidam/i4c` |
| NanoTrace | `AnshuPriya-1/NanoTrace_Semicon_Hackathon` |
| NanoBolts | `aashishniranjanb/Drift-Sense-SEM-Localization` |
| Black_Pearl | `TharunBabu-05/I4C_Drift_sense_D_RAM_submission` |
| SILICOFORGE | likely `Roohith6/drift-sense` (the supplied audit marks this mapping as likely rather than fully confirmed) |

The supplied team file says **ChipUp** and **Team 664** do not yet have confirmed public repositories in that source.

---

# 8. Practical Phase 2 checklist derived only from supplied/verified requirements

For the Highest in the Room repository to satisfy the screenshot-visible Phase 2 contract, the implementation should be checked for all of the following:

- `register.py` batch entry point exists.
- It accepts `--input pairs.csv --output predictions.csv`.
- It outputs every input `pair_id` exactly once.
- Output contains `x`, `y`, `theta`, `scale`, `found`, `score`.
- Scale search/recovery supports `[8,12]`.
- Rotation search/recovery supports `±5°`.
- `found=0` is supported for no-instance pairs.
- Confidence is calibrated enough to rank likely-correct above likely-incorrect predictions.
- It runs under Python 3.11.
- It runs without GPU.
- It requires no network.
- All weights/assets are packaged locally.
- Median target runtime is ≤5 s per pair on the stated 4-core CPU reference environment.
- Per-pair hard timeout risk is below 20 s.
- Generator can create present and absent pairs with Phase 2 pose ranges.
- Generator avoids organizer blind-test leakage.
- Required citations and a ≤2-page failure analysis are included.
- The core approach remains an extension of the declared Phase 1 method, rather than a materially unrelated replacement.

---

# 9. Source provenance

## Supplied project files
- `drift-sense-submission.zip`
- `README.md-of-existing-repo.txt`
- `drift_sense_repo_code_audit_v2.md`
- `few-architectural-suggestions.txt`
- `details-about-teams-and-their-projects.txt`
- `Standard recording 19.mp3`
- ten screenshots named `Screenshot 2026-08-27 *.png`

## Public sources checked
- SEMICON India — Hackathon 2026 page
- SEMICON India — 2026 main event page
- i4C — SEMICON India Hackathon 2026 page
- Applied Materials public events page
- Public Drift-Sense synthetic-data repository page on Hugging Face

## Important qualification
The screenshot briefing contains **Applied Materials Confidential** markings. This project source documents only what is visibly present in user-supplied screenshots. It should not be treated as a public organizer publication unless Applied Materials releases the same information publicly.


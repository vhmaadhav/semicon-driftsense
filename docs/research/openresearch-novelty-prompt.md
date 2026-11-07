# OpenResearch prompt: novelty and paper-readiness for semicon-driftsense

Paste everything below the line into OpenResearch.

---

You are doing an end-to-end research pass on a repository, to find what is novel in it, what prior work it overlaps with, and whether it can become a research paper. Do not write the paper yet. Your output is an evidence-backed novelty report and a go / no-go recommendation. The owner will decide whether to draft.

## 0. Entry point and rules

- Repository: https://github.com/vhmaadhav/semicon-driftsense (branch `main`, about 440 tracked files, about 121 MB with weights). Clone it, then read it. Use `gh` for GitHub reads if available. Do not open issues, PRs, comments or posts anywhere. Keep all output local.
- Read the full text of: `README.md`, `FAILURE_ANALYSIS.md`, `TRAINING.md`, `docs/PHASE3_MEASUREMENT.md`, `docs/PHASE3_COMPLETENESS.md`, `scripts/bundle_docs/phase_3.md`, `generator/REPORT.md`, `phase1/RESULTS.md`, `CITATIONS.md`, `NOTICE`, `results/results.json`. Then read the core code: `driftsense/model.py`, `driftsense/matching.py`, `driftsense/cad_anchor.py`, `driftsense/calibration.py`, `driftsense/policy.py`, `driftsense/verification.py`, `driftsense/engine.py`, `driftsense/generate.py`, `driftsense/gds.py`, `register.py`, `phase3.py`. Skim the rest (`scripts/`, `tests/`, `phase1/`, `generator_i4c/`) for claims the docs do not state.
- Open issues and PRs on the repo are part of the record. Known open items at time of writing: issue #109 (Phase 3 label convention and noise floor), #101 (per-pair raster shear from lattice aliases), #76, #70, #69, #68, #66, #65. Open PRs #102 and #107 are not drafts and are conflicting; they are not shipped. PRs #97 (strip rotation, #88) and #98 (drift-row shrinkage, #89) are merged, although `FAILURE_ANALYSIS.md` still describes #89 as unmerged. Read them and report their status.
- Confidential and restricted material. Do not put any of the following into your output:
  - the per-pair tables for the organizer's 25-pair mentor set (the README says these stay local);
  - any organizer rubric wording beyond what the public README and docs already state;
  - any copied text from the vendored `generator/` tree or from the checkpoints. `NOTICE` says the vendored generator and the checkpoints have undetermined upstream terms, so the paper cannot redistribute them. Describe them; do not reproduce them.
- Quote at most one short phrase (under 15 words) from any source. Paraphrase the rest and cite it.
- Mark every claim as one of: **measured in repo** (cite file and line), **cited** (cite a paper, verified by you), or **unverified**. Do not state numbers from memory.

## 1. What the project is (verify each point against the repo)

Task: navigation-error recovery in semiconductor metrology. Locate a reference patch inside a search frame of a repeating layout and report position, rotation, scale, and whether it is present at all.

- **Phase 1 (SEM vs SEM, fixed 10x zoom, always present):** Siamese, SiamRPN++-style cross-correlation network with a CenterNet-style focal loss and a sub-cell offset head. Decoding is two-stage: the network chooses which repeat is correct, then full-resolution ZNCC chooses the exact position. `phase1/RESULTS.md` and `README.md` report 98.0% acc@5px on 700 self-generated held-out scenes, against a ZNCC baseline of 57.3%. `TRAINING.md` §6 reports 0.9800 for the final checkpoint but 0.9757 for the earlier one on the same split. On 1000 fresh scenes the paired gain over that earlier checkpoint is +1.5 points, 95% CI [+0.001, +0.029] (`TRAINING.md` §5–6). Check which numbers are current.
- **Phase 2 (unknown zoom in [8,12], rotation ±5°, about 20% absent):** adds pose hypotheses (top-3 coarse scale peaks), canonicalisation, native-resolution verification, and pose polish. Shipped 1.02M-parameter "wide" checkpoint, about 76.97/85 on an internal set. An extension on a harder generator (raster shear, white row jitter, pixel-centre labels) reaches 83.18 on a holdout (`FAILURE_ANALYSIS.md` §6–9).
- **Phase 3 (reference is a GDSII design, up to 8 layers):** a classical CAD-anchored path with no neural network on the primary route. It finds the reference's integer-nanometre offset in the search design, fits rotation from a polar spectrum and tile displacements, and fits per-layer grey levels by least squares. Reports 85/85 on 600 self-generated pairs when the whole-canvas search GDS is present, and 45.32/85 when it is absent (`FAILURE_ANALYSIS.md` §10).

Verify the exact numbers yourself. Several documents disagree with each other, and that disagreement is itself a finding (see §3).

## 2. Candidate novelty angles (hypotheses to test, not conclusions)

The repository has already run a lot of measurements that are unusual in this area. For each candidate below, find out whether it is genuinely new, what the closest prior work is, and what minimum evidence would make it publishable.

- **A. Alias-aware evaluation for periodic registration.** In Phase 3 generated data, the ground truth is the global ZNCC maximum in only 8 of 55 pairs. Lattice aliases beat the true site by a few thousandths (`docs/PHASE3_COMPLETENESS.md`). Ask whether a proper alias-aware metric, or a failure taxonomy (wrong repeat, wrong mat, sub-pixel), is already standard in template matching for periodic structure, and whether this work adds to it.
- **B. Scan-drift bias in rotation estimation.** A rotation error d displaces template points by (d·v, −d·u), so horizontal raster drift masquerades as rotation. A 1–3 px shear equals about 0.06–0.17°. The fix is to estimate rotation only from vertical offsets of vertical strips (`FAILURE_ANALYSIS.md` §8; `driftsense/matching.py:strip_rotation`). Dev rotation credit 0.895 → 0.958, holdout 0.879 → 0.954. Check against SEM drift and scan-distortion correction literature.
- **C. Benchmark validity and label convention in synthetic SEM data.** Three findings:
  - Upstream ground-truth convention moves labels by 3.1 px on average and up to 22 px. Correction reduces ZNCC residual from 2.5–4.8 px to 0.4–0.9 px (`TRAINING.md` §2).
  - A half-pixel edge-vs-centre convention flips the score by several points, in different directions on different generators (mentor set 76.71 → 81.73; the original generator 82.75 → 79.93) (`FAILURE_ANALYSIS.md` §6).
  - A pinned severity (`lo == hi`) silently generated severity 0 on every pair. The earlier 81.45 / 81.93 headlines (PR #51) were withdrawn for this reason. With the bug in place the score is 81.57 against 79.54 with the severity ladder working (`README.md`, "Retired" section; `docs/PHASE3_MEASUREMENT.md`).
  Together these may make a benchmark-hygiene paper. Check whether it is already covered.
- **D. CAD-anchored registration.** Design-to-design exact offset when the search design is available, versus image fallback. Measured swing 39.7 points and 7× runtime (`FAILURE_ANALYSIS.md` §10). Also the per-layer brightness fit with R² reported as a calibration signal. Check prior art on CAD-to-SEM registration and layout-to-SEM alignment.
- **E. Negative results that are worth publishing.** Edge matching loses 5.89 points to intensity ZNCC (72.72 vs 66.83, 5 splits), and still loses with edge-brightening physics added to the forward model. Rank transform loses 16.7. CAD-to-SEM conversion ties (−0.31). "Better proposals beat better arbitration" recurs across phases. A fused 6-feature calibration overfit on the pool and lost 0.43 on the holdout. Check whether these are already known.
- **F. Confidence margin for pose-basin quality.** Best peak minus best competing peak separates good from bad basins at AUC 0.948 (`docs/PHASE3_MEASUREMENT.md`). Median-filtered ZNCC separates present from absent under impulse and speckle noise where the raw statistic cannot at any threshold (`FAILURE_ANALYSIS.md` §7). Check prior art on peak-quality and confidence measures in correlation tracking (e.g. PSR, APCE, peak-to-sidelobe and peak ratio).

Include any other angle you find that the repository supports. The repo's own author lean, for you to test: the learned-localiser accuracy claim is the weakest contribution against prior art. The evaluation-methodology and mechanism findings (A, B, C) are the most specific to this data. Say whether you agree.

## 3. Things that will attack the paper, and must be checked

- **Only synthetic data.** No real SEM frame, and no organizer data, is in the repo. The organizer's blind split is unseen, and the Phase 3 "85/85" depends on the `search_gds_path` column, which is withheld on the blind split. Check that every claim is scoped to the synthetic generator.
- **Ambiguous generator.** The vendored generator is third-party with undetermined terms. Find out what a paper can release (code, data, checkpoints) and under what licence.
- **Baselines are weak.** The main baseline is ZNCC. Propose and assess stronger ones: phase correlation, multi-scale `matchTemplate`, LoFTR or ELoFTR (issue #69 is about this), RoMa, and an off-the-shelf SiamRPN++ on the same data.
- **Statistics.** The 700-scene test CI spans zero ([-0.29, +1.14]). The 1000-scene Phase 1 CI only just excludes zero (lower bound +0.001). The #89 drift-row gain on the 500-pair holdout has a CI that includes zero and is not shipped (`FAILURE_ANALYSIS.md` §9). The #88 rotation gain's CI excludes zero (§8). Say what sample size each claim needs.
- **Document drift.** `TRAINING.md` (0.46M model in some places, 1.02M shipped in others), `README.md` (threshold 0.18 vs 0.1587), and `failure_analysis.pdf` (built from a CSV, not auto-synced with `FAILURE_ANALYSIS.md`) disagree in places. List every inconsistency you find.
- **Unshipped work.** The #89 drift-row shrinkage is merged and enabled on `main` (PR #98). Its 83.31 holdout figure is historical, not a fresh rescore of current `main`. The #101 shear work and the #109 label-noise floor are open. PRs #102 and #107 are open and unmerged; their numbers are unverified. Do not count unshipped work as a result unless you say so.
- **Phase 3 gaps.** Rotation and scale are pinned in the generated Phase 3 set, so the pose path is untested there. Barrel distortion is unhandled in labels. Label convention for Phase 3 is correct only by accident (`FAILURE_ANALYSIS.md` §11–15).

## 4. Work to do, in order

1. **Ingest.** Clone, read the required files in §0. Produce a claims ledger: every quantitative claim, its file and line, its evidence type, and whether the code and the docs agree.
2. **Literature.** Search (arXiv, Google Scholar, IEEE Xplore, SPIE Digital Library, ACM DL, OpenReview). Find at least 25 closely related works and verify each one: title, venue, year, DOI or arXiv ID. Cover at least these areas: template matching for periodic and lattice structures; SEM image registration and drift correction; CAD-to-SEM and layout-to-image alignment and overlay metrology; SiamRPN, SiamFC, CenterNet, and cross-correlation tracking; sub-pixel registration (ZNCC, phase correlation, LoFTR and successors); peak-quality and confidence measures for correlation tracking; benchmark-validity and label-noise reporting in synthetic-data ML. Mark any citation you could not verify as **unverified**.
3. **Novelty verdict per candidate.** For each of A–F plus any you add: the closest prior work, what this repo adds beyond it, and a verdict of *novel*, *incremental*, or *already known*. Justify each verdict with citations.
4. **Minimum experiments.** For each candidate that survives, state the smallest experiment that would make the claim publishable. Include: sample size and the paired-bootstrap design, baselines, ablations, and a stopping rule. Say which experiments the repo can already run locally and which need Kaggle-scale compute. The owner runs heavy training on Kaggle, not on the laptop.
5. **Paper framing.** Propose two or three coherent framings, each with a one-paragraph claim, its supporting candidates, and the venue class it fits (e.g. metrology and lithography journals such as JM3, IEEE Transactions on Semiconductor Manufacturing, SPIE proceedings; computer-vision workshops; methodology and negative-results venues). Rank them by how strongly the existing evidence already supports them.
6. **Go / no-go.** Recommend one of: go with a named framing, go conditionally on named experiments, or no-go. Give the single biggest risk.

## 5. Output format

Produce one document with these sections, in this order:

1. **Bottom line:** three sentences maximum, including go / no-go.
2. **Claims ledger:** table of claim, source (file:line), evidence type, status (agrees / disagrees / unverified).
3. **Prior-art table:** title, year, venue, DOI or arXiv ID, relevance to candidates A–F, verified yes/no.
4. **Novelty verdicts:** one row per candidate with verdict and justification.
5. **Recommended framing(s)** and the evidence each one already has.
6. **Required experiments**, ordered by value per cost.
7. **Risks and inconsistencies** found in §3, plus any others.
8. **Open questions for the owner**, limited to decisions only the owner can make (data release, licence, venue, which candidate to pursue).

Do not draft the paper. If you are tempted to, stop and put the outline in section 5 instead.

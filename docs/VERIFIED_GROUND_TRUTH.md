# Verified ground truth — Phase 2 (PS-02)

**Purpose:** every convention, formula, tolerance and calibration fact the
published Phase 2 task definition pins down, in one place, so any future
session or agent starts from the same ground truth instead of re-deriving (or
mis-reading) it. Everything below is **verified** — each fact was checked
against the published task materials or measured directly, with the method
recorded. The reference sample set used for verification stays local-only and
is **never committed** — only this description is tracked.

Compiled 2026-09-02 from the published task materials (prompt document, task
deck, and the reference sample set shipped with them) plus the reference
baseline implementation they include.

---

## 1. Task contract (what register.py must honour)

* Entry point: `python register.py --input pairs.csv --output predictions.csv`.
* `pairs.csv` columns: `pair_id, search_path, reference_path` (that spelling).
* Output: one row per input pair **in input order**, columns
  `pair_id, x, y, theta, scale, found, score`. A pair that raises must still
  emit a row (`found=0`, pose columns zeroed) — a missing row scores zero.
* No network access at inference; CPU-only; weights ship in the ZIP.
* Ranges disclosed to solvers and hard-codable: `z ∈ [8, 12]`,
  `θ ∈ [-5°, +5°]`, ~20% of pairs absent.

## 2. Scoring rubric (task deck, authoritative)

| Component | Weight | Rule |
|---|---:|---|
| Localisation | 40 | present A/B pairs; tiers ≤1 px 1.00, ≤2 0.80, ≤3 0.60, ≤5 0.40; total = (0.45·A + 0.55·B)·40; **a declined pair forfeits it** |
| Pose — scale | 10 | rel err ≤1% 1.00, ≤2% 0.60, ≤5% 0.30 |
| Pose — rotation | 10 | abs err ≤0.25° 1.00, ≤0.5° 0.60, ≤1.0° 0.30 |
| Rejection | 15 | F1 on `found` over the 180 grayscale pairs (positive class ambiguous; report both) |
| Calibration | 10 | AUC of `score` vs per-pair correctness |
| Bonus | +6 / +4 | Set D credit ≥ 0.40 **with** Sets A–C ≥ 0.50; rejection F1 ≥ 0.90 |
| Efficiency | 5 | relative ranking; ≤5 s median per pair budget |

Blind-set composition is fixed (task deck): **A 70 / B 70 / C 40 / D 20**.

## 3. Theta sign convention — pinned by fiat (prompt §2.2)

Image coordinates `x` right, `y` down; a canvas point maps to search as

```
p_search = (1/z) · R(theta) · (p_canvas − c_canvas) + c_search
R(theta) = [[ cos t,  sin t],
            [−sin t,  cos t]],   t = radians(theta)
```

so **positive theta turns the pattern counter-clockwise as displayed**. Both
signs are self-consistent while generating, which is why it is fixed by fiat:
*"a solver calibrated against an opposite-sense set inverts theta on ours."*

**Verified (2026-09-02):**
* `driftsense.generate.search_affine` builds `cv2.getRotationMatrix2D((cc, cc),
  theta, 1/z)`, whose linear part is numerically identical to `(1/z)·R(theta)`
  above (checked at θ=2.5°, z=10 — exact to 1e-6).
* End-to-end on the reference sample pairs: `pred_theta = +gt_theta` — median
  |raw err| 0.052 vs |flipped| 5.017 (~97× worse); the pair with gt θ = +4.6
  decodes at 4.6000. **No inversion bug** at any level of the stack.

## 4. Ground-truth semantics (task materials §2)

* Geometry is ONE affine `canvas → search`: rotate by +θ about the canvas
  centre, scale by 1/z, translate to centre a 1000×1000 output. GT is the
  reference crop's centre through that affine.
* Raster drift and barrel distortion are applied **after** the pose affine, so
  the true feature is displaced away from the affine coordinate — worth 1–2 px
  at high severity. The per-row drift vector is captured and the GT point is
  pushed through **both maps** (Newton inversion for the barrel cubic):
  **labels are exact**, not nominal.
* Absent rows (`present=0`) carry zeros in the pose columns.
* Set D is 3-channel; Sets A/B/C are single-channel.

## 5. The reference naive baseline (prompt §5.1 + published source)

Definition (published `baseline_zncc.py`): brute-force `cv2.TM_CCOEFF_NORMED`
over the coarse grid **z = 8.0…12.0 step 0.5 × θ = −5…+5 step 1.0°**, template
= box-blur + warpAffine of the reference, presence by thresholding the peak at
**0.55**. This is the calibration instrument for generator authors — not a
team's baseline.

Its published results on the reference sample set:

| statistic | value |
|---|---|
| Set A mean credit | **1.000** |
| Set B mean credit | **0.467** |
| Set D mean credit | 1.000 |
| **Overall present** | **0.800** |
| present peaks | 0.338–0.956 |
| absent peaks | 0.279–0.393 |
| **separation gap** | **−0.055** (negative: NOT threshold-separable, by design) |
| rejection @0.55 | precision 1.00, recall 0.81, **F1 0.897** |
| pose on coarse grid | scale ≤3.0% worst / 1.0% median; θ ≤1.10° worst / 0.35° median |

## 6. The 0.30–0.55 band — what it is and is not

Prompt §5.1: *"Target: overall mean credit on present pairs between 0.30 and
0.55. Above that your set is too easy to separate a field of teams; near zero
and it is noise."*

* It is a **generator-authoring target** for whoever builds a dataset — not a
  scoring gate and not a solver requirement.
* The published reference sample scores 0.800 against it; the accompanying
  README says so plainly ("Set A is too easy… 0.80 is well above that") and
  recommends shifting the real 200-pair set's B-severity toward 3–4.
* Therefore: "our set's naive baseline is ~0.9" is a **difficulty-calibration
  caveat** (numbers measured on it read optimistic), not a correctness bug and
  not a rubric violation. Report it as a caveat with our eval numbers.
* Severity assignment rule from §2.4: grid-aligned poses (z on the 0.5 grid, θ
  whole degrees incl. 0.00) are easier at any severity — anti-correlate
  grid-alignment with severity, and record which pairs are grid-aligned.

## 7. GT verifiability gate (task materials §6)

Every present pair is checked at generation time by rigid template match at
its own labelled pose: the global correlation peak must land within 3 px of
the label, crops resampled up to 14 attempts. Our generator implements the
equivalent gate — keep it on.

## 8. Verification status (2026-09-02) — all checks complete

* [x] Conventions extracted and pinned (§1–§7).
* [x] Our generator affine == published convention, numerically (§3).
* [x] **`register.py` end-to-end on the 20 reference sample pairs — ALL PASS
      (deterministic, byte-identical rerun):**
  * Format: exact 7-column contract, 20 unique ids, input order, found=0 rows
    zero-filled.
  * Localisation credit: Set A **1.000**, Set B **0.9667** (naive reference
    baseline: 0.467), Set D 0.900, overall present **0.975 vs 0.800**.
    14/16 present pairs ≤1 px.
  * **Theta sign: `pred_theta = +gt_theta`** (see §3).
  * **Scale semantics: `scale` = z** (direct zoom): median rel err 0.16%,
    max 0.56%; a 1/z bug would read 87–98%.
  * Absent: 4/4 declined; absent scores ≤0.061 vs present ≥0.565 (9× gap).
  * Runtime: median 3.12 s, p90 3.44 s, max 6.19 s (4 threads, local).
* [x] **Reference-baseline calibration on our `data/ext_p2` (96-pair
      deterministic sample):**
  * Harness check first: the published scorer reproduces its published
    calibration exactly (peaks, gap −0.055, F1 0.897) once its
    `pred_present = peak ≥ 0.55` credit gate is applied — omitting that gate
    mis-reads Set B (0.933 vs published 0.467).
  * **Our gated naive-baseline credit: Set A 0.525, Set B 0.190, overall
    present 0.357 — inside the published 0.30–0.55 target band** (reference
    sample: 0.800, above its own band). The earlier "~0.92" claim conflated
    our *solver's* Set A accuracy with the *baseline's* credit: refuted.
  * Separation gap **−0.576** vs reference −0.055; rejection@0.55 F1 0.794 vs
    0.897 — our absent pairs are *harder* to threshold off than the reference
    ones. Numbers measured on this set are therefore **conservative, not
    optimistic**.
  * **"Search not proper" refuted at the strongest level:** on the very pairs
    where the naive baseline fails with high peaks (0.65–0.89) at 40–850 px
    from GT, local ZNCC *at the labelled pose* reads 0.66–0.87 with the local
    peak ≤0.8 px from the label — the reference is exactly where the label
    says and the baseline locked onto a decoy, which is the intended
    difficulty (prompt §5.1: reference ambiguity is the lever).

Efficiency note: the one open latency lever — E3 coarse-grid pruning — was
audited and **not shipped** (equality bit-exact on 200/200 pairs, but the
single-process clock reads 1.00x; the skipped work is noise against the
network forward + refine + polish). Median runtime on the reference sample is
3.12 s (max 6.19 s) against the ≤5 s median budget on local hardware; a slow
4-core grading machine is the remaining latency risk.

Practical note for reruns: the sample drop ships the search images **flat**
(`Dataset_AMP_Phase 2/pNNN.png` — these ARE the search frames); the documented
`search/` subfolder does not exist until reconstructed. The published
`baseline_zncc.search_pose` reproduces the published calibration to the digit
on those files, which is the identity proof.

## 9. Where the verification material lives — and the never-commit rule

* Local copy: `.agents/ref_material/` (task documents, 20 sample pairs, GT,
  reference baseline/generator source). Git-ignored.
* Task rules: the sample set is shared privately for validation and must
  **never be committed or shipped**. Only this conventions file is tracked.
  No threshold or hyperparameter may be tuned on it (single-pass validation
  only, shipped config).

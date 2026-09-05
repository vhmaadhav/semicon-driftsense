# G2 — the full 2,250-pair severity-weighted re-run

**Date:** 2026-09-05 · **Issue:** #3 · **Branch:** `fix/stage0-measurement-trust`
· **Head at run:** `c1d871a` (+ the scorer fix recorded in §1)
· **Weights:** `weights/driftsense.pt` (sha256 `e6506b7c3b2ccfd4…`, see
`G2_FULL_RERUN.log`) · **Config:** `driftsense.config` — threshold 0.18,
`legacy_min` confidence, band OFF, verification `zncc`, subpixel-rows ON
· **Data:** `data/ext_p2`, generator bundle
`1f336770031301e6b776b93ab91c2bd0f85aebe7378e45bb505bbf5b68550658`
· **Machine:** local x86, 12 workers × 2 threads (wall-clock only — not an
efficiency measurement) · **Per-pair predictions:**
`.agents/G2_ext_full_shipped.csv` (2,500 rows, retained)

This is the run `G2_SEVERITY_CAMPAIGN.md` §4.1 asked for and issue #3 tracks.
The 80-pair smoke test recorded there is superseded by everything below.

---

## 0. Headline

Full 2,500-pair scorecard at the shipped operating point:

| component | credit | points |
| --- | ---: | ---: |
| Localisation (40) — A 0.9755 / B 0.8402 | 0.9011 | 36.04 |
| Pose — scale (10) | 0.8967 | 8.97 |
| Pose — rotation (10) | 0.9008 | 9.01 |
| Rejection (15) — F1(reject) @0.18 | 0.9126 | 13.69 |
| Calibration (10) — AUC present-only | 0.9870 | 9.87 |
| **SUBTOTAL (85 measurable)** | | **77.58** |
| Bonus Set D (optical) | 0.9792 | +6 condition met |

Rejection confusion on the 2,250 grayscale pairs: correct-rej 454 / lost-real
41 / missed-abs 46.

**The answer to the question the issue asks:** weighting Set B toward severity
3–4 costs **1.5 to 3.1 points** of the 85, and drops the +4 bonus gate from
**69% to 47–59%**. It does not cost 9 points. The 9-point figure from the
80-pair smoke test is reproducible, but only for a comparison the blind set
will not present (§4).

---

## 1. Precondition — the two scorers were not the same scorer

Reviewed on issue #3 before the run: `scripts/eval_phase2.py` at `c1d871a`
fixed confidence, F1 polarity and threshold, but still

* computed localisation from **raw** predictions, with no mask for declined
  present pairs — `register.py` zero-fills the pose/location columns of a
  declined answer, so those pairs earn nothing on the graded CSV; and
* **pooled A and B** instead of applying the published 0.45 / 0.55 weighting.

Both inflate, and the second is why the smoke test's component deltas could not
be multiplied by rubric weights and compared against an `eval_ext` subtotal —
which is exactly what its "≈ −9 points of 85" did.

Fixed by extracting `eval_ext.py`'s scorer verbatim into **`driftsense/rubric.py`**
and pointing both evaluators at it. The A/B path is unchanged (pinned by
`tests/test_eval_found_masking.py`, `tests/test_score_semantics.py`); an
unlabelled frame — every split our own generator writes, since it emits no
`phase2_set` column — is scored as one pooled present-pair stratum with the
header saying so, rather than silently 0.45-weighted. `tests/test_rubric_shared.py`
(8 tests) pins it.

`scripts/grade_emulation.py` keeps its own numpy rubric because it runs 20,000
times per sweep point; `test_severity_mixture.py` now pins it against the shared
scorer component by component, so the third copy cannot drift either.

---

## 2. Severity is realised in B and C — and is a decorative label in A and D

Checked before using it, because a severity that was asked for and not rendered
is this project's most expensive recurring defect:

| set | `severity_continuous` | drift_jitter L1 → L4 | speckle L1 → L4 | verdict |
| --- | --- | ---: | ---: | --- |
| A | **not recorded** | 0.491 → 0.457 | 0.042 → 0.039 | **label only** |
| B | 0.165 / 0.410 / 0.638 / 0.887 | 0.555 → 1.891 | 0.070 → 0.286 | realised |
| C | 0.154 / 0.399 / 0.649 / 0.882 | 0.504 → 1.252 | 0.053 → 0.176 | realised |
| D | **not recorded** | 0.520 → 0.489 | 0.037 → 0.042 | **label only** |

Sets A and D carry `severity_level` 1–4 in the manifest — 219/219/219/218 and
63/63/62/62, a clean-looking ladder — while every physical parameter the
severity ladder drives is **flat across the four levels**, and
`severity_continuous` is null on all 1,125 of those rows. Their measured
localisation credit is flat to match (A: 0.9808 / 0.9699 / 0.9781 / 0.9734).

This is not a defect in `ext_p2` — Set A is meant to be the clean set and the
severity ladder is specified for Set B — but the label is there and it grades
as if it meant something. **Anyone stratifying "severity 4" across all sets is
averaging 875 A/D pairs of nominal difficulty into the number.** The audit is
now part of `scripts/severity_breakdown.py --manifest` and prints a warning
naming the label-only sets.

---

## 3. What severity actually does, per level (Set B, submission-masked)

| level | n | loc credit | ≤1 px | ≤5 px | median | scale | rotation | med rot | declined |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 219 | 0.9735 | 92.2% | 100.0% | 0.35 px | 0.8479 | 0.9364 | 0.097° | 2 |
| 2 | 219 | 0.9205 | 77.2% | 98.2% | 0.49 px | 0.8047 | 0.8743 | 0.112° | 3 |
| 3 | 219 | 0.7945 | 56.6% | 90.9% | 0.84 px | 0.8152 | 0.8015 | 0.182° | 6 |
| 4 | 218 | 0.6716 | 43.6% | 82.6% | 1.23 px | 0.7844 | 0.7052 | **0.267°** | 20 |
| **Δ 1→4** | | **−0.302** | −48.6 pp | −17.4 pp | +0.88 px | −0.064 | **−0.231** | crosses 0.25° | ×10 |

Rejection errors, by level:

| level | B lost-real /219 | C missed-abs /125 | C correct-rej /125 |
| ---: | ---: | ---: | ---: |
| 1 | 2 | 6 | 119 |
| 2 | 3 | 3 | 122 |
| 3 | 6 | 14 | 111 |
| 4 | 20 | 23 | 102 |

Three things the smoke test called correctly, confirmed at full n:

1. **Rotation's median error crosses the 0.25° full-credit tier** under
   severity — 0.097° → 0.267°. Its credit is the tier, so that single crossing
   is most of the −0.231.
2. **≤5 px falls, 100% → 82.6%.** At severity 3–4 roughly one present pair in
   six is a gross basin failure, not a sub-pixel miss — consistent with
   `FAILURE_ANALYSIS.md` §2 and `SUBPIXEL_DRIFT.md` §10.
3. **Declining real pairs grows sharply** — 2 → 20 of 219. Each forfeits
   localisation *and* pose on top of hurting F1.

One thing it called backwards, see §4.

---

## 4. The 9 points — what that number was measuring

The smoke test reported "≈ −9 points of 85 from severity alone". At full n that
magnitude **reproduces exactly**, for the endpoint contrast:

| Set B drawn entirely at | mean F1 | P(F1 ≥ 0.90) | E[total] |
| --- | ---: | ---: | ---: |
| severity 1 | 0.9340 | 0.883 | **81.53** |
| severity 4 | 0.8698 | 0.192 | **72.36** |
| | | | **−9.17** |

So the smoke test's arithmetic was sound for what it compared: its two arms
(realised severity 0.511 vs 0.878) sit near the ends of the ladder.

**But that is not the comparison G2 asks about.** G2 says the blind set will
*weight* B toward severity 3–4, against our pool which is uniform over the
four. Sweeping that weighting — the only thing that changes, A and C held
fixed, 20,000 stratified A70/B70/C40 draws each:

| B at severity 3–4 | mix (L1,L2,L3,L4) | mean F1 | sd | 95% CI | P(F1 ≥ 0.90) | E[total] |
| ---: | :--- | ---: | ---: | :--- | ---: | ---: |
| 50.0% *(pool as-is)* | 17,18,17,18 | 0.9122 | 0.0311 | [0.846, 0.964] | **0.686** | **77.53** |
| 65.7% | 12,12,23,23 | 0.9066 | 0.0317 | [0.840, 0.963] | 0.622 | 76.58 |
| 74.3% | 9,9,26,26 | 0.9033 | 0.0319 | [0.835, 0.963] | 0.585 | 76.04 |
| 85.7% | 5,5,30,30 | 0.8991 | 0.0323 | [0.831, 0.962] | 0.535 | 75.32 |
| 100.0% | 0,0,35,35 | 0.8939 | 0.0326 | [0.827, 0.951] | **0.470** | **74.42** |

Monte-Carlo se on every gate rate is ±0.003–0.004.

**The grade-level exposure is −1.5 points at a three-quarters weighting and
−3.1 at the extreme, not −9.** G2's warning is confirmed in direction and sign;
the magnitude quoted in issue #3 overstates it by roughly 3–6×, because it
contrasts the ends of the severity ladder rather than a shift in its mixture.

### The component ordering reverses back

The smoke test concluded rotation is "the most severity-fragile component,
losing more than twice localisation's proportional credit", and that this
reverses the pose priority. At 2,250 pairs it does not hold:

* localisation loses **−0.302** of 1.0 on Set B (−31% relative);
* rotation loses **−0.231** (−24.7% relative).

And in rubric points the gap is far wider than the raw credits suggest, because
localisation carries 40 points at B's 0.55 weight while rotation carries 10
pooled across A *and* B present pairs:

* localisation: −0.302 × 0.55 × 40 = **−6.6 points**
* rotation: −0.231 × ~0.5 (pooling with an undegraded A) × 10 = **−1.2 points**

**Localisation is ~5× the exposure of rotation under severity.** The smoke
test's ordering was an artifact of measuring localisation *unmasked* — its
localisation arm did not charge for the declined present pairs that severity
produces, and those are exactly what grows (2 → 20). Point 3 of
`G2_SEVERITY_CAMPAIGN.md` §4 ("re-rank the pose work: rotation before scale")
survives — rotation still loses 3.6× what scale does — but the claim that
rotation outranks *localisation* does not.

---

## 5. Set C matters more to the bonus gate than Set B does

The rejection F1's true positives are correct rejections, so it is Set C, not
Set B, that decides most of it. Repeating the sweep with Set C also drawn at
severity 3–4:

| B at severity 3–4 | mean F1 | P(F1 ≥ 0.90) | E[total] |
| ---: | ---: | ---: | ---: |
| 50.0% *(pool as-is)* | 0.8801 | **0.305** | 77.03 |
| 74.3% | 0.8713 | 0.228 | 75.55 |
| 100.0% | 0.8620 | **0.158** | 73.92 |

Hardening Set C alone — B left uniform — takes the gate from **0.686 to 0.305**.
Hardening Set B all the way, with C uniform, only takes it to 0.470. The
mechanism is in §3's table: C's missed-absent count runs 6 / 3 / 14 / 23 across
the levels, so a severity-weighted C nearly quadruples the false-accept rate,
and every one of those is a lost true positive.

**Read this arm as a sensitivity, not a prediction.** Slide 4 attributes the
four undisclosed severity levels to Set B; that Set C also carries realised
severity is our generator's choice, and nothing in the official materials says
the organizers' decoys vary the same way. It is recorded because it changes
where the effort should go: **G3 (the decoy-signature audit the DOCX demands,
still unrecorded) is worth more to the bonus than any further Set B pose work.**

---

## 6. So is the +4 bonus banked?

No, and the harder mixture makes it worse rather than settling it:

| scenario | P(F1 ≥ 0.90) |
| --- | ---: |
| `REJECTOR_NONLINEAR.md` prior estimate (t=0.202, older decode) | 0.53 – 0.65 |
| this run, pool as-is | **0.686** |
| B weighted 74% to severity 3–4 | 0.585 |
| B weighted 100% to severity 3–4 | 0.470 |
| B as-is, C weighted to severity 3–4 | 0.305 |
| both weighted to severity 3–4 | 0.158 |

The shipped decode's as-is gate rate is **better** than the 53–65% previously
carried (0.686), because the current configuration measures F1 0.9126 where the
older CSV measured 0.9078. Under the severity weighting G2 predicts it lands
back at roughly a coin flip. The planning line in `REJECTOR_NONLINEAR.md` §
"the +4 bonus is not banked" stands, and this run gives it a measured range
rather than a single point: **0.47 – 0.69 on B-severity alone, and as low as
0.16 if the organizers' decoys harden with severity too.**

---

## 7. Caveats — read before quoting

* **This is one generator's severity ladder.** Every number is `ext_p2`
  (bundle `1f336770…`), the same generator our training shards come from. It
  measures sensitivity to *that* ladder. `PHASE2_STATE.md` §"`data/ext_p2` is
  not organizer data" applies unchanged, as does the off-generator finding that
  the presence model does not transfer to a foreign renderer.
* **The mixture arms resample ext_p2's own B pairs.** That is deliberate: it
  holds generator, seed, decoys, Set A and Set C byte-identical so the only
  thing varying is B's severity distribution — which regenerating a fresh set
  could not have guaranteed. It also means the arms share pairs and are *not*
  independent samples; the 95% CIs describe stage-to-stage sampling within this
  pool, not generator-to-generator variation.
* **The blind set is 200 pairs drawn from the organizers' data, not ours.**
  Every P(gate) here is a model-risk estimate in the sense
  `STAGED_BOOTSTRAP_T018.md` sets out at length, not a grading-event
  probability.
* **`--severity-sweep` reports the realised fraction**, which is why the first
  row reads 50.0% and the others 65.7 / 74.3 / 85.7: an integer quota of 70
  cannot hit every requested proportion. The requested figure is not quoted
  anywhere.
* **No inference behaviour changed in this work.** The scorer fix, the mixture
  option and the breakdown script are measurement only; the decode is
  `c1d871a`'s.

---

## 8. Follow-ups

1. **G3 is now the highest-value open compliance item**, not further Set B pose
   work: Set C's severity response moves the bonus gate more than Set B's does
   (§5), and the decoy-signature audit is what would tell us whether our C
   decoys resemble the organizers' at all.
2. **Localisation, not rotation, is where severity costs points** (§4). The
   gross-basin failures that grow 0% → 17.4% at ≤5 px are the target;
   `FAILURE_ANALYSIS.md` §2's wrong-basin analysis is the relevant lead.
3. **Re-check `refit_xy` and the rescue pass on severity-weighted draws.**
   `G2_SEVERITY_CAMPAIGN.md` §4.4 asked for this and it is still open — both
   were measured out on easier distributions and both target gross-basin
   failures.
4. **G5 remains unresolved** and this run does not settle it: the present-only
   AUC reads 0.9870 while the submitted-output variant reads 0.7637 on the same
   predictions. That 0.22 spread is 2.2 rubric points hanging on a definition
   the materials do not fix. It is worth the T+3 question.

## 9. Reproducing

```bash
# the run (~21 min, 12 workers)
python scripts/eval_ext.py data/ext_p2/test_{A_0000,A_0001,B_0000,B_0001,C_0000,D_0000} \
    --jobs 12 --threads 2 --out .agents/G2_ext_full_shipped.csv

# realised-severity audit + per-level components
python scripts/severity_breakdown.py --csv .agents/G2_ext_full_shipped.csv \
    --manifest 'data/ext_p2/*_manifest.csv'

# the mixture sweep
python scripts/grade_emulation.py --csv .agents/G2_ext_full_shipped.csv \
    --threshold 0.18 --draws 20000 --seed 7 --severity-sweep
python scripts/grade_emulation.py --csv .agents/G2_ext_full_shipped.csv \
    --threshold 0.18 --draws 20000 --seed 7 --severity-sweep --c-mix 0,0,50,50
```

Raw sweep output: `.agents/G2_SWEEP_RESULTS.txt`. Run log and provenance:
`.agents/G2_FULL_RERUN.log`.

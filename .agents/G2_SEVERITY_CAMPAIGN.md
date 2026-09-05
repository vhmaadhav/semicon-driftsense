# G2 — what a severity-3/4-heavy Set B actually costs

> **SUPERSEDED 2026-09-05 by `G2_FULL_RERUN.md`** (issue #3). The full
> 2,250-pair run is done; read that document for any number. This note is kept
> for its §0 (the pin trap, still the live finding) and as the record of what
> an 80-pair smoke test could and could not see. Two of its conclusions did not
> survive full n — both flagged inline below:
>
> * the **≈ −9 points** headline reproduces only for the ladder's *endpoints*
>   (all-severity-1 vs all-severity-4: 81.53 → 72.36). The mixture shift G2
>   actually predicts costs **−1.5 to −3.1** points;
> * **rotation is not the most severity-fragile component.** Localisation loses
>   more, both proportionally and (5×) in rubric points. The ordering here was
>   an artifact of scoring localisation without the submission mask.

**Date:** 2026-09-04 · **Machine:** local x86, 4-thread cap, idle, pinned
`requirements.txt` on Python 3.11 · **Scorer:** `scripts/eval_phase2.py` *after*
the 2026-09-04 correction (reject-positive F1 at the shipped threshold, scoring
`confidence` rather than the raw network score).

Compliance item G2 (`PHASE2_COMPLIANCE_ISSUES.md`) says the jury expects the real
200-pair set to shift Set B toward severity 3–4, that *"our ext_p2 B distribution
is our own choice and skews easy"*, and that therefore *"our severity-robustness
margin is thinner than our self-scores suggest."* The prescribed fix was to
**weight severity 3–4 in the next generator run and re-test rejection F1 before
quoting P(bonus)**. This is that run.

---

## 0. A blocker found on the way in — the pin trap is still live

The obvious way to ask for "severity 4 only" is `--severity-range 1.0 1.0`. That
silently produces the **opposite**:

| command | realised `severity_continuous` | drift_jitter_px | speckle_sigma |
| --- | --- | --- | --- |
| `--severity-range 1.0 1.0` | **0.0** | 0.52 – 1.53 | 0.0 |
| `--severity-range 1.0 1.000001` | **1.0000005** | 2.25 – 2.35 | 0.28 – 0.36 |

`PoseSpec.severity` disables the ladder when `hi <= lo` and makes *no draw at
all* — deliberate, and what keeps the Phase 1 splits byte-reproducible
(`driftsense/generate.py:542-544`). The consequence is that a pinned level
renders **nominal, easiest-possible** data while the manifest still reports the
level that was asked for.

This is the defect that retracted the 81.45 / 81.93 headline figures (README,
*"Retired: the 81.45 / 81.93 figures"*). It was fixed in the Issue 45 audit
fixture with a 1e-6 epsilon; **the core generator kept the strict comparison**,
so `generate_dataset.py --severity-range` still walked into it — directly in the
path of the G2 work.

Closed by a CLI guard in `generate_dataset.py::build_pose_spec` that rejects
equal non-zero endpoints with an actionable message, plus
`tests/test_severity_pin_guard.py` (7 tests). `0 0` remains the documented "off"
idiom, and the Phase 1 path is untouched.

---

## 1. The measurement

Two 80-pair `--phase2` sets, **same seed (2026)**, same composition (65 present /
15 absent), differing only in the severity band:

| set | `--severity-range` | realised severity (mean) | drift_jitter_px (mean) |
| --- | --- | ---: | ---: |
| baseline | `0.0 1.0` | 0.511 | 1.37 |
| B-heavy | `0.75 1.0` | 0.878 | 2.10 |

| component | baseline | **B-heavy** | delta |
| --- | ---: | ---: | ---: |
| Localisation credit | 0.868 | **0.745** | **−0.123** |
| &nbsp;&nbsp;≤1 px | 71% | 62% | −9 pp |
| &nbsp;&nbsp;≤5 px | 94% | **82%** | **−12 pp** |
| &nbsp;&nbsp;median error | 0.58 px | 0.69 px | +0.11 px |
| Scale credit | 0.844 | 0.774 | −0.070 |
| **Rotation credit** | 0.877 | **0.615** | **−0.262** |
| &nbsp;&nbsp;median rot error | 0.15° | **0.34°** | crosses the 0.25° tier |
| Rejection F1 @ 0.18 (reject-pos) | 0.727 | 0.703 | −0.024 |
| Calibration AUC | 0.958 | **0.870** | −0.088 |

Applying the rubric block weights (loc 40 / scale 10 / rot 10 / calib 10),
severity alone costs roughly **−4.9 loc, −0.7 scale, −2.6 rot, −0.9 calib ≈
−9 points of 85.** G2's warning is confirmed, and it is not a rounding error.

> **CORRECTION (2026-09-05, `G2_FULL_RERUN.md` §4).** The −9 is real but it is
> an *endpoint* contrast, and this arithmetic is not a subtotal delta: the
> scorer these components came from pooled A/B and did not mask declined
> present pairs, so the weights above cannot be applied to it. Measured on the
> full 2,250 with the shared scorer, weighting B toward severity 3–4 — the
> shift G2 predicts — costs **−1.5 points at 74% and −3.1 at 100%**, and the
> all-L1-vs-all-L4 endpoints cost −9.17. Quote those.

---

## 2. What the numbers say

**Rotation is the most severity-fragile component, by a wide margin.**
> **RETRACTED (2026-09-05, `G2_FULL_RERUN.md` §4).** At 2,250 pairs
> localisation loses **−0.302** on Set B against rotation's **−0.231**, and in
> rubric points −6.6 against −1.2 — localisation is ~5× the exposure. This
> paragraph's ordering came from an unmasked localisation arm, which did not
> charge for the declined present pairs severity produces (2 → 20 of 219). The
> *scale-vs-rotation* half of the ranking survives: rotation still loses 3.6×
> what scale does.

It loses 0.262 of 1.0 — more than twice localisation's proportional loss — because its
median error moves 0.15° → 0.34° and steps straight over the 0.25° full-credit
tier. The pose search's rotation estimate degrades faster under acquisition
noise than its scale estimate does (scale: 0.54% → 0.80% against a 1% tier, which
it still mostly clears). **If pose work is prioritised, rotation now outranks
scale** — this reverses the ordering suggested by the audit's randomized-set
measurement, which saw rotation as the safer of the two.

**≤5 px falls 94% → 82%.** On the easier band, localisation failures are
sub-pixel-tier misses; at severity 3–4, roughly one present pair in six becomes a
*gross* failure. That matches `FAILURE_ANALYSIS.md` §2 (wrong pose basin) and
`SUBPIXEL_DRIFT.md` §10 — no sub-pixel correction rescues a pair whose pose
basin is wrong.

**The dominant rejection error is declining real pairs, not accepting absent
ones.** correct-rej 12 / lost-real 6 / missed-abs 3 (baseline) and 13 / 9 / 2
(B-heavy). Every lost-real forfeits that pair's localisation *and* pose credit on
top of hurting F1 — the asymmetry `driftsense/config.py` documents as the reason
the threshold is biased low. Under severity the pipeline gets *more*
conservative, exactly where it can least afford to.

---

## 3. Caveats — read these before quoting anything above

* **n = 80 per arm, 15 absent pairs.** The rejection F1 figures in particular are
  built on 15 absent pairs; a single flip moves F1 by ~0.03. `PHASE2_STATE.md` §2
  puts sampling noise at σ ≈ 1.2 points on a *200*-pair grade — this is smaller
  than that. Treat the **direction and rough magnitude** as the result; do not
  quote these as scores.
* **These are `--phase2` randomized sets, not the A70/B70/C40/D20 spec
  composition**, so they are not comparable to the README's campaign numbers.
* **The absolute rejection F1 here (0.727 baseline) is far below the 0.9078 /
  0.9198 recorded on `data/ext_p2`** at the same shipped threshold. Part of that
  is the small absent count, but the gap is large enough that it should not be
  assumed to be noise. It is consistent with G2's core claim — that ext_p2 is
  easier than a realistic set — but **confirming that requires re-running the
  full 2,250 under a severity-weighted regeneration**, which this campaign does
  not do.

---

## 4. Follow-ups this opens

1. ~~**Regenerate `data/ext_p2` (or a successor) with severity 3–4 weighted** and
   re-run the full 2,250.~~ **DONE 2026-09-05 — `G2_FULL_RERUN.md`.** Done by
   severity-stratified *resampling* of ext_p2's own B pairs rather than
   regeneration, which holds the generator, seed, decoys, Set A and Set C
   byte-identical so that only B's severity distribution varies. Everything above is a 160-pair smoke test pointing at
   that; the promotion-grade number has to come from the full set.
2. ~~**Do not quote P(+4 bonus) from ext_p2 numbers** until (1) is done.~~
   **ANSWERED:** 0.686 as-is, 0.470–0.585 under a B-heavy mix, 0.158–0.305 if
   Set C hardens too (`G2_FULL_RERUN.md` §6). The bonus
   gate is already assessed at a 53–65% coin flip (`REJECTOR_NONLINEAR.md`) on
   data this campaign suggests is optimistic.
3. **Re-rank the pose work: rotation before scale.** Rotation loses 0.262 under
   severity against scale's 0.070.
4. Re-check the `refit_xy` and rescue-pass verdicts on severity-weighted data.
   Both were measured out on easier distributions, and both target exactly the
   gross-basin failures that grow from 6% to 18% here. Neither is likely to flip
   — but both were closed on data this campaign shows is not representative.

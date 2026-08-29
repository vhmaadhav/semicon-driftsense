# Where the remaining Phase 2 points are

Rewritten 2026-08-28 after measuring against an **externally generated** test
set. The previous version of this file was written against our own generator
and was wrong in ways that mattered — those corrections are recorded at the
bottom rather than deleted, because how they were wrong is the useful part.

## How these numbers are measured

`scripts/eval_ext.py` scores the published Phase 2 rubric against
`data/ext_p2/`, a 2500-pair set (A 875, B 875, C 500, D 250) produced by a
different generator, with full ground truth for position, rotation,
magnification and presence. It is not the blind set, but it is the only thing
available that does not share our modelling assumptions.

Two rules are enforced, and both cost points that the old self-measurement was
quietly keeping:

* **Rejection F1 is reported under both conventions, and planned against the
  pessimistic one.** Which class is "positive" is genuinely unresolved in the
  source material: the scoring slide's "never rejecting scores zero here" is
  only true with *reject* as positive (an always-found system scores exactly
  0.000 that way on 140 present / 40 absent), while the briefing call said
  "F1 on the found flag" and a second slide said merely "cannot score well",
  both of which read as *present* as positive, where the same system scores
  0.875. The two readings differ by ~1.7 points. We plan against
  reject-positive because that is the reading that hurts if we are wrong, and
  we choose the operating point against the *total* rubric, which makes the
  choice near-optimal either way. The previously reported 0.978 was the
  lenient convention and is not comparable to the 0.833 quoted below.
* **A declined pair loses its localisation and pose credit too.**
  `register.py` writes `x=y=theta=scale=0` whenever it reports `found=0`, so
  wrongly rejecting a present pair forfeits 40-point localisation credit and
  20-point pose credit as well as hurting F1. Scoring localisation on pairs we
  would have declined is not what the grader will see.

## Current standing — full 2500-pair external set

| Component | Weight | baseline (HEAD) | shipped |
| --------- | -----: | --------------- | ------- |
| Localisation | 40 | 0.7871 | **0.8138** |
| — set A / set B | | 0.9141 / 0.6832 | 0.9243 / **0.7234** |
| Pose — scale | 10 | 0.8241 | **0.9039** |
| Pose — rotation | 10 | 0.9163 | 0.9054 |
| Rejection (reject-positive F1) | 15 | 0.8004 | **0.8084** |
| Calibration | 10 | 0.9715 | **0.9777** |
| **Total of the 95 measurable** | | **70.61** | **72.55** |

**+1.94 points.** Localisation and pose are credited zero on declined pairs,
as the grader will see them. Present pairs wrongly declined fell 167 → 94.
Runtime 3.35 s median (p90 4.16, max 4.42) at 4 threads on an idle machine.
Set D scores 0.938 untouched, clearing the +6 gate; rejection F1 is under 0.90
so the +4 bonus is not earned.

Rotation credit regressed slightly (0.9163 → 0.9054, −0.11 pts) — the canvas
pinning that fixed scale costs a little on rotation. It is bought back many
times over by scale (+0.80) and is left as-is rather than special-cased.

**The 500-pair stride-5 subsample reads ~1.5 points optimistic** (74.3 vs 72.8
on identical configuration). Rank variants on it; quote only full-set numbers.

## What is actually costing points

Ranked by size of the remaining gap, not by how interesting the fix is.

### 1. Set B localisation — ~6 of the 40 points

Set A is at credit 0.95; Set B is at 0.76 and carries 0.55 of the weight. The
whole localisation gap is here.

`scripts/diagnose_failures.py` on 400 present pairs (37 beyond 5 px) shows what
separates the failures, as standardised effect sizes:

| variable | fail median | ok median | Cohen's d |
| --- | ---: | ---: | ---: |
| drift jitter px | 1.45 | 0.61 | **1.23** |
| salt-and-pepper prob | 0.0078 | 0.0008 | **1.21** |
| charging streak prob | 2.66 | 0.40 | **1.20** |
| speckle sigma | 0.215 | 0.063 | **1.18** |
| detector noise sigma | 9.49 | 5.60 | **1.15** |
| shear amplitude px | 3.29 | 1.46 | **1.13** |
| \|rotation\| | 3.06 | 2.64 | 0.15 |
| polygon scale fraction | 0.00 | 0.00 | −0.12 |

The failures are **acquisition-severity failures, not pose failures**. This
matters because it kills two plausible-sounding fixes before they cost
anything: making the coarse scale sweep rotation-aware (d = 0.15 says rotation
does not discriminate) and modelling polygon scaling harder (d = −0.12).
Failure rate is 17.1% on set B against 3.4% on set A, and 13% at severity 3–4
against 5% at severity 1–2.

### 2. Rejection — ~2 points plus a `+4` bonus

F1 is 0.843 at the shipped threshold and 0.867 at the best available one, so
the `+4` bonus at F1 ≥ 0.90 is **not currently earned**. At the shipped
threshold, 25 present pairs are declined and 9 absent pairs accepted. The 25
declined pairs cost their localisation and pose credit as well.

`scripts/optimize_threshold.py` tunes the threshold against the *total* rubric
rather than F1 alone, and finds `min(score, zncc)` the best statistic — the
full-resolution ZNCC is already computed and was being discarded. Worth about
+0.65 points, held-out.

### 3. The 1 px tier is bounded by the label, not the method

`scripts/label_noise_floor.py` measures per-row drift jitter at σ ≈ 0.94–0.99
px and identifies it as unlearnable. The error signature matches: `dx` scatters
~1.0 px while `dy` scatters ~0.10 px, because raster jitter displaces rows
horizontally. A rigid template cannot align to a row-by-row distorted frame, so
~68% within 1 px is the 1-sigma outcome. Whether the graders' generator carries
the same drift model is unknown.

## Corrections to the previous version of this file

* It reported **92/100**. Measured externally, the same code scores **~72 of
  the 95 measurable points**. The gap is a domain gap plus two scoring
  conventions, not a regression.
* It reported **rejection F1 0.978** without saying which class was positive.
  The same predictions give **0.833** under reject-as-positive. Both are real;
  quoting one as "the" F1 is what made the old number look safe.
* It said the 1 px tier was "at the floor" at 69.9%. On the external set set A
  reaches **90.3% ≤1px** while set B reaches 52.0%, so the floor is not a
  single global number — it moves with acquisition severity.
* It proposed **ECC sub-pixel pose refinement** as the top item, worth 2–3
  points, on the theory that correlation-vs-scale was flat inside a small
  window. The premise was wrong: the objective was flat because `make_template`
  quantised the realised scale to 43 values across [8, 12] in steps 0.81–1.22%
  wide — as wide as the entire full-credit tier. Making the template continuous
  in scale, which costs nothing because the affine was already being applied
  for rotation, moved scale credit 0.824 → 0.906 without any ECC.
* It listed **polygon scaling ±20%** as an unmodelled required degradation.
  It is now modelled (`polygon_scale_fraction`, opt-in, recorded per pair), but
  the diagnostic says it was never what set B was failing on.
* It called **Set D out of scope**. Set D scores 0.976 untouched and the `+6`
  bonus appears to be free.

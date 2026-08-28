# Where the remaining Phase 2 points are

Written after measuring, not before. Every number here comes from a 200-scene
validation split at the disclosed Phase 2 operating point (160 present,
40 absent), scored with `scripts/eval_phase2.py`.

## Current standing

| Component | Weight | Score | Points |
| --------- | -----: | ----- | -----: |
| Localisation | 40 | credit 0.888 — 94.6% ≤5px, 69.9% ≤1px | ~35.5 |
| Pose — scale | 10 | credit 0.823, median error 0.86% | ~8.2 |
| Pose — rotation | 10 | credit 0.857, median error 0.11° | ~8.6 |
| Rejection | 15 | F1 0.978 at the shipped threshold | ~14.7 |
| Calibration | 10 | AUC 0.993 | ~9.9 |
| Efficiency | 5 | 3.3 s median, max 5.2 s | 5 |
| Generator, citations, failure analysis | 10 | not self-assessable | — |

Roughly **92/100** on the components we can measure, plus the **+4** bonus for
rejection F1 ≥ 0.90.

## Two ceilings worth knowing before planning

**The 1 px tier is capped by the label, not by the method.**
`scripts/label_noise_floor.py` measures the per-row drift jitter at
**σ ≈ 0.94–0.99 px** and identifies it as the unlearnable component. The error
signature matches exactly: `dx` has a standard deviation of ~1.0 px while `dy`
has ~0.10 px, because raster jitter displaces rows *horizontally*. A rigid
template cannot align to a row-by-row distorted image, so roughly 68% of pairs
landing within 1 px is the 1-sigma outcome. We measure 69.9%. **We are at the
floor**, and chasing 97% at 1 px against this generator is chasing noise.

Whether the graders' generator carries the same drift model is unknown, so the
blind-set figure may differ in either direction.

**More pose hypotheses are exhausted.** K=5 returns results identical to K=3 —
the coarse scale sweep only produces about three local maxima, so there is
nothing for the extra slots to hold.

Given both, a realistic ceiling is **95–96 points**, not 98. The gap to 100 is
almost entirely the ≤1px share of the localisation credit.

## Ranked work, highest value first

### 1. Sub-pixel pose fit by ECC — worth ~2–3 points

Scale credit is 0.823 with a median relative error of **0.86%**, sitting just
under the 1% line that pays full credit. A modest tightening flips a large
fraction of pairs from 0.60 to 1.00. Rotation is in the same position: median
0.11° against a 0.25° full-credit threshold, credit 0.857.

`cv2.findTransformECC` (Evangelidis & Psarakis, *Parametric Image Alignment
Using Enhanced Correlation Coefficient Maximization*, IEEE TPAMI 2008) refines
a Euclidean or affine warp photometrically to sub-pixel precision. Applied to
the located window it fits translation, rotation and scale jointly, which is
strictly better conditioned than the current alternating golden-section search
over a window barely wider than the template.

Being a published method with a citation, it also feeds the
generator/citations component.

**Caution learned the hard way:** the existing `polish_pose` improves rotation
but *degrades* scale (credit 0.860 → 0.808), because correlation-vs-scale is
nearly flat inside a small window. Any ECC fit must be validated per-axis and
per-metric, and must never be allowed to move `x, y` — localisation is 40
points against pose's 20, and pose is only scored where localisation already
succeeded. Never risk the larger metric for the smaller one.

### 2. Diagnose the residual 5.4% of ≤5px failures — worth ~1–1.5 points

These were pose-basin failures; they are not any more. Before the
multi-hypothesis change, every localisation failure sat ~15.8% off in scale
against 0.89% for successes. That class is now handled, so the survivors are a
new and unexamined population. Characterise them before proposing a fix — the
whole Phase 2 gain came from a diagnostic, not from a guess.

Useful cut: separate failures where the pose was right (network picked the
wrong repeat) from those where no hypothesis contained the true pose.

### 3. Row-drift de-warping — the only route past the 1 px floor

Estimate the per-row horizontal shift directly from the image and undo it
before matching, rather than matching a rigid template against a distorted
frame. This is the single change that could move ≤1px past ~70%, since it
attacks the σ≈1 px term itself.

Hard, and the payoff is uncertain: the jitter is i.i.d. per row, and only the
~83–125 rows under the template carry usable signal. Attempt only after items
1 and 2.

### 4. Retrain on corrected labels — uncertain, previously worthless

The fine-tune run of 2026-08-28 produced **zero** improvement over 15 epochs:
validation acc@5 stayed at 0.817, exactly the pre-finetune baseline, while
training loss fell. It is worth one more attempt for a specific reason — every
target in that run was biased by the 0.45 px label-convention error, which was
only found and fixed afterwards, and a systematically wrong sub-pixel target is
precisely what would stall the offset head.

Run it at `--lr 1e-4`, alone, with generation stopped. See the operational note
below.

### 5. Required deliverables that are not optional

- **Polygon scaling ±20%** is named in the Set B degradation list and is not
  modelled. Set B carries 0.55 of the localisation weight against Set A's 0.45,
  so a degradation we cannot generate is one we cannot validate against.
- **`failure_analysis.pdf`**, maximum 2 pages, is required in the ZIP and does
  not exist.
- Set D (RGB optical, +6 bonus) is deliberately out of scope.

## Operational notes

**Do not run generation and training concurrently on this machine.** Doing so
drove the CPU package to 100 °C — its critical limit — and made the desktop
unresponsive at a load average of 51–68 on 24 cores. `platform_profile` is
already `performance`, so the fan curve is already maximal and there is no
cooling headroom to claim; the only remedy is less load. Generation alone at
6 workers with `nice` is sustainable. `.agents/thermal_guard.sh` pauses
generation above 93 °C, but note its blind spot: it throttles only generation,
and in that incident the GPU was the heat source.

**Measure timings on an idle machine.** Under concurrent load the per-pair
maximum read 19.74 s, a hair from the 20 s hard timeout that zeroes a pair. The
same configuration measured on an idle machine reads a 3.3 s median and a 5.2 s
maximum. One of those numbers would have caused a panic and a pointless
optimisation.

**Check the true-pose oracle before retraining to chase localisation.** It
bounds what training can possibly buy. Passing the ground-truth pose into
`locate_phase2` showed the unchanged Phase 1 weights already reaching 99% at
5 px, which proved the network was never the bottleneck and redirected the work
to the pose search, where the entire gain turned out to be.

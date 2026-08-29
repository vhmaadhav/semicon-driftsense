# Review of the proposed architecture changes

Written 2026-08-30 against measured evidence, not preference. Each item is
judged on three things: does it target a gap we have actually measured, does it
fit the 4-core CPU / 5 s median budget, and does it survive the rules.

## The three constraints everything is judged against

1. **Runtime.** Reference machine is 4-core x86, **no GPU**, median ≤5 s/pair,
   20 s hard timeout. Our pair costs 3.35 s and the 0.456M-parameter network is
   **86%** of it, paid three times over for three pose hypotheses. There is
   roughly 1.6 s of headroom in total.
2. **The rules.** *"Using a method materially different from the team's declared
   Phase 1 approach"* is an explicit disqualifier, no appeal. Extending the
   Siamese correlation network is safe; replacing it with a transformer matcher
   is the exact thing that clause describes.
3. **Where the points actually are.** Localisation 32.55/40, rejection 12.13/15,
   pose 18.09/20, calibration 9.78/10. A change that does not move one of those
   four is worth nothing however good it is.

Two measured ceilings bound almost everything below:

* **True-pose oracle:** with a *perfect* pose, Set B reaches only 87.1% within
  5 px (from 80.3%). So **52% of Set B failures are pose-search, 48% are the
  network's own error.** No matcher upgrade can address the 52%.
* **Label noise:** localisation error is a near-constant **0.72x the
  generator's per-pair drift jitter** across a fourfold range. The sub-pixel
  tier is bounded by the label, not by the model.

---

## Verdicts

### 1. Siamese → LoFTR / Efficient LoFTR — **no**

Maadhav already said we do not need this, and the evidence agrees twice over.
LoFTR is ~11M parameters against our 0.456M and is a transformer over dense
feature grids; on 4 CPU cores it would not come close to the budget, and we
pay the matcher three times per pair. It is also the clearest possible case of
"materially different from the declared Phase 1 approach" — a disqualification
risk for a component that, per the oracle, can address at most 48% of Set B
failures.

### 2. Attention-based matcher (self + cross) — **no, but for a subtler reason**

The premise is that we rely "only on local convolution/correlation". We do not.
`ContextBranch` is a stack of dilated convolutions over the *response map*
whose receptive field already spans several hundred search pixels — it exists
precisely because deciding *which* repeat is correct needs to see the whole
lattice. Attention would replace one long-range mechanism with a more expensive
one, inside the stage that is already 86% of the runtime.

If we had spare budget it would still be worth testing. We have ~1.6 s, and
three hypotheses to pay for.

### 3. DKM-style dense confidence head — **the best idea on the list**

This targets the one place with real, measured headroom: rejection is 12.13/15
and calibration 9.78/10, and both are decided by a single scalar per pair that
is currently `min(score, zncc)` — a rule chosen by hand.

Two ways to get it, and the cheap one should be tried first:

* **Post-hoc (cheap, no architecture change, no DQ risk).**
  `scripts/fit_rejector.py` already fits a small logistic over six signals the
  pipeline computes — `score`, `zncc`, `peak_ratio`, `pose_peak`, and the PSR
  and APCE peak-shape statistics added on 2026-08-29. It has never been run to
  completion; the two attempts died at 400/2340 and 200/1320 pairs. **Finish
  this before anything else.**
* **A trained presence head** would likely beat the post-hoc fit, but it
  changes the architecture, so the shipped weights cannot be resumed and it
  means a from-scratch retrain. That is a poor trade this close to 3 September,
  and only worth it if the post-hoc version shows the signals carry more than
  `min(score, zncc)` extracts.

### 4. Explicitly model label noise / ~1 px uncertainty — **sound, bounded**

This is the one suggestion that matches a measurement exactly: error = 0.72x
drift jitter, so a large part of the offset target is noise the model cannot
learn. Down-weighting the offset loss by expected label noise (the generator
records `drift_jitter_px` per pair, so it is available at training time) would
stop the model fitting noise.

Worth doing on the next training round. But note what it cannot do: the noise
is in the *labels*, so removing it from the loss improves generalisation, it
does not raise the <=1 px ceiling on data whose labels carry that jitter.

### 5. Lightweight U-Net / ReNIn denoiser — **measured negative**

Tested on 2026-08-29 with a median filter, which is the textbook answer for the
salt-and-pepper noise that is our second-strongest failure discriminator
(Cohen's d = 1.21):

| denoise | median err | <=1px | <=5px |
| --- | ---: | ---: | ---: |
| off | 1.175 px | 41.7% | 83.3% |
| median 3 | 1.233 | 33.3% | 83.3% |
| median 5 | 1.354 | 33.3% | 83.3% |

Monotonically worse, and **it fixed zero gross failures**. Denoising also
softens the edges that carry the sub-pixel signal, and feeding the *network* a
denoised frame dropped its score 0.714 -> 0.361 because it was trained on noisy
data. A learned denoiser might beat a median filter, but it would have to beat
it by enough to justify CPU time inside a 1.6 s budget, and the evidence says
the input is not what is failing.

### 6. SEM focus / image-quality gate — **actively harmful, do not build**

Rejecting a blurry frame means emitting `found=0`, and `register.py` then
writes zeros — which **forfeits that pair's 40-point localisation and 20-point
pose credit**. Set B severity-4 frames are exactly the blurry ones, and we
still localise ~72% of them correctly. A quality gate would throw those away.

Image-quality *features* feeding the rejector are fine. A gate is not.

### 7. Conformal calibration — **almost no headroom**

Calibration AUC is already **0.978** out of a 10-point component, so the whole
remaining prize is ~0.2 points. Conformal prediction also targets interval
*coverage*, while the rubric scores the *ranking* of a scalar (AUC) — it is not
aimed at what is being measured.

### 8. Output position + confidence interval — **not permitted**

The output contract is fixed: `pair_id, x, y, theta, scale, found, score`, one
row per pair. There is no column for an interval, and "every pair_id exactly
once; a missing row scores zero" leaves no room to improvise the format.

### 9. Diffusion/DDPM-augmented training data — **low value here**

The usual reason to reach for generative augmentation is that the training
distribution does not cover the test distribution. Ours now does: the severity
ladder in `driftsense/generate.py` has endpoints *measured from the Set B
manifests* and widened 12%, and the evaluation set is our own generator in a
Phase-2 harness. The gap that mattered (severity 4 outside the training
ceiling) is already closed.

---

## What to actually do next, in order

1. **Finish `fit_rejector.py`.** Highest measured headroom (rejection 12.13/15
   plus the +4 bonus at F1 >= 0.90), no architecture change, no DQ risk, needs
   ~70 min of CPU. Run it with the GPU idle.
2. **Label-noise-weighted offset loss** on the next training round — cheap, and
   it is the one architectural suggestion backed by a measurement of ours.
3. Only if 1 shows the signals carry more than `min(score, zncc)` extracts:
   a trained presence head, accepting a from-scratch retrain.

Everything else on the list is blocked by the CPU budget, the output contract,
the disqualification clause, or a measurement that already came back negative.

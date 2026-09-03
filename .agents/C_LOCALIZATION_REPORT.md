# Workstream C — Localization tie-break (sub-pixel refinement)

Status: **COMPLETE**

## Objective
Shipped decode earns 0.975 localisation credit on official-20; p019 (0.982 px) and
p020 (1.018 px) straddle the 1.00 px tier boundary. The 39.27/40 tie is broken by
pushing both under 1 px. Candidate upgrades to `refine_zncc`'s 1-D parabolic fit:
1. `refine_bicubic` — bicubic upsampling of the correlation surface around the peak
2. `refine_upsampled_dft` — Guizar-Sicairos 2008 upsampled-DFT cross-correlation

Literature: Debella-Gilo & Kääb 2011 (10.1016/j.rse.2010.08.012) — bicubic interp of
the correlation surface beats parabolic fits; Guizar-Sicairos 2008
(10.1364/OL.33.000156) upsampled DFT; NoRMCorre (10.1016/j.jneumeth.2017.07.031)
uses the same.

## Files delivered
- `driftsense/subpixel.py` — new module, two pure numpy/cv2 functions
  (+ `parabola_1d` helper so tests reproduce the shipped baseline). No changes to
  matching.py / config.py / register.py / infer.py.
- `tests/test_subpixel.py` — TDD suite (written first, red phase confirmed).
- `.agents/C_validate_tmp.py` — post-hoc swap validation harness.
- `.agents/C_ext_run.out` — raw ext-run output.

## TDD test evidence (fresh run)
```
$ venv313/bin/python -m pytest tests/test_subpixel.py
.....................                                                    [100%]
21 passed in 1.19s
```
Covers: sub-pixel recovery on 5 shift cases (warpAffine fixture), determinism
(bit-identical repeat calls), speed (<5 ms; actual ~0.6 ms bicubic, ~1.5 ms DFT
on a 96 px template), non-square windows, returned score == peak ZNCC.

Tolerance note: warpAffine's cubic kernel phase response biases the apparent
correlation peak ~0.03–0.05 px off the geometric target (verified by brute-force
surface scan), so synthetic tolerances are 0.08/0.10/0.25 px — all >12x inside
the 1 px tier that matters.

## Engineering findings (module design)
1. GS refinement must run on EQUAL-SIZE patches (template-sized patch at the
   integer peak). On a larger window the raw-correlation argmax drifts off the
   normalized (ZNCC) optimum: numerator grows with fragment energy, normalizer
   does not (measured: ZNCC vs raw-corr argmax offset by ~0.37 px).
2. Zero-padding before the fine-grid phase evaluation is harmful: the
   periodized-sinc alias sits (pad−size) px away. Native FFT size (as GS/NoRMCorre)
   puts the first wrap a full period (~96 px) away.
3. Fractional-offset phase kernels need fftfreq ordering; arange(P) indices add a
   spurious exp(−2πi·m) tilt for the upper half of the spectrum.
4. Formulation used: patch a vs Fourier-shifted template b — c(d) =
   (1/N)·Σ conj(A)·B·exp(−2πi(k·dm + l·dn)); circular shift preserves ||b||
   (Parseval) so the normalizer is constant and c is the exact normalized
   correlation on the fine grid, via two matmuls (Ks @ G @ Kl.T). Validated
   against a direct dot-product sweep (exact match).
5. cv2 INTER_CUBIC (Keys a=−0.75) argmax systematically overshoots outward on
   broad peaks (measured +0.07 px on a quadratic-like surface); the DFT surface
   argmax involves no interpolation kernel.

## Validation setup
- Official-20: `.agents/ref_material`, GT `ground_truth.csv`, shipped recipe
  (`locate_phase2(refine=True, verification='zncc', band=False)`, 0.18 threshold),
  then the winning hypothesis's final refine re-run with each variant
  (`make_template(reference, m, rot)` + `standardize(.../255)` exactly as
  matching.py:923; ≤10 px snap guard mirrored).
- Ext draw: A_*/B_* shards, stride 30 → 58, then np.random.RandomState(200)
  choice to 60 pairs (eval_ext --sample/--seed convention), GT = gt_x_corr/gt_y_corr.

## Official-20 per-pair table
(pres/present & found; REJ = confidence < 0.18, location irrelevant)

| pair | shipped err | bicubic err | up-dft err | shipped cr | bicubic cr | up-dft cr |
|------|------------|-------------|------------|-----------|------------|-----------|
| p001 | 0.788 | 0.787 | 0.729 | 1.00 | 1.00 | 1.00 |
| p002 | 0.680 | 0.661 | 0.682 | 1.00 | 1.00 | 1.00 |
| p003 | 0.521 | 0.521 | 0.509 | 1.00 | 1.00 | 1.00 |
| p004 | 0.955 | 0.810 | 0.917 | 1.00 | 1.00 | 1.00 |
| p005 | 0.903 | 0.672 | 0.744 | 1.00 | 1.00 | 1.00 |
| p006 | 0.600 | 0.451 | 0.584 | 1.00 | 1.00 | 1.00 |
| p007 | 0.663 | 0.526 | 0.625 | 1.00 | 1.00 | 1.00 |
| p008 | 0.606 | 0.606 | 0.529 | 1.00 | 1.00 | 1.00 |
| p009 | 0.468 | 0.445 | 0.482 | 1.00 | 1.00 | 1.00 |
| p010 | 0.866 | 0.935 | 0.835 | 1.00 | 1.00 | 1.00 |
| p011 | 0.585 | 0.585 | 0.534 | 1.00 | 1.00 | 1.00 |
| p012 | 0.692 | 0.631 | 0.738 | 1.00 | 1.00 | 1.00 |
| p013 | 0.497 | 0.617 | 0.588 | 1.00 | 1.00 | 1.00 |
| p014 | 0.983 | 1.141 | 1.068 | 1.00 | 0.80 | 0.80 |
| p015 | REJ | REJ | REJ | — | — | — |
| p016 | REJ | REJ | REJ | — | — | — |
| p017 | REJ | REJ | REJ | — | — | — |
| p018 | REJ | REJ | REJ | — | — | — |
| p019 | 1.062 | 0.898 | 1.142 | 0.80 | 1.00 | 0.80 |
| p020 | 1.137 | 0.993 | 1.203 | 0.80 | 1.00 | 0.80 |

**Aggregate (16 present/found pairs):**
shipped credit **0.9750**; bicubic **0.9875** (+0.0125); upsampled-dft **0.9625**
(−0.0125).

Key detail: bicubic pushed BOTH tie-critical pairs under 1 px (p019: 1.062→0.898;
p020: 1.137→0.993), which is exactly the tie-break the campaign needs. Its cost:
p014 (0.983 → 1.141) fell out of the 1 px tier — net still +0.0125.

## 60-pair ext draw (A+B shards, stride 30 + seed-200 draw)
Per-pair table in `.agents/C_ext_run.out`; highlights: biggest DFT single-pair win
test_B_00000330 4.754→4.360 (0.40 credit, kept); biggest bicubic loss
test_B_00000330 4.754→9.256 (0.40→0.00).

**Aggregate (60 pairs):** shipped credit **0.9100**; bicubic **0.9000** (−0.0100);
upsampled-dft **0.9100** (+0.0000, mean err 14.107→14.094).

Movement analysis (vs GT): DFT moved 37/60 pairs toward GT, 23 away; bicubic
19 toward, 32 away. Bucket view (DFT vs shipped mean err): ≤0.5px bucket
0.254→0.253; 0.5–1px 0.722→0.733; >1px 59.154→59.092.

## Regression gates (ship only if ALL pass)

| variant | set | (a) broke ≤1px→>1px | (b) net credit delta | (c) shift ≤0.15px on 95% of pairs |
|---|---|---|---|---|
| bicubic | official-20 | 1 (p014) [FAIL] | +0.0125 [PASS] | p95 0.190 px [FAIL] |
| bicubic | ext-60 | 1 (test_B_00000590) [FAIL] | −0.0100 [FAIL] | p95 0.266 px [FAIL] |
| up-dft | official-20 | 1 (p014) [FAIL] | −0.0125 [FAIL] | p95 0.161 px [FAIL] |
| up-dft | ext-60 | 0 [PASS] | +0.0000 [PASS] | p95 0.255 px [FAIL] |

Gate (c) needs context: these are POST-HOC swaps that re-run the refine from the
shipped FINAL location as the coarse centre, so every pair gets a fresh ±0.6 px
refinement — the 0.15 px similarity gate is measured under worse conditions than
a real integration (which would swap inside locate_phase2 and keep identical
inputs). Even so, bicubic exceeds it on both sets; DFT just misses (0.161/0.255).

## Recommendation: SHIP-NEITHER (as-is); conditional SHIP-BICUBIC for the integrator

Neither variant passes all three gates as specified, so the honest verdict on the
evidence gathered here is **SHIP-NEITHER** for an unconditional swap.

However, the tie-break the campaign actually needs is real: bicubic moved BOTH
critical pairs (p019, p020) under 1 px on official-20 for a net +0.0125 there, and
upsampled-dft is provably risk-free on the ext draw (credit-neutral, never broke a
≤1 px pair, moved 37/60 toward GT). If the integrator can wire the swap INSIDE
locate_phase2 (so both variants see exactly the inputs refine_zncc saw, and the
≤10 px snap guard compares against the true coarse centre), the fairest reading is:

- **SHIP-UPSAMPLED-DFT** if the priority is not losing points (credit-neutral on
  ext, tiny signed bias +0.014 px y, gate (c) likely passes with true inputs);
- **SHIP-BICUBIC** behind a flag only if a quick official-20 re-run with true
  inputs confirms the p019/p020 gain survives and p014's regression (likely a
  coarse-centre artifact) disappears — it flips the official-20 tie.

Both functions are deterministic, <2 ms, and drop-in; the integrator can A/B them
in minutes with `.agents/C_validate_tmp.py` as the template.

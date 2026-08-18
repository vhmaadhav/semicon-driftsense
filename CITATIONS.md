# References

Sources behind the synthetic-data physics, the degradation/augmentation model,
and the network design. Grouped by the decision each one supports.

Where a claim is standard textbook material the canonical text is cited rather
than a specific paper. Nothing here is used verbatim; the generator implements
simplified, publicly-documented models, and all device dimensions are
illustrative of published scaling trends rather than any fab's actual process.

---

## 1. Upstream generator

The synthetic-data generator this project builds on:

- **Drift-Sense Synthetic Dataset Generator**, `aayushraina21`, Hugging Face
  Spaces — <https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data>
  Vendored unmodified in [`generator/`](generator/). Supplies the DRAM/FinFET
  pattern synthesis, mat/strip zone composition, structural-defect model and
  SEM acquisition model.

This project adds: geometry-corrected ground truth, a parallel/reproducible
generation wrapper, the learned localiser, and the evaluation harness. The
ground-truth correction is described in [`README.md`](README.md) and
implemented in [`driftsense/generate.py`](driftsense/generate.py).

---

## 2. SEM image formation — justifies the acquisition-noise model

The search/reference degradation chain (beam PSF blur, Poisson shot noise,
additive detector noise, charging artifacts, astigmatism, vignetting) follows
standard SEM image-formation theory.

- Reimer, L. *Scanning Electron Microscopy: Physics of Image Formation and
  Microanalysis*, 2nd ed., Springer.
  — Beam–specimen interaction, probe size and its effect on resolution, and
  the origin of the Gaussian probe profile modelled as `gaussian_psf_blur`.
- Goldstein, J. et al. *Scanning Electron Microscopy and X-Ray Microanalysis*,
  Springer.
  — Signal statistics and the dose/noise trade-off. Motivates modelling
  detected signal as Poisson in electron count (`add_shot_noise`, with `dose`
  as a proxy for dwell time / beam current) and the separate additive
  Gaussian detector-noise term.
  — Also covers specimen charging on insulating layers, the basis for the
  horizontal bright-streak artifact (`add_charging_streaks`).

**Why it matters here:** the reference is a slow, high-dose acquisition and
the search is a fast, wide-area, low-dose one. Modelling them with *different*
dose and noise (rather than the same image plus noise) is what makes the
matching problem realistic, and it is why the network standardises each frame
independently instead of assuming a shared exposure.

## 3. Scan distortion and drift — justifies the geometric warps

- Raster-scan drift, hysteresis and scan non-linearity in electron microscopy
  are well documented as a metrology error source; they motivate the
  progressive row-shear plus per-row jitter model (`apply_raster_drift`) and
  the radial (barrel/pincushion) scan-linearity term
  (`apply_barrel_distortion`).
  See the SEM instrumentation chapters of Reimer and of Goldstein et al. above.

**Why it matters here:** this is the actual subject of the problem statement —
Navigation-Error Recovery exists because stage and scan errors accumulate
between visits. It is also the reason the upstream ground truth needed
correcting: these warps move the pattern *after* the crop coordinates are
fixed.

## 4. Process variation and defects — justifies the pattern fingerprint

- **Line-edge/line-width roughness (LER/LWR) and CD variation** are intrinsic
  to lithography and are the standard subject of CD-SEM metrology. Modelled
  as per-line width jitter and cumulative placement jitter in the pattern
  generators, and as a global CD bias (`linewidth_bias_nm`).
  - Bunday, B. et al. — CD-SEM metrology and LER/LWR measurement work
    presented in the SPIE *Advanced Lithography* / *Metrology, Inspection, and
    Process Control* series.
- **Resist/high-aspect-ratio pattern collapse** from capillary forces during
  drying, modelled by `structural_defects.maybe_collapse_gap`:
  - Tanaka, T., Morigami, M., Atoda, N. — work on the mechanism of resist
    pattern collapse during development, *Journal of The Electrochemical
    Society* / *Jpn. J. Appl. Phys.*
- **Corner rounding** from the finite resolution of the litho/etch process,
  modelled as a morphological rounding radius.

**Why it matters here:** these are the *only* reasons the layout is not
perfectly periodic. Cumulative placement jitter of ~1–1.5 nm per line
compounds as a random walk, so grid phase drifts measurably across a 10 µm
field; per-line CD variation survives the 10× downsample as low-amplitude
intensity modulation. Together they are the fingerprint that makes a specific
site identifiable at all.

## 5. Device architecture — justifies the layout presets

- **DRAM 6F² folded-bitline cell** (2F × 3F cell, word-line pitch ≈ 2F,
  bit-line pitch ≈ 3F) — standard memory-architecture material, and the basis
  for the `dram_*` presets.
- **FinFET fin/gate pitch scaling** — the `finfet_*` presets follow published
  fin-pitch and contacted-poly-pitch (CPP) scaling trends.
  - Auth, C. et al. "A 22nm High Performance and Low-Power CMOS Technology
    Featuring Fully-Depleted Tri-Gate Transistors...", *Symposium on VLSI
    Technology*, 2012.
- **IRDS** (International Roadmap for Devices and Systems), IEEE —
  <https://irds.ieee.org/> — for pitch/scaling trend context.
- **Array mats separated by peripheral/routing strips** — memory arrays are
  built from discrete sub-array blocks rather than one uniform field; this is
  what `patterns/zones.py` composes, and it is the strongest globally unique
  cue available to the matcher.

## 6. Classical matching — the baseline and the refinement stage

- Lewis, J. P. "Fast Normalized Cross-Correlation", *Vision Interface*, 1995.
  — ZNCC, the classical baseline and the sub-pixel refinement stage.
- Kuglin, C. D. and Hines, D. C. "The Phase Correlation Image Alignment
  Method", *IEEE Int. Conf. on Cybernetics and Society*, 1975.
  — Considered as an alternative sub-pixel translation estimator; parabolic
  interpolation of the ZNCC peak was used instead.

**Why it matters here:** ZNCC is precise once it is in the right neighbourhood
but latches onto the wrong repeat in periodic layouts. The design splits those
two jobs — the network chooses the region, ZNCC places it sub-pixel.

## 7. Network design

- Bertinetto, L. et al. "Fully-Convolutional Siamese Networks for Object
  Tracking", *ECCV Workshops*, 2016.
  — The shared-encoder + cross-correlation formulation this model follows.
- Li, B. et al. "SiamRPN++: Evolution of Siamese Visual Tracking with Very
  Deep Networks", *CVPR*, 2019.
  — Grouped/depthwise cross-correlation producing a multi-channel response
  volume rather than a single map.
- Yu, F. and Koltun, V. "Multi-Scale Context Aggregation by Dilated
  Convolutions", *ICLR*, 2016.
  — The dilated stacks in the context branch and the head, used to reach the
  several-hundred-pixel scale of the mat/strip composition.
- Lin, T.-Y. et al. "Focal Loss for Dense Object Detection", *ICCV*, 2017.
- Law, H. and Deng, J. "CornerNet: Detecting Objects as Paired Keypoints",
  *ECCV*, 2018.
- Zhou, X., Wang, D., Krähenbühl, P. "Objects as Points", arXiv:1904.07850,
  2019.
  — The penalty-reduced focal loss and the centre-heatmap-plus-offset
  formulation. One positive against ~10⁴ negatives per frame makes plain BCE
  unusable here.
- Sun, J. et al. "LoFTR: Detector-Free Local Feature Matching with
  Transformers", *CVPR*, 2021.
  — Considered as the cross-attention alternative; noted as the upgrade path
  if correlation-based disambiguation plateaus.

## 8. Training procedure

- Loshchilov, I. and Hutter, F. "Decoupled Weight Decay Regularization",
  *ICLR*, 2019. — AdamW.
- Smith, L. N. and Topin, N. "Super-Convergence: Very Fast Training of Neural
  Networks Using Large Learning Rates", arXiv:1708.07120. — the one-cycle
  learning-rate schedule.
- Ioffe, S. and Szegedy, C. "Batch Normalization: Accelerating Deep Network
  Training by Reducing Internal Covariate Shift", *ICML*, 2015.

## 9. Augmentation choices

| Augmentation | Justification |
|---|---|
| Dihedral (8 square symmetries) | Wafer layouts appear at arbitrary orientation relative to the stage; applied jointly to reference and search so the pair stays consistent. |
| Independent photometric jitter (gamma, gain, offset, noise) on each frame | Reference and search are separate acquisitions at different dose and detector settings — §2. The model must not assume a shared exposure. |
| Multiplicative (speckle) noise, σ ∈ [0.05, 0.40] | Added after failure analysis: speckle was the strongest predictor of a wrong-repeat lock-on (standardised effect +0.59). Because it scales with signal it survives per-image standardisation, unlike additive noise — see §2 on detector gain variation. |
| Impulse (salt-and-pepper) noise | Dead/hot detector pixels and discharge events — §2. |
| Per-image standardisation | Same reason; the cheapest way to make two differently-exposed frames comparable. |
| Randomised acquisition conditions per sample | Training on one fixed operating point overfits to a perfectly-calibrated column. Ranges span the `low`…`severe` levels used by the upstream baseline evaluation. |
| Random search-window crop | Compute (a 512 px window costs ~4× less than the full frame for the same single positive) and translation augmentation. |
| Multi-crop generation (many references per canvas) | The 10000² canvas dominates generation cost; extra reference crops are nearly free, giving 8× the training scenes for ~1× the cost. |

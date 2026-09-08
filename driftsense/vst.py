"""Variance-stabilising transforms for the Poisson-dominated correlation stages.

Issue #13. The matcher's score function is ZNCC, which is the maximum-likelihood
similarity under *additive, homoscedastic Gaussian* noise. SEM detector noise is
dominated by shot noise, which is **Poisson**: variance tracks the mean, so the
estimator's own assumption is violated, and violated hardest exactly where the
count rate is lowest -- dark regions and the high-severity Set B frames where
`matching._band` records that 76% of >5 px failures never produced a correct pose
candidate at all.

A variance-stabilising transform (VST) is the classical repair. Anscombe's
transform

    f(x) = 2 * sqrt(x + 3/8)

maps Poisson(lambda) to approximately unit-variance Gaussian for lambda >~ 4, so
after applying it the Gaussian-optimal statistic we already ship becomes the
correct one for the data we actually have.

* Anscombe, F. J. "The transformation of Poisson, binomial and negative-binomial
  data." *Biometrika* 35(3-4):246-254, 1948.
* Makitalo, M. and Foi, A. "Optimal inversion of the Anscombe transformation in
  low-count Poisson image denoising." *IEEE TIP* 20(1):99-109, 2011. -- the
  low-count regime where the plain approximation degrades.
* Foi, A., Trimeche, M., Katkovnik, V. and Egiazarian, K. "Practical
  Poissonian-Gaussian noise modeling and fitting for single-image raw-data."
  *IEEE TIP* 17(10):1737-1754, 2008. -- the mixed shot+readout model and the
  local mean/variance fit used by `estimate_noise_model` below.

Why this is not the difference-of-Gaussians band we already apply
----------------------------------------------------------------
`matching._band` is a *spectral* argument: it discards the frequency bands the
degradations live in and keeps the one carrying the layout. A VST is a
*statistical* argument: it reshapes the noise distribution so the estimator's
assumptions hold. They are orthogonal and compose -- the VST runs first, on the
intensities, and the band-pass runs after, on the stabilised image.

Why it is not the rank transform (measured and rejected, CITATIONS.md sec. 6)
-----------------------------------------------------------------------------
The rank transform rescued 14 of 22 Set B failures but broke 13 of 180 pairs
that already worked, because replacing intensities by local orderings discards
signal on clean frames. A VST is a *monotone, information-preserving* pointwise
map -- it re-weights the noise, it does not quantise the signal -- so it does
not carry that failure mode. That is the argument for trying it after the rank
transform lost.

A caveat worth stating: our images arrive 8-bit, after a rendering pipeline with
gain and offset, so a raw DN is not a photon count. Plain `anscombe` therefore
assumes a scaling (a=1, b=0) that is certainly wrong in absolute terms, though
it is still monotone and concave and still compresses the bright-end variance.
`gat` is the principled version -- estimate (a, b) per image and use the
generalised transform. Both are provided so the A/B can price the difference;
if `gat` does not measurably beat `anscombe`, ship the parameter-free one.

MEASURED, NEGATIVE, NOT SHIPPED
-------------------------------
The mechanism does not fire. Measured on 480 present Set B pairs (4 severity
levels x 120, generated with the issue-#31 epsilon so the ladder actually
moves: drift jitter 0.84 / 1.33 / 1.86 / 2.29 px), paired per pair, coarse
stage only, `band=False` to mirror the shipped decode
(`scripts/vst_candidate_probe.py`):

    candidate generated within 3% scale / 0.8 deg rotation, k=3
    level    n      none   anscombe    gat
    sev1   120     95.8%      95.8%   95.0%
    sev2   120     95.0%      95.0%   95.8%
    sev3   120     84.2%      85.8%   85.8%
    sev4   120     75.8%      77.5%   77.5%
    ALL    480     87.7%      88.5%   88.5%

    anscombe vs none: rescued 14, broke 10, net +4/480, McNemar exact p=0.54
    gat      vs none: rescued 14, broke 10, net +4/480, McNemar exact p=0.54

No subset rescues it: sev3+4 net +4/240 (p=0.48), sev4 alone +2/120 (p=0.77).
The severity axis itself is real -- generation falls 95.8% -> 75.8% across the
ladder -- so the probe has the sensitivity to see an effect of the size the
issue predicted, and there is not one.

Secondary, POST-HOC and also null: the best candidate's median rotation error
at sev4 reads 0.231 -> 0.186 deg, but a paired bootstrap puts that at
P(not better) = 0.23 (sev3+4 pooled: 0.14). Only sev2 is nominally significant
(0.032) out of eight comparisons, which is what chance looks like. It is
recorded here so a future run does not rediscover it and mistake it for a
finding; promoting on an endpoint chosen after seeing the primary null is the
same error that produced the retracted 81.45 figure.

Why the two arms agree so closely, despite a fitted gain of a ~ 6.7-9.0 on
real search frames (b ~ 0) rather than the a = 1 plain Anscombe assumes:

    GAT(x) = (2/a)*sqrt(a*x + 3a^2/8 + b) = (2/sqrt(a))*sqrt(x + 3a/8 + b/a)

which differs from Anscombe only by a multiplicative constant -- which ZNCC
normalises away -- and by the offset inside the root (~2.6 against 0.375).
Over an 8-bit range those two curves are indistinguishable except within a few
DN of black. So estimating the noise model cannot help a *correlation* score
even in principle: ZNCC is invariant to the affine part, and the affine part is
where nearly all of the gain information lives. That is a structural reason,
like the one that sank the Fourier-Mellin route (`scripts/spectral_pose.py`,
commit 7ee5473), not a tuning failure.

Runtime is not the obstacle either -- median coarse-stage cost 0.410 s (none),
0.324 s (anscombe), 0.494 s (gat) per pair over the same 480.

The module is kept, defaulted off (`config.SHIPPED_VST = "none"`, verified
bit-identical to the pre-#13 `pose_candidates` on real pairs), because the
negative result is the useful part and because a future stage that is NOT a
normalised correlation -- a learned proposer over a cost volume (#14), or an
absolute-likelihood verifier -- would not enjoy ZNCC's affine invariance and
could still have something to gain from a correct noise model.

Everything here is pure numpy. No new dependencies.
"""

from __future__ import annotations

import os

import numpy as np

MODES = ("none", "anscombe", "gat")

# Stages the transform can be switched on for, independently, so the A/B is
# decomposable. Each reads DRIFTSENSE_VST_<STAGE>, falling back to
# DRIFTSENSE_VST, falling back to the config default.
STAGES = ("coarse", "verify", "refine")

# Stages actually wired up in matching.py. "coarse" is the one the issue's
# mechanism points at -- `matching._band` locates the failure in the coarse
# sweep, not in verification.
#
# "verify" and "refine" are NOT wired, deliberately. Both act inside
# `locate_phase2` on `search_corr_std`, which is already mean-centred by
# `matching.standardize`; a square-root VST applied to a zero-mean array
# clamps its entire negative half and is simply wrong there. Wiring them
# correctly means moving the transform upstream of `standardize`, which
# changes a cached array shared by every hypothesis. That work is deferred
# until the coarse-stage result says the mechanism is real -- the issue
# predicts these two stages are a no-op or slightly negative anyway, because
# the sub-pixel residual is drift-limited rather than noise-limited
# (FAILURE_ANALYSIS.md sec. 1).
#
# Requesting a transform on an unwired stage RAISES rather than silently doing
# nothing: an experiment flag that quietly measures the baseline twice is the
# expensive kind of mistake, and this module exists to make that impossible.
WIRED_STAGES = ("coarse",)


def _default_mode() -> str:
    from driftsense.config import SHIPPED_VST
    return SHIPPED_VST


def resolve_mode(stage: str) -> str:
    """The VST mode in force for one pipeline stage.

    Precedence: DRIFTSENSE_VST_<STAGE>, then DRIFTSENSE_VST, then
    driftsense.config.SHIPPED_VST. An unrecognised value raises rather than
    silently running the shipped path -- a typo in an experiment flag that
    quietly measures the baseline twice is the expensive kind of mistake.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown VST stage {stage!r}; expected one of {STAGES}")
    mode = os.environ.get(f"DRIFTSENSE_VST_{stage.upper()}")
    if mode is None:
        mode = os.environ.get("DRIFTSENSE_VST")
    if mode is None:
        mode = _default_mode()
    mode = mode.strip().lower()
    if mode not in MODES:
        raise ValueError(f"unknown VST mode {mode!r}; expected one of {MODES}")
    if mode != "none" and stage not in WIRED_STAGES:
        raise NotImplementedError(
            f"VST stage {stage!r} is not wired up (see vst.WIRED_STAGES); "
            f"asked for {mode!r}. Only {WIRED_STAGES} take effect today, so "
            f"honouring this silently would report a baseline run as a treated "
            f"one.")
    return mode


def anscombe(img: np.ndarray) -> np.ndarray:
    """Anscombe (1948): 2*sqrt(x + 3/8). Poisson -> ~unit-variance Gaussian.

    Negative inputs are clamped at zero; the transform is undefined below
    -3/8 and our intensities are non-negative by construction, so a negative
    value means an upstream float error rather than data.
    """
    x = np.asarray(img, dtype=np.float32)
    return (2.0 * np.sqrt(np.maximum(x, 0.0) + 0.375)).astype(np.float32)


def generalized_anscombe(img: np.ndarray, a: float, b: float) -> np.ndarray:
    """GAT for the Poissonian-Gaussian model var(x) = a*E[x] + b (Foi 2008).

        f(x) = (2/a) * sqrt(a*x + 3*a^2/8 + b)

    `a` is the gain (DN per photoelectron) and `b` the additive read-noise
    variance in DN^2. As a -> 0 the model becomes purely Gaussian and the
    transform degenerates towards the identity, so a floor is applied.
    """
    x = np.asarray(img, dtype=np.float32)
    a = float(max(a, 1e-3))
    b = float(max(b, 0.0))
    rad = a * np.maximum(x, 0.0) + (3.0 / 8.0) * a * a + b
    return ((2.0 / a) * np.sqrt(np.maximum(rad, 0.0))).astype(np.float32)


def _block_stats(img: np.ndarray, block: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Per-block (mean, robust noise variance).

    The variance is estimated from first differences rather than from the block
    itself, because a block of layout is mostly *structure* and its raw variance
    would measure the pattern, not the noise. Differencing removes anything
    locally smooth; the MAD then suppresses the edges that survive it. Both
    differencing directions are computed and the smaller estimate kept -- an
    edge is rarely equally strong along both axes, and the noise is i.i.d., so
    the minimum is the less structure-contaminated of two valid estimates.
    """
    x = np.asarray(img, dtype=np.float32)
    h, w = x.shape
    nb_h, nb_w = h // block, w // block
    if nb_h < 2 or nb_w < 2:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    x = x[:nb_h * block, :nb_w * block]
    tiles = x.reshape(nb_h, block, nb_w, block).transpose(0, 2, 1, 3)

    means = tiles.mean(axis=(2, 3))
    # MAD of first differences -> sigma. The sqrt(2) removes the variance
    # doubling from differencing two independent samples; 0.6745 is the
    # standard normal MAD-to-sigma constant.
    out = []
    for axis in (2, 3):
        d = np.diff(tiles, axis=axis)
        med = np.median(d, axis=(2, 3), keepdims=True)
        mad = np.median(np.abs(d - med), axis=(2, 3))
        out.append((mad / (0.6745 * np.sqrt(2.0))) ** 2)
    var = np.minimum(out[0], out[1])
    return means.ravel().astype(np.float32), var.ravel().astype(np.float32)


def estimate_noise_model(img: np.ndarray, block: int = 16, bins: int = 24,
                         envelope_q: float = 0.25) -> tuple[float, float]:
    """Fit var = a*mean + b from one image (Foi 2008, local mean/variance).

    Structure inflates a block's variance estimate but can never deflate it, so
    the *lower envelope* of the (mean, variance) cloud is the noise-only curve.
    Blocks are binned by mean and a low quantile taken per bin, then a
    least-squares line is fitted through the bin points.

    Returns (a, b), clamped to a sane range. On a degenerate fit -- too few
    populated bins, or a non-positive slope -- returns (0.0, 0.0), which
    callers should read as "no usable model, fall back to plain Anscombe".
    """
    means, var = _block_stats(img, block=block)
    if means.size < 4 * bins:
        return 0.0, 0.0

    lo, hi = float(means.min()), float(means.max())
    if hi - lo < 1e-3:
        return 0.0, 0.0
    edges = np.linspace(lo, hi, bins + 1)
    idx = np.clip(np.digitize(means, edges) - 1, 0, bins - 1)

    xs, ys = [], []
    for k in range(bins):
        sel = var[idx == k]
        if sel.size < 8:          # too few blocks to trust a quantile
            continue
        xs.append(0.5 * (edges[k] + edges[k + 1]))
        ys.append(float(np.quantile(sel, envelope_q)))
    if len(xs) < 4:
        return 0.0, 0.0

    a, b = np.polyfit(np.asarray(xs, np.float64), np.asarray(ys, np.float64), 1)
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0.0:
        return 0.0, 0.0
    return float(a), float(max(b, 0.0))


def apply(img: np.ndarray, mode: str) -> np.ndarray:
    """Transform one image under `mode`. "none" returns the input unchanged.

    The output is float32 in a different numeric range from the input; every
    downstream consumer here is ZNCC or a band-pass, both of which are invariant
    to affine intensity changes, so the range shift is immaterial -- but the
    array is deliberately NOT rescaled back to 0-255, because doing so would
    reintroduce a scale the transform exists to fix.
    """
    if mode == "none":
        return img
    if mode == "anscombe":
        return anscombe(img)
    if mode == "gat":
        a, b = estimate_noise_model(img)
        if a <= 0.0:
            return anscombe(img)
        return generalized_anscombe(img, a, b)
    raise ValueError(f"unknown VST mode {mode!r}; expected one of {MODES}")


def apply_pair(reference: np.ndarray, search: np.ndarray,
               stage: str) -> tuple[np.ndarray, np.ndarray]:
    """Transform both images of a pair under the mode in force for `stage`.

    Both sides must be transformed, and with the same mode: ZNCC is invariant to
    an affine intensity relation between template and window, and a VST applied
    to only one side destroys that relation outright. Under `gat` the noise
    model is fitted per image, which is intended -- reference and search are
    separate acquisitions at different dose (`dose_reference` vs `dose_search`
    in the generator manifest), so they genuinely have different (a, b).
    """
    mode = resolve_mode(stage)
    if mode == "none":
        return reference, search
    return apply(reference, mode), apply(search, mode)

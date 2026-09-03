"""Post-hoc confidence calibration for the Phase-2 present/absent decision.

Phase-2's Calibration component (10 pts) is the AUC of our reported `score`
against per-pair correctness. The shipped confidence is the scalar
`min(score, zncc)` (driftsense.config.SHIPPED_THRESHOLD = 0.18 gates on it).

Why a feature-vector logistic and not temperature/Platt/isotonic scaling:
monotone maps of a scalar are rank-preserving, and AUC is a rank statistic,
so *no* monotone recalibration of the shipped scalar can move Calibration AUC
(Guo et al., "On Calibration of Modern Neural Networks", arXiv:1706.04599,
Sec. 2 — the calibration maps considered are all monotone and the paper's
metrics separate ranking from calibration accordingly). Any AUC gain must
come from recombining signals the decoder already emits. This module does
exactly that: a 9-feature logistic over the outputs `locate_phase2` already
produces. Zero inference cost — no second pass, no image access.

Feature semantics (see scripts/fit_rejector.py for the full argument):
  score       network peak confidence (relative — can be confident on a decoy)
  zncc        full-resolution correlation at the chosen centre (absolute)
  peak_ratio  runner-up / winner among separated peaks (contested decision)
  pose_peak   coarse correlation of the winning pose hypothesis
  psr         peak-to-sidelobe ratio (Bolme et al., MOSSE, CVPR 2010)
  apce        average peak-to-correlation energy
  rank        normalised winner strength vs the hypothesis pool
  band        response-band quality at the winner
  margin      winner's min(score, zncc) margin over the runner-up — the
              closest existing analogue of the correlogram r_delta of
              Buniatyan et al. (arXiv:1705.08593); see
              .agents/B_CALIBRATION_REPORT.md for the true r_delta spec.

NOTE (integrator handoff): rank and band are NOT available at inference on
the SHIPPED default decode path (verification="zncc", no
return_hypotheses — matching.py computes them only when verification !=
"zncc" or return_hypotheses=True, lines 874-886), so this 9-feature artefact
cannot be wired to the default path as-is. The SHIPPABLE feature sets are
subsets of {score, zncc, peak_ratio, pose_peak, psr, apce, margin}; see
scripts/fit_calibration.py --feature-set {6,7m,9} and the "shippable sets"
section of .agents/B_CALIBRATION_REPORT.md for the measured comparison and
the winning constants the integrator should use for the shipped module.

COEFS/INTERCEPT below are FROZEN constants fit offline by
scripts/fit_calibration.py on the FULL 2,250-pair holdout
(.agents/ext_features_full.csv) AFTER the 4-fold CV numbers were recorded —
the CV tables are the honest evidence; these constants are the shipped
artefact refit on all of it. Do not refit them in place: an accidental refit
fails tests/test_calibration.py::test_frozen_coefficients loudly.
Public API:
  FEATURES    frozen ordered feature list
  COEFS       frozen per-feature raw-scale coefficients
  INTERCEPT   frozen intercept
  calibrate(features: dict) -> float in [0, 1]   (P(present))
  fit(X, y)   offline helper: (w, mu, sd) in the fit_rejector.py convention

Only numpy + stdlib. No sklearn.
"""
from __future__ import annotations

import math

import numpy as np

# Frozen feature list — the 9-feature artefact (6 shipped + issue-#6
# rank/band/margin). Order matters for the arrays in scripts/fit_calibration
# diagnostics; calibrate() itself is dict-keyed so ordering cannot bite.
# See the integrator note above: rank/band are unavailable on the shipped
# default decode path; shippable subsets are compared in
# scripts/fit_calibration.py --feature-set and B_CALIBRATION_REPORT.md.
FEATURES = ["score", "zncc", "peak_ratio", "pose_peak",
            "psr", "apce", "rank", "band", "margin"]

# Frozen constants: GD(l2=1e-3) fit on the FULL 2,250-pair holdout, post-CV,
# emitted verbatim by `scripts/fit_calibration.py --freeze` (self-check
# max |P_std - P_raw| = 5.0e-16). Note on signs: raw-space coefficients mix
# signs (pose_peak, margin negative) even though every feature correlates
# positively with presence alone — the linear model is a joint ranker over
# correlated features and the held-out CV AUC (0.9915) is the property being
# shipped, not any single coefficient's sign.
_FROZEN_CONSTANTS_PLACEHOLDER = False
COEFS = {
    "score":       10.60423859851734,
    "zncc":        2.6805808255644283,
    "peak_ratio":  -0.4487484332316817,
    "pose_peak":   -6.709839254143747,
    "psr":         0.0014481513887072556,
    "apce":        3.976937953099698e-05,
    "rank":        3.240837308768774,
    "band":        0.8526308159408791,
    "margin":      -2.3637465613832664,
}
INTERCEPT = -0.7961563537640459


def _sigmoid(z):
    # Overflow-safe: exp(-|z|) never blows up; sign fold handles z<0.
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def calibrate(features: dict) -> float:
    """Calibrated P(present) from the decoder's own outputs.

    `features` must carry every name in FEATURES (extra keys are ignored).
    Pure arithmetic on existing inference outputs — zero inference cost.
    """
    z = INTERCEPT
    for f in FEATURES:
        z += COEFS[f] * float(features[f])
    return float(_sigmoid(np.asarray(z)))


# --- offline helpers (not used at inference time) ---------------------------

def standardize(X):
    """(mu, sd) and the standardised matrix, fit_rejector.py convention."""
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd, mu, sd


def design(Z):
    return np.hstack([Z, np.ones((len(Z), 1))])


def fit(X, y, iters=4000, lr=0.5, l2=1e-3, seed=0):
    """Plain fixed-seed gradient-descent logistic regression.

    Identical optimiser and hyperparameters to scripts/fit_rejector.py
    (iters=4000, lr=0.5, l2=1e-3, standardised features, intercept unregularised)
    so numbers stay comparable across the campaign. Convergence is *documented,
    not assumed*: at these hyperparameters the gradient norm falls >300x from
    its initial value on the real 2,250-pair holdout, and the parameter drift
    over the last 1,000 iterations is < 1e-3 (printed by
    scripts/fit_calibration.py --convergence-check). Deterministic: no
    initialisation randomness (w starts at 0), `seed` exists only for API
    stability.
    """
    Z, mu, sd = standardize(X)
    D = design(Z)
    y = np.asarray(y, float)
    w = np.zeros(D.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-D @ w))
        g = D.T @ (p - y) / len(y)
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    return w, mu, sd


def fit_irls(X, y, ridge=1e-6, iters=100, tol=1e-10):
    """Newton/IRLS logistic fit — quadratic convergence, used by
    scripts/fit_calibration.py for the frozen-constant artifacts and the
    convergence diagnostics (deterministic, seed-free). Returns (w, mu, sd)
    in the same convention as fit(): w[:-1] on standardised features, w[-1]
    the intercept."""
    Z, mu, sd = standardize(X)
    D = design(Z)
    y = np.asarray(y, float)
    w = np.zeros(D.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-D @ w))
        W = np.clip(p * (1 - p), 1e-10, None)
        g = D.T @ (y - p) - ridge * np.r_[w[:-1], 0.0]
        H = (D * W[:, None]).T @ D + ridge * np.diag(np.r_[np.ones(D.shape[1] - 1), 0.0])
        step = np.linalg.solve(H + 1e-12 * np.eye(len(w)), g)
        w = w + step
        if np.max(np.abs(step)) < tol:
            break
    return w, mu, sd


# --- SHIPPED inference statistic (integrator, 2026-09-03) -------------------
#
# The 9-feature calibrate() above is the OFFLINE ARTEFACT: rank/band are not
# computable on the shipped default decode path (they require the research
# verification feature maps), so it cannot run at inference. The SHIPPED
# statistic is the 6-feature logistic below -- the winner of the shippable-set
# comparison (held-out 4-fold CV AUC: 6 -> 0.9915, 6+margin -> 0.9907,
# 9 -> 0.9911-not-shippable; .agents/B_CALIBRATION_REPORT.md ADDENDUM 2,
# protocol identical to REJECTOR_FINDINGS.md). Its features are exactly what
# locate() already returns, so it costs nothing at inference.
#
# Constants are FROZEN: GD(l2=1e-3, iters=4000, lr=0.5) -- the CV protocol's
# optimiser -- refit on the FULL 2,250-pair holdout after CV was recorded.
# Reproduce with: venv313/bin/python scripts/fit_calibration.py --feature-set 6
# Conversion self-check max |P_std - P_raw| = 3.89e-16.
SHIPPED_FEATURES = ["score", "zncc", "peak_ratio", "pose_peak", "psr", "apce"]
SHIPPED_COEFS = {
    "score":       8.792353455411558,
    "zncc":        4.771002619826103,
    "peak_ratio":  -0.22820720932034558,
    "pose_peak":   -6.636234556838288,
    "psr":         0.001277399759015737,
    "apce":        3.981647734693036e-05,
}
SHIPPED_INTERCEPT = -0.6718057933029007


def calibrate_shipped(features: dict) -> float:
    """Shipped P(present) from the six statistics locate() already computes.

    NOT the shipped statistic. `SHIPPED_CONFIDENCE` is "legacy_min" and
    `SHIPPED_THRESHOLD` is 0.18; this logistic is retained behind that constant
    after measuring out (-0.43 on an untouched holdout, P(better) = 0.011 --
    see driftsense/config.py). If it is re-enabled it must be paired with its
    own threshold of 0.4870, because the statistic and the threshold are one
    unit system.

    `features` must carry every name in SHIPPED_FEATURES (extra keys are
    ignored).

    **Non-finite features are imputed to 0.0, matching the fit exactly.**
    `locate_phase2` deliberately sets `pose_peak = NaN` for explicit-pose
    candidates (`pose=` supplied, matching.py:1187) and for rescue-generated
    candidates (matching.py:1244), because neither came from a coarse pose
    sweep and so has no sweep peak to report. Without this guard `z` becomes
    NaN, the sigmoid returns NaN, and `NaN >= threshold` evaluates False --
    silently forcing `found=0` on a pair the decode may have located perfectly.

    Imputing 0.0 is not a guess: `scripts/fit_calibration.py` builds every
    design matrix with `np.nan_to_num(..., nan=0.0)` (lines 132, 178, 197, 268,
    325), so a NaN feature contributed exactly 0.0 during fitting too. The
    inference path now reproduces the training preprocessing rather than
    diverging from it.

    Infinities are also mapped to 0.0. That is a deliberate *divergence*: the
    fit's `nan_to_num` would have mapped +/-inf to +/-1.8e308 and saturated the
    logistic, but no feature in the fit data was ever infinite, so there is no
    trained behaviour to preserve -- and saturating to a confident PRESENT on a
    corrupt statistic is the worst available failure.
    """
    z = SHIPPED_INTERCEPT
    for f in SHIPPED_FEATURES:
        v = float(features[f])
        z += SHIPPED_COEFS[f] * (v if math.isfinite(v) else 0.0)
    out = float(_sigmoid(np.asarray(z)))
    # A non-finite z can only come from a non-finite coefficient or intercept,
    # which would be a corrupted constant table; fail closed rather than emit
    # NaN into the found decision.
    return out if math.isfinite(out) else 0.0

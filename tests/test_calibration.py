"""Workstream B: tests for driftsense/calibration.py.

The module maps the nine inference outputs the pipeline already records onto a
calibrated P(present) via a frozen logistic. Constraint from Guo et al.
(arXiv:1706.04599): no monotone recalibration of a *scalar* can move AUC, so
the value has to come from the feature vector — hence a multivariate logistic,
not temperature/isotonic scaling of the shipped confidence.

Tests pin behaviour, not numbers, except `test_frozen_coefficients`, which
exists to make an accidental refit of the shipped constants fail loudly: the
pinned literals below must equal the constants in calibration.py, and any
change to either is a deliberate, reviewable act.
"""
import numpy as np
import pytest

from driftsense.calibration import COEFS, FEATURES, INTERCEPT, calibrate, fit

# Plausible extremes from the recorded ranges in .agents/ext_features_full.csv
# (2250-pair holdout). "present-like" = what a confident correct lock-on looks
# like; "absent-like" = a Set-C style miss. Only the six inference-available
# features (shipped default path) appear in the frozen list; margin/rank/band
# keys below are ignored by calibrate() — kept to prove extra keys are legal.
PRESENT_LIKE = {
    "score": 0.90, "zncc": 0.90, "peak_ratio": 0.05, "pose_peak": 0.90,
    "psr": 2000.0, "apce": 40000.0, "rank": 0.60, "band": 0.95, "margin": 0.80,
    "score_zncc_min": 0.90, "score_zncc_prod": 0.81, "score_zncc_gap": 0.0,
    "zncc_over_score": 1.0, "peak_pose_prod": 0.81, "n_hyp": 3.0,
}
ABSENT_LIKE = {
    "score": 0.05, "zncc": 0.02, "peak_ratio": 0.95, "pose_peak": 0.15,
    "psr": 20.0, "apce": 100.0, "rank": 0.01, "band": 0.05, "margin": 0.0,
    "score_zncc_min": 0.02, "score_zncc_prod": 0.001, "score_zncc_gap": 0.03,
    "zncc_over_score": 0.4, "peak_pose_prod": 0.015, "n_hyp": 1.0,
}


def _row(base, **over):
    row = {f: base[f] for f in FEATURES}
    row.update(over)
    return row


def test_features_are_frozen_and_complete():
    """The 9-feature artefact's feature list is a frozen constant, and every
    coefficient names a feature in it. (The shipped module the integrator
    writes will use a shippable subset — see
    scripts/fit_calibration.py --feature-set and B_CALIBRATION_REPORT.md;
    this pin guards the artefact as-is.)"""
    assert FEATURES == ["score", "zncc", "peak_ratio", "pose_peak",
                        "psr", "apce", "rank", "band", "margin"]
    assert set(COEFS) == set(FEATURES)


def test_output_in_unit_interval():
    """calibrate() returns a probability: [0, 1] on the plausible extremes and
    on a dense random sweep between them."""
    rng = np.random.RandomState(0)
    for _ in range(500):
        row = {f: float(rng.uniform(ABSENT_LIKE[f], PRESENT_LIKE[f]))
               for f in FEATURES}
        p = calibrate(row)
        assert 0.0 <= p <= 1.0, (row, p)
    assert 0.0 <= calibrate(_row(ABSENT_LIKE)) <= 1.0
    assert 0.0 <= calibrate(_row(PRESENT_LIKE)) <= 1.0


def test_present_like_above_absent_like():
    """The whole point: the shipped statistic must rank a confident correct
    lock-on above a Set-C style miss."""
    assert calibrate(_row(PRESENT_LIKE)) > calibrate(_row(ABSENT_LIKE))


def test_monotone_in_score():
    """Holding every other feature fixed, raising `score` can never lower the
    calibrated confidence (positive weight on the raw score)."""
    base = {f: (PRESENT_LIKE[f] + ABSENT_LIKE[f]) / 2 for f in FEATURES}
    ps = [calibrate(_row(base, score=s)) for s in np.linspace(0.0, 0.95, 40)]
    assert all(b >= a - 1e-12 for a, b in zip(ps, ps[1:])), ps


def test_matches_direct_logistic_apply():
    """calibrate() is exactly sigmoid(sum c_f * f + b) — no hidden transform."""
    row = _row(PRESENT_LIKE, score=0.42)
    z = sum(COEFS[f] * row[f] for f in FEATURES) + INTERCEPT
    assert calibrate(row) == pytest.approx(1.0 / (1.0 + np.exp(-z)), abs=1e-12)


def test_missing_feature_raises():
    with pytest.raises(KeyError):
        calibrate({f: 0.0 for f in FEATURES[:-1]})


def test_frozen_coefficients():
    """Accidental refits fail loudly. The literals below are the shipped
    constants recorded in .agents/B_CALIBRATION_REPORT.md (fit on the FULL
    2,250-pair holdout AFTER CV, via scripts/fit_calibration.py --freeze).
    Changing COEFS/INTERCEPT in calibration.py without updating this pinned
    copy — or vice versa — fails this test, so any refit is a deliberate,
    reviewable act."""
    import driftsense.calibration as _cal
    if getattr(_cal, "_FROZEN_CONSTANTS_PLACEHOLDER", True):
        pytest.fail("calibration.py still carries placeholder constants: "
                    "run scripts/fit_calibration.py --freeze and paste the "
                    "real fitted values before shipping.")
    # Pinned copy of the constants in driftsense/calibration.py. Written by
    # scripts/fit_calibration.py --freeze (full-2250 GD(l2=1e-3) fit, post-CV,
    # 9-feature artefact); see B_CALIBRATION_REPORT.md for provenance. The
    # integrator writes the shipped module from the WINNING shippable-set
    # constants in the report; this pin guards the artefact as-is.
    expected_coefs = {
        "score":       +10.60423859851734,
        "zncc":        +2.6805808255644283,
        "peak_ratio":  -0.4487484332316817,
        "pose_peak":   -6.709839254143747,
        "psr":         +0.0014481513887072556,
        "apce":        +3.976937953099698e-05,
        "rank":        +3.240837308768774,
        "band":        +0.8526308159408791,
        "margin":      -2.3637465613832664,
    }
    expected_intercept = -0.7961563537640459
    for f in FEATURES:
        assert COEFS[f] == pytest.approx(expected_coefs[f], abs=1e-9), f
    assert INTERCEPT == pytest.approx(expected_intercept, abs=1e-9)


def test_fit_deterministic():
    """Fixed-seed gradient descent: the same data fits to the same constants,
    bit-for-bit, twice."""
    rng = np.random.RandomState(1)
    X = rng.rand(200, len(FEATURES))
    X[:, :6] = X[:, :6] * [0.9, 0.9, 1.0, 0.9, 4000, 50000]  # plausible scales
    y = (X[:, 0] + X[:, 1] > 0.9).astype(float)
    a = fit(X, y)
    b = fit(X, y)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_fit_recovers_separable_signal():
    """The fitter actually learns: on data that is linearly separable, held-out
    predictions order the classes correctly."""
    rng = np.random.RandomState(2)
    n = 400
    X = np.column_stack([
        np.r_[rng.rand(n // 2) * 0.2 + 0.7, rng.rand(n // 2) * 0.2],        # score
        np.r_[rng.rand(n // 2) * 0.2 + 0.7, rng.rand(n // 2) * 0.2],        # zncc
        rng.rand(n), rng.rand(n), rng.rand(n) * 4000, rng.rand(n) * 50000,
    ])
    y = (X[:, 0] > 0.5).astype(float)
    idx = np.random.RandomState(3).permutation(n)   # classes are block-ordered
    tr, te = idx[:300], idx[300:]
    w, mu, sd = fit(X[tr], y[tr])
    z = (X[te] - mu) / sd
    p = 1.0 / (1.0 + np.exp(-(np.hstack([z, np.ones((100, 1))]) @ w)))
    assert p[y[te] == 1].mean() > p[y[te] == 0].mean() + 0.3

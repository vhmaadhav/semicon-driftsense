"""Opt-in offline confidence experiment; not imported by the submission path.

Seven decoder signals plus explicit missing-margin availability. The fitted
standardisation is frozen with the head. This is a research candidate, not the
shipped confidence policy, and requires new-domain confirmation before adoption.
"""
from __future__ import annotations

import hashlib
import numpy as np

from driftsense.calibration import fit, _sigmoid

BASE_FEATURES = ('net_score', 'zncc', 'peak_ratio', 'pose_peak', 'psr', 'apce')
FEATURES = (*BASE_FEATURES, 'winner_margin', 'margin_missing')
VERSION = 'log-peak-shape-margin-v1'


def feature_matrix(frame):
    """Keep an unavailable margin distinct from a measured zero-margin tie."""
    missing = set((*BASE_FEATURES, 'winner_margin')) - set(frame.columns)
    if missing:
        raise ValueError(f'missing decoder features: {sorted(missing)}')
    x = frame.loc[:, list(BASE_FEATURES)].to_numpy(dtype=float, copy=True)
    if not np.isfinite(x).all():
        raise ValueError('base decoder features must all be finite')
    if (x[:, 4:6] < 0).any():
        raise ValueError('PSR and APCE must be nonnegative for log1p')
    x[:, 4:6] = np.log1p(x[:, 4:6])
    margin = frame['winner_margin'].to_numpy(dtype=float)
    absent = ~np.isfinite(margin)
    return np.column_stack((x, np.where(absent, 0., margin), absent.astype(float)))


def partition(groups, seed):
    """Stable ~60/20/20 fit/tune/assessment assignment per source group."""
    if groups.isna().any():
        raise ValueError('source groups must be present for every pair')
    result = []
    for group in groups:
        key = f'confidence85-v1:{seed}:{group}'.encode('utf-8')
        u = int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / 2**64
        result.append('fit' if u < .6 else 'tune' if u < .8 else 'assess')
    return np.asarray(result)


def fit_head(frame, present):
    """Only pass the fit partition; all hyperparameters are predeclared."""
    x = feature_matrix(frame)
    y = np.asarray(present, dtype=float)
    if y.shape != (len(x),) or set(np.unique(y)) != {0., 1.}:
        raise ValueError('fit labels must contain both presence classes')
    w, mu, sd = fit(x, y, iters=4000, lr=.5, l2=.001)
    return dict(version=VERSION, features=list(FEATURES), coefficients=w.tolist(),
                mean=mu.tolist(), std=sd.tolist(), target='presence',
                fit_rows=len(x), optimizer=dict(iterations=4000, lr=.5, l2=.001))


def predict(frame, model):
    """Label-free scores; a batch never changes stored fit statistics."""
    if model['version'] != VERSION or model['features'] != list(FEATURES):
        raise ValueError('incompatible confidence experiment artifact')
    x = feature_matrix(frame)
    z = (x - np.asarray(model['mean'])) / np.asarray(model['std'])
    w = np.asarray(model['coefficients'])
    return _sigmoid(z @ w[:-1] + w[-1])

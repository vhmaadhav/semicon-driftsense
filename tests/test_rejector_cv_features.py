"""Codex review round 2, item 1: eval_ext ALWAYS emits rank/band columns --
on a run without --features they are all-NaN. Feature availability must
require usable finite data, otherwise rejector_cv runs its extended trials
on features that were never computed and prints plausible-looking results.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

pd = pytest.importorskip("pandas")


def _load():
    path = os.path.join(REPO_ROOT, "scripts", "rejector_cv.py")
    spec = importlib.util.spec_from_file_location("rejector_cv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_all_nan_columns_are_not_available():
    """The regression: a default eval CSV (no --features) has rank/band
    columns that are 100% NaN. They must be reported missing, not fitted on
    after nan_to_num fills zeros."""
    mod = _load()
    df = pd.DataFrame({
        "rank": [np.nan] * 10,
        "band": [np.nan] * 10,
        "margin": np.linspace(0.0, 1.0, 10),
    })
    ok, missing = mod.available(df, ["rank", "band", "margin"])
    assert ok == ["margin"]
    assert set(missing) == {"rank", "band"}


def test_usably_finite_columns_are_available():
    mod = _load()
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "rank": rng.random(10),                     # fully finite
        "band": [np.nan] + list(rng.random(9)),     # 90% finite (realistic)
        "margin": [np.nan] * 9 + [0.5],             # 10% finite: unusable
    })
    ok, missing = mod.available(df, ["rank", "band", "margin"])
    assert set(ok) == {"rank", "band"}
    assert missing == ["margin"]


def test_absent_columns_still_reported_missing():
    mod = _load()
    df = pd.DataFrame({"score": [0.5, 0.6]})
    ok, missing = mod.available(df, ["score", "rank"])
    assert ok == ["score"]
    assert missing == ["rank"]

"""eval_ext --sample N: emulate the organisers' 200-pair blind grade.

The grading set is 200 pairs drawn from the organisers' own distribution;
a random sample of ext_p2 scored under the same rubric is the closest
local emulation of it (PHASE2_STATE's noise-floor table was built the same
way). The sampler must be deterministic per seed and never repeat a pair.
"""

import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

pd = pytest.importorskip("pandas")


def _load():
    path = os.path.join(REPO_ROOT, "scripts", "eval_ext.py")
    spec = importlib.util.spec_from_file_location("eval_ext", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_sample_is_exact_size_and_without_replacement():
    mod = _load()
    df = pd.DataFrame({"pair_id": [f"p{i}" for i in range(500)]})
    out = mod.sample_pairs(df, 200, seed=7)
    assert len(out) == 200
    assert out["pair_id"].nunique() == 200


def test_sample_is_deterministic_per_seed_and_varies_across_seeds():
    mod = _load()
    df = pd.DataFrame({"pair_id": [f"p{i}" for i in range(500)]})
    a1 = mod.sample_pairs(df, 50, seed=3).pair_id.tolist()
    a2 = mod.sample_pairs(df, 50, seed=3).pair_id.tolist()
    b = mod.sample_pairs(df, 50, seed=4).pair_id.tolist()
    assert a1 == a2
    assert a1 != b


def test_sample_larger_than_frame_returns_everything():
    mod = _load()
    df = pd.DataFrame({"pair_id": [f"p{i}" for i in range(30)]})
    out = mod.sample_pairs(df, 200, seed=0)
    assert len(out) == 30

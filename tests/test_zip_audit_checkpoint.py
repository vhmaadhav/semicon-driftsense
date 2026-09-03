"""Regression tests for the checkpoint instantiation check in
scripts/check_submission_zip.py (Phase 2 review fix, r2).

The audit's weights check must go beyond torch.load + key-presence: from
the extracted artifact it must exercise the REAL ship loader
(import infer; infer.load_model(<artifact>/weights/driftsense.pt)) and
require a non-None model. A checkpoint that loads as pickle but cannot
instantiate (shape/arch mismatch) must FAIL the audit.

Two tiny artifacts are built in tmp_path:

* one with a deliberately shape-incompatible checkpoint (a real small
  state_dict from the shipped checkpoint with one tensor's shape corrupted,
  saved as a plain dict of tensors -- weights_only-safe) -> the audit's
  weights check must FAIL;
* one with the real repo checkpoint wired in -> the weights check must PASS
  with a non-None model.

The tests are gated on torch AND a torch-capable interpreter (the same
probe the checker uses); without one the instantiation check is a required
SKIP in artifact mode and there is nothing to assert here.
"""

import importlib.util
import os
import shutil
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")          # infer.py imports cv2 at module import

REAL_WEIGHTS = os.path.join(REPO_ROOT, "weights", "driftsense.pt")


def _load_checker():
    path = os.path.join(REPO_ROOT, "scripts", "check_submission_zip.py")
    spec = importlib.util.spec_from_file_location("check_submission_zip_test",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


@pytest.fixture(scope="module")
def interpreter(checker):
    """The checker's own torch-capable interpreter probe must succeed."""
    interp = checker.torch_interpreter()
    if interp is None:
        pytest.skip("no torch-capable interpreter for the checker's "
                    "ship-loader subprocess")
    return interp


def _make_artifact(tmp_path, weights_source):
    """Minimal extractable artifact root: infer.py + driftsense package +
    weights/driftsense.pt. Returns the root path as str."""
    root = tmp_path / "artifact"
    root.mkdir()
    shutil.copy(os.path.join(REPO_ROOT, "infer.py"), root / "infer.py")
    shutil.copytree(
        os.path.join(REPO_ROOT, "driftsense"), root / "driftsense",
        ignore=shutil.ignore_patterns("__pycache__"))
    wdir = root / "weights"
    wdir.mkdir()
    shutil.copy(weights_source, wdir / "driftsense.pt")
    return str(root)


def _corrupted_checkpoint(tmp_path):
    """A plain-dict-of-tensors checkpoint (weights_only-safe) whose 'model'
    state_dict has one tensor's shape deliberately corrupted, so torch.load
    succeeds but DriftSenseNet.load_state_dict cannot."""
    if not os.path.isfile(REAL_WEIGHTS):
        pytest.skip("weights/driftsense.pt missing from the repo")
    ckpt = torch.load(REAL_WEIGHTS, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    assert isinstance(state, dict) and state, "unexpected checkpoint layout"
    key = sorted(state.keys())[0]
    tensor = state[key]
    shape = list(tensor.shape)
    # Guarantee a DIFFERENT shape regardless of the original dims.
    shape[0] = shape[0] + 1 if shape[0] <= 1 else max(1, shape[0] // 2)
    state[key] = torch.zeros(shape, dtype=tensor.dtype)
    out = tmp_path / "corrupted.pt"
    torch.save({"model": state}, str(out))   # plain dict of tensors
    return out


def _instantiation_check(new_checks):
    for c in new_checks:
        if c["name"].startswith("checkpoint instantiates via the ship loader"):
            return c
    raise AssertionError(
        "audit did not run the ship-loader instantiation check; got: "
        + repr([(c["name"], c["ok"], c["skipped"]) for c in new_checks]))


def test_shape_incompatible_checkpoint_fails_audit(tmp_path, checker,
                                                   interpreter):
    """A checkpoint that torch.loads fine but cannot instantiate must FAIL
    the audit's weights check (not merely load as pickle)."""
    before = len(checker.checks)
    root = _make_artifact(tmp_path, _corrupted_checkpoint(tmp_path))
    checker.audit_weights_load(root)
    new_checks = checker.checks[before:]

    load_c = next(c for c in new_checks
                  if c["name"].startswith("checkpoint loads"))
    # The pickle-level load genuinely succeeds (plain dict, 'model' key):
    assert load_c["ok"] and not load_c["skipped"], load_c["detail"]

    inst = _instantiation_check(new_checks)
    assert not inst["skipped"], inst["detail"]
    assert not inst["ok"], (
        "shape-incompatible checkpoint PASSED the ship-loader "
        "instantiation check: " + inst["detail"])
    assert "exit" in inst["detail"] or "None" in inst["detail"], inst["detail"]


def test_real_checkpoint_passes_audit_with_non_none_model(tmp_path, checker,
                                                          interpreter):
    """The real repo checkpoint must PASS the ship-loader instantiation
    check with a non-None model."""
    if not os.path.isfile(REAL_WEIGHTS):
        pytest.skip("weights/driftsense.pt missing from the repo")
    before = len(checker.checks)
    root = _make_artifact(tmp_path, REAL_WEIGHTS)
    checker.audit_weights_load(root)
    new_checks = checker.checks[before:]

    load_c = next(c for c in new_checks
                  if c["name"].startswith("checkpoint loads"))
    assert load_c["ok"] and not load_c["skipped"], load_c["detail"]

    inst = _instantiation_check(new_checks)
    assert not inst["skipped"], inst["detail"]
    assert inst["ok"], inst["detail"]
    assert "DriftSenseNet" in inst["detail"], inst["detail"]

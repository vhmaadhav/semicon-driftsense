"""C-01 of the static audit: evaluation must measure the exact shipped
inference policy.

Before this fix, evaluate.run_split called locate/locate_tta directly, while
infer.predict ran choose_pose + adaptive routing first -- the reported
metrics did not measure the behaviour users receive. Both entry points must
go through one shared policy (driftsense.policy.predict_policy), and a
parity test must hold: identical arrays in -> identical coordinates, score,
route and method out.
"""

import csv
import importlib.util
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")

from driftsense.model import DriftSenseNet  # noqa: E402

WEIGHTS = os.path.join(REPO_ROOT, "weights", "driftsense.pt")


@pytest.fixture(scope="module")
def model():
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    net = DriftSenseNet()
    net.load_state_dict(ck.get("model", ck))
    net.eval()
    return net


def _scene(tmp_path, seed):
    """Deterministic synthetic (reference, search) pair: a textured patch
    embedded in a differently-textured search frame."""
    rng = np.random.default_rng(seed)
    reference = (rng.integers(0, 255, size=(100, 100), dtype=np.uint8))
    search = (rng.integers(0, 255, size=(200, 200), dtype=np.uint8))
    # Plant the reference (downsampled 2x, as the magnification implies) at a
    # known location with light noise so a matcher has something to find.
    tpl = cv2.resize(reference, (50, 50), interpolation=cv2.INTER_AREA)
    x0, y0 = 120, 40
    patch = search[y0:y0 + 50, x0:x0 + 50]
    search[y0:y0 + 50, x0:x0 + 50] = (0.5 * patch + 0.5 * tpl).astype(np.uint8)
    ref_path = str(tmp_path / f"ref_{seed}.png")
    sea_path = str(tmp_path / f"sea_{seed}.png")
    assert cv2.imwrite(ref_path, reference)
    assert cv2.imwrite(sea_path, search)
    return reference, search, ref_path, sea_path


def _policy():
    import driftsense.policy as policy
    return policy


@pytest.mark.parametrize("seed", [1, 2])
def test_infer_predict_and_policy_are_identical(tmp_path, model, seed, monkeypatch):
    import infer as infer_mod

    reference, search, ref_path, sea_path = _scene(tmp_path, seed)
    policy = _policy()

    # Pin the CLI path to the same CPU model instance: the parity contract is
    # about the decode policy, not device dispatch (MPS may route otherwise).
    monkeypatch.setattr(infer_mod, "load_model", lambda w: (model, "cpu"))

    res_cli = infer_mod.predict(ref_path, sea_path, WEIGHTS)
    res_policy = policy.predict_policy(model, reference, search, "cpu", tta=True)

    for key in ("x", "y", "score", "method", "routed"):
        assert res_cli[key] == pytest.approx(res_policy[key]) if key == "score" \
            else res_cli[key] == res_policy[key], key


def test_run_split_goes_through_the_shipped_policy(tmp_path, model, monkeypatch):
    import evaluate as evaluate_mod
    policy = _policy()

    reference, search, ref_path, sea_path = _scene(tmp_path, 7)
    assert cv2.imwrite(str(tmp_path / "ref_b.png"), reference)
    assert cv2.imwrite(str(tmp_path / "sea_b.png"), search)
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "reference_path", "search_path",
                                           "gt_x_corr", "gt_y_corr", "gt_x", "gt_y"])
        w.writeheader()
        w.writerow({"id": "a", "reference_path": f"ref_7.png",
                    "search_path": f"sea_7.png", "gt_x_corr": 145.0,
                    "gt_y_corr": 65.0, "gt_x": 145.0, "gt_y": 65.0})
        w.writerow({"id": "b", "reference_path": "ref_b.png",
                    "search_path": "sea_b.png", "gt_x_corr": 145.0,
                    "gt_y_corr": 65.0, "gt_x": 145.0, "gt_y": 65.0})

    calls = {"n": 0}
    real = policy.predict_policy

    def spy(*a, **kw):
        calls["n"] += 1
        return real(*a, **kw)

    monkeypatch.setattr(policy, "predict_policy", spy)
    # run_split must import the policy through its module so the spy applies.
    result = evaluate_mod.run_split(str(tmp_path), model, "cpu", do_baseline=False,
                                    tta=True)

    assert calls["n"] == 2, "run_split bypassed the shipped inference policy"
    assert result["routes"] and sum(result["routes"].values()) == 2


def test_run_split_fails_loudly_on_unreadable_image(tmp_path, model):
    import evaluate as evaluate_mod

    reference, search, ref_path, sea_path = _scene(tmp_path, 9)
    with open(tmp_path / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "reference_path", "search_path",
                                           "gt_x_corr", "gt_y_corr"])
        w.writeheader()
        w.writerow({"id": "a", "reference_path": "gone.png",
                    "search_path": "sea_9.png", "gt_x_corr": 145.0,
                    "gt_y_corr": 65.0})
    with pytest.raises(Exception, match="gone.png"):
        evaluate_mod.run_split(str(tmp_path), model, "cpu", do_baseline=False)

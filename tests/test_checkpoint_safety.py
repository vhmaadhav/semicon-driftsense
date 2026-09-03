"""C-03 of the static audit: checkpoints must not require unrestricted
pickle deserialization. Every tracked weight file must load with
``weights_only=True``, and no inference-path code may pass
``weights_only=False`` (train.py's ``--resume`` is the one explicitly
trusted training artifact and is exempt).
"""

import glob
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

WEIGHTS = sorted(glob.glob(os.path.join(REPO_ROOT, "weights", "*.pt")))


@pytest.mark.parametrize("path", WEIGHTS, ids=lambda p: os.path.basename(p))
def test_weight_files_load_with_weights_only_true(path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(ck, dict)
    assert "model" in ck or all(hasattr(v, "shape") for v in ck.values())


def test_no_unrestricted_pickle_loading_on_inference_paths():
    """``weights_only=False`` is allowed only in train.py --resume, which loads
    an explicitly trusted training-resume artifact."""
    bad = []
    for dirpath, dirnames, filenames in os.walk(REPO_ROOT):
        # Any virtualenv, not just the two that existed when this was written:
        # a second env (venv-train, venv-hf) puts torch's own source inside the
        # walk, and torch uses weights_only=False in ~40 places of its own.
        dirnames[:] = [d for d in dirnames
                       if d not in (".git", "__pycache__", "graphify-out",
                                    ".skill-port", ".dsh", ".sdd", ".venv")
                       and not d.startswith("venv")
                       and not os.path.exists(os.path.join(dirpath, d, "pyvenv.cfg"))]
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, REPO_ROOT)
            if rel == os.path.join("train.py"):
                continue
            if path == os.path.abspath(__file__):
                continue  # the scanner's own source quotes the pattern
            with open(path, encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, 1):
                    if "weights_only=False" in line:
                        bad.append(f"{rel}:{lineno}")
    assert bad == [], (
        "unrestricted torch.load on inference paths (audit C-03): "
        + ", ".join(bad))

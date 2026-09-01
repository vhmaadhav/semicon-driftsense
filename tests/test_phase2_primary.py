"""Repository-level contract: Phase 2 is the canonical product surface."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_phase2_entrypoint_and_shared_runtime_are_present():
    assert os.path.isfile(os.path.join(REPO_ROOT, "register.py"))
    assert os.path.isfile(os.path.join(REPO_ROOT, "driftsense", "runtime.py"))


def test_readme_points_new_users_to_register_not_legacy_infer():
    readme = open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8").read()
    assert "Phase 2 is the canonical project" in readme
    assert "python register.py --input pairs.csv --output predictions.csv" in readme
    assert "Legacy Phase 1 compatibility" in readme

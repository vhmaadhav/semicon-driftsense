"""The submission ZIP is constructed from a whitelist, not filtered from the tree.

`scripts/check_submission_zip.py` audits an artifact that already exists; it
cannot make a bad one good. These tests guard the builder's three gates, so a
future whitelist edit cannot quietly put internal material or a second
checkpoint into the artifact (PR #48 review, item 3).

They stage the file list only -- no ZIP is written, so the suite stays fast.
"""
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))

B = pytest.importorskip("build_submission_zip")


@pytest.fixture(scope="module")
def staged():
    return B.whitelist_paths(REPO, with_tests=False)


def test_required_organizer_layout_is_staged(staged):
    """Everything check_submission_zip.py demands at the extraction root."""
    from check_submission_zip import REQUIRED_PATHS
    missing = [p for p in REQUIRED_PATHS if p not in staged]
    assert not missing, f"whitelist would fail the artifact audit: {missing}"


def test_only_the_shipped_checkpoint_is_staged(staged):
    """weights/ holds 20 checkpoints; exactly one may ship."""
    checkpoints = [p for p in staged if p.endswith((".pt", ".pth"))]
    assert checkpoints == [B.ALLOWED_WEIGHT], checkpoints


def test_no_internal_material_is_staged(staged):
    """The denylist is the last gate, and it must accept the current whitelist."""
    B.check_denylist(staged)          # raises SystemExit(2) on a violation
    for rel in staged:
        posix = rel.replace(os.sep, "/")
        assert not posix.startswith(".agents/"), rel
        assert not posix.startswith("data/"), rel
        assert not posix.startswith("results/"), rel


def test_import_closure_is_covered(staged):
    """Every locally imported module -- including the sys.path-inserted
    generator `src.*` tree -- must already be in the whitelist, or the
    artifact breaks on the grader's box."""
    n = B.check_import_closure(REPO, staged)   # raises SystemExit(2) if not
    assert n > 0


def test_denylist_actually_rejects(staged):
    """A gate that never fires is not a gate."""
    with pytest.raises(SystemExit):
        B.check_denylist(staged + [os.path.join(".agents", "pr48_full.csv")])
    with pytest.raises(SystemExit):
        B.check_denylist(staged + [os.path.join("weights", "soup_all.pt")])

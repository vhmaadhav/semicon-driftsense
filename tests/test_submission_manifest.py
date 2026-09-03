"""Contract tests for scripts/build_submission_zip.py's manifest.

The ZIP this builder writes is the graded artifact, so the two ways it can go
wrong both have to fail loudly here rather than in a judge's extraction:

*   something confidential or irrelevant ships (.agents/, phase1/, the 15
    unused checkpoints), or
*   something required silently stops shipping because a file was renamed.

The third case is subtler and is what actually bit during the first build: a
shipped test importing a module that stays behind (tests/test_subpixel_drift.py
imports test_submission_parity, which loads scripts/eval_ext.py). That is a
green repo suite and a red extraction, so it is checked directly.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _builder():
    path = os.path.join(REPO, "scripts", "build_submission_zip.py")
    if not os.path.isfile(path):
        pytest.skip("scripts/build_submission_zip.py not present")
    spec = importlib.util.spec_from_file_location("build_submission_zip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder():
    return _builder()


@pytest.fixture(scope="module")
def manifest(builder):
    members, missing = builder.collect(REPO)
    assert not missing, "manifest names paths that do not exist: " + repr(missing)
    return [arcname for arcname, _ in members]


def test_every_allow_entry_resolves(builder):
    """A rename must break the build, not quietly hole the submission."""
    _, missing = builder.collect(REPO)
    assert missing == []


def test_no_denied_paths_in_manifest(builder, manifest):
    offenders = {name: builder.deny_reason(name) for name in manifest}
    offenders = {n: why for n, why in offenders.items() if why}
    assert offenders == {}


@pytest.mark.parametrize("forbidden", [".agents", "phase1", ".git", ".github",
                                       "scripts", "venv", "data", "results"])
def test_forbidden_top_level_directories_absent(manifest, forbidden):
    assert [n for n in manifest if n.split("/")[0] == forbidden] == []


def test_only_the_shipped_checkpoint_ships(builder, manifest):
    weights = [n for n in manifest if n.startswith("weights/")]
    assert weights == [builder.WEIGHTS_ONLY]


def test_organizer_required_paths_are_all_in_the_manifest(manifest):
    """Slide 5's required-ship list, spelled out independently of the builder."""
    required = ["register.py", "infer.py", "generate_dataset.py",
                "requirements.txt", "failure_analysis.pdf",
                "weights/driftsense.pt"]
    assert [p for p in required if p not in manifest] == []


def test_generator_deliverables_are_in_the_manifest(manifest):
    """DOCX section 7's generator deliverable set."""
    required = ["generator/generate_phase2.py", "generator/baseline.py",
                "generator/score.py", "generator/contact_sheet.py",
                "generator/REPORT.md", "generator/src/pipeline.py"]
    assert [p for p in required if p not in manifest] == []


def test_generator_output_package_is_in_the_manifest(manifest):
    """Section 7 names output/ itself, images included.

    The builder prunes scratch directories by name; output/ must survive that
    pruning or a graded deliverable silently stops shipping.
    """
    required = ["generator/output/pairs.csv",
                "generator/output/ground_truth.csv",
                "generator/output/manifest.csv",
                "generator/output/baseline_calibration.txt",
                "generator/output/contact_sheet.png",
                "generator/output/REPORT.md"]
    assert [p for p in required if p not in manifest] == []
    for sub in ("reference", "search"):
        images = [n for n in manifest
                  if n.startswith("generator/output/" + sub + "/")
                  and n.endswith(".png")]
        assert len(images) == 20, sub + ": " + str(len(images)) + " images"


def test_archive_is_flat(manifest):
    """register.py at the archive root: the organizer runs it from there."""
    assert "register.py" in manifest
    assert not any(n.startswith("/") or ".." in n.split("/") or "\\" in n
                   for n in manifest)


def _module_targets(path):
    """Top-level names a test file imports, absolute and relative alike."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_shipped_tests_only_import_shipped_modules(manifest):
    """A shipped test that imports a module left behind fails on collection.

    The extraction's sys.path is the extraction root plus generator/ (see
    tests/conftest.py), so a local import resolves only against what ships.
    """
    shipped = set(manifest)
    # Top-level importable names the extraction provides.
    local = {n[:-3] for n in shipped if n.endswith(".py") and "/" not in n}
    local |= {n.split("/")[0] for n in shipped if "/" in n}
    local |= {n.split("/")[1] for n in shipped
              if n.startswith("generator/") and n.count("/") > 1}
    # Modules importable as siblings inside tests/ and generator/tests/.
    for pkg in ("tests/", "generator/tests/"):
        local |= {n[len(pkg):-3] for n in shipped
                  if n.startswith(pkg) and n.endswith(".py")}

    # Anything resolvable in the repo but NOT shipped is the failure mode.
    unshipped = set()
    for entry in sorted(os.listdir(REPO)):
        if entry.endswith(".py"):
            unshipped.add(entry[:-3])
        elif os.path.isdir(os.path.join(REPO, entry)):
            unshipped.add(entry)
    for entry in sorted(os.listdir(os.path.join(REPO, "tests"))):
        if entry.endswith(".py"):
            unshipped.add(entry[:-3])
    unshipped -= local

    broken = {}
    for name in sorted(n for n in shipped
                       if n.startswith(("tests/", "generator/tests/"))
                       and n.endswith(".py")):
        bad = sorted(_module_targets(os.path.join(REPO, name)) & unshipped)
        if bad:
            broken[name] = bad
    assert broken == {}, (
        "shipped tests import modules that stay behind; either ship the "
        "module or drop the test from ALLOW: " + repr(broken))


def test_builder_writes_a_clean_archive(builder, tmp_path):
    """End-to-end: build, then re-audit the finished archive's own namelist."""
    import zipfile

    out = str(tmp_path / "submission.zip")
    assert builder.build(out, REPO) == 0
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert zf.testzip() is None
    assert [n for n in names if builder.deny_reason(n)] == []
    assert "register.py" in names
    assert [n for n in names if n.startswith("weights/")] == \
        [builder.WEIGHTS_ONLY]


def test_build_refuses_a_tree_that_would_leak(builder, tmp_path, monkeypatch):
    """The DENY guard must abort the build, not warn."""
    fake = tmp_path / "repo"
    (fake / ".agents").mkdir(parents=True)
    (fake / ".agents" / "PHASE2_ADDENDUM.md").write_text("confidential")
    (fake / "register.py").write_text("# entry point\n")
    monkeypatch.setattr(builder, "ALLOW", ["register.py", ".agents"])
    out = str(tmp_path / "leak.zip")
    assert builder.build(out, str(fake)) == 1
    assert not os.path.exists(out)


def test_build_refuses_a_missing_manifest_entry(builder, tmp_path, monkeypatch):
    fake = tmp_path / "repo"
    fake.mkdir()
    (fake / "register.py").write_text("# entry point\n")
    monkeypatch.setattr(builder, "ALLOW", ["register.py", "does_not_exist.py"])
    out = str(tmp_path / "incomplete.zip")
    assert builder.build(out, str(fake)) == 1
    assert not os.path.exists(out)


def test_build_refuses_a_truncated_checkpoint(builder, tmp_path, monkeypatch):
    """A stub or unfetched LFS pointer must not ship as the weights file."""
    fake = tmp_path / "repo"
    (fake / "weights").mkdir(parents=True)
    (fake / "weights" / "driftsense.pt").write_bytes(b"version https://git-lfs")
    (fake / "register.py").write_text("# entry point\n")
    monkeypatch.setattr(builder, "ALLOW",
                        ["register.py", "weights/driftsense.pt"])
    out = str(tmp_path / "stub.zip")
    assert builder.build(out, str(fake)) == 1
    assert not os.path.exists(out)


def test_build_is_reproducible(builder, tmp_path):
    """Two builds of the same tree are byte-identical."""
    first = str(tmp_path / "a.zip")
    second = str(tmp_path / "b.zip")
    assert builder.build(first, REPO) == 0
    assert builder.build(second, REPO) == 0
    with open(first, "rb") as fa, open(second, "rb") as fb:
        assert fa.read() == fb.read()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__]))

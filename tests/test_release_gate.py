"""The release gate must actually fail.

`scripts/release_gate.py` is the control added after the tree shipped a
state whose ZIP could not be built: every individual check existed and
passed, and nothing composed them. A gate that only ever prints PASS would
reproduce exactly that failure, so these tests assert the *negative* side --
each forbidden condition, built into a synthetic archive, must be caught.

The `.agents/` case is the one with teeth: that directory carries a deck
transcribed from material marked "Applied Materials Confidential" and an
organizer SharePoint capability URL. `.gitignore` documents the danger in
prose; this is the part that enforces it.
"""

import importlib.util
import io
import os
import sys
import zipfile

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

GATE_PATH = os.path.join(REPO_ROOT, "scripts", "release_gate.py")
WORKFLOW = os.path.join(REPO_ROOT, ".github", "workflows", "tests.yml")


def _load_gate():
    spec = importlib.util.spec_from_file_location("release_gate_test",
                                                  GATE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gatemod():
    """A fresh module per test -- `results` is module-level state."""
    mod = _load_gate()
    del mod.results[:]
    return mod


def _zip(tmp_path, entries, name="artifact.zip"):
    """Build a synthetic archive. `entries` maps arcname -> bytes."""
    path = tmp_path / name
    with zipfile.ZipFile(str(path), "w") as archive:
        for arcname, payload in entries.items():
            archive.writestr(arcname, payload)
    return str(path)


CLEAN = {
    "register.py": b"# entry point\n",
    "infer.py": b"# loader\n",
    "requirements.txt": b"torch==2.13.0\n",
    "weights/driftsense.pt": b"\x00" * 64,
    "driftsense/__init__.py": b"",
}


def test_clean_archive_passes_the_content_gate(tmp_path, gatemod):
    """Control: the gate must not fire on a well-formed artifact, or its
    failures below would prove nothing."""
    assert gatemod.gate_forbidden_content(_zip(tmp_path, CLEAN)) is True


def test_agents_directory_is_caught(tmp_path, gatemod):
    """The confidential-material case. Must fail loudly."""
    entries = dict(CLEAN)
    entries[".agents/PHASE2_ADDENDUM.md"] = b"Applied Materials Confidential\n"
    assert gatemod.gate_forbidden_content(_zip(tmp_path, entries)) is False
    failed = [r for r in gatemod.results if not r["ok"]]
    assert any("denied path" in r["name"] for r in failed), gatemod.results
    assert any(".agents" in r["detail"] for r in failed), gatemod.results


def test_agents_nested_below_another_directory_is_caught(tmp_path, gatemod):
    """A prefix test alone would miss `docs/.agents/notes.md`."""
    entries = dict(CLEAN)
    entries["docs/.agents/notes.md"] = b"internal\n"
    assert gatemod.gate_forbidden_content(_zip(tmp_path, entries)) is False


def test_phase1_and_scripts_are_caught(tmp_path, gatemod):
    entries = dict(CLEAN)
    entries["phase1/train.py"] = b"# archived duplicate\n"
    entries["scripts/release_gate.py"] = b"# tooling is not submission content\n"
    assert gatemod.gate_forbidden_content(_zip(tmp_path, entries)) is False


def test_a_second_checkpoint_is_caught(tmp_path, gatemod):
    """48 MB of unused checkpoints live in weights/; exactly one may ship."""
    entries = dict(CLEAN)
    entries["weights/driftsense_wide.pt"] = b"\x00" * 64
    assert gatemod.gate_forbidden_content(_zip(tmp_path, entries)) is False
    failed = [r for r in gatemod.results if not r["ok"]]
    assert any("checkpoint" in r["name"] for r in failed), gatemod.results


def test_git_lfs_pointer_is_caught(tmp_path, gatemod):
    """An LFS stub is a promise to fetch bytes over a network the graded
    machine does not have."""
    entries = dict(CLEAN)
    entries["weights/driftsense.pt"] = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + b"0" * 64 + b"\nsize 16504444\n")
    assert gatemod.gate_forbidden_content(_zip(tmp_path, entries)) is False
    failed = [r for r in gatemod.results if not r["ok"]]
    assert any("LFS" in r["name"] for r in failed), gatemod.results


def test_gate_records_every_result_for_the_provenance_file(tmp_path, gatemod):
    """The provenance record is the artifact's evidence; it must carry the
    per-gate outcomes, not just a verdict."""
    gatemod.gate_forbidden_content(_zip(tmp_path, CLEAN))
    assert gatemod.results, "no gate results recorded"
    for entry in gatemod.results:
        assert set(entry) == {"name", "ok", "detail"}, entry


def test_release_gate_runs_in_ci():
    """Contract pin. The gate only prevents a recurrence if CI runs it --
    the vst break reached main because nothing ran the builder between the
    merge and the discovery."""
    if not os.path.isfile(WORKFLOW):
        pytest.skip("no tests.yml workflow")
    with io.open(WORKFLOW, encoding="utf-8") as handle:
        workflow = handle.read()
    assert "release_gate.py" in workflow, (
        "scripts/release_gate.py is not invoked by .github/workflows/"
        "tests.yml; an uninvoked gate is not a control")

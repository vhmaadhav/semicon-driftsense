"""A measurement must carry the conditions that produced it.

PR #30 is the standing example of what this costs. Its 2,500-pair headline run
exists, but the log records `dirty: 2 files`, so the number cannot be
reproduced from the commit it names -- and the whole severity campaign sits
behind redoing it on the one machine that holds `data/ext_p2`. Before this
module, neither `scripts/eval_ext.py` nor `scripts/eval_phase2.py` nor
`scripts/grade_emulation.py` recorded a commit, a dirty flag, a seed or an
environment at all; that note was written by hand, after the fact.

The `_porcelain_paths` tests are a regression pin, not theory. The first
version of this module called `.strip()` on `git status --porcelain` output.
Porcelain encodes status in the first two columns, so a modified-not-staged
file reads `" M scripts/eval_phase2.py"` -- stripping ate the leading space
and `line[3:]` then reported `cripts/eval_phase2.py`. A provenance record that
silently mangles the paths it exists to record is worse than none.
"""

import ast
import io
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from driftsense import provenance  # noqa: E402


# -- porcelain parsing ----------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    (" M scripts/eval_phase2.py", "scripts/eval_phase2.py"),   # the bug
    ("M  scripts/eval_phase2.py", "scripts/eval_phase2.py"),   # staged
    ("?? driftsense/provenance.py", "driftsense/provenance.py"),
    ("MM a/b/c.py", "a/b/c.py"),
    ("A  weights/driftsense.pt", "weights/driftsense.pt"),
    ('?? "path with spaces.md"', "path with spaces.md"),
    ("R  old/name.py -> new/name.py", "new/name.py"),
])
def test_porcelain_paths_preserves_the_whole_path(line, expected):
    assert provenance._porcelain_paths(line) == [expected]


def test_porcelain_paths_never_drops_a_leading_character():
    """The specific regression: every path must survive byte-for-byte."""
    status = "\n".join([" M scripts/eval_phase2.py",
                        "?? driftsense/provenance.py",
                        " M scripts/release_gate.py"])
    got = provenance._porcelain_paths(status)
    assert got == ["scripts/eval_phase2.py",
                   "driftsense/provenance.py",
                   "scripts/release_gate.py"], got
    assert not any(p.startswith("cripts") for p in got), got


def test_porcelain_paths_on_a_clean_tree():
    assert provenance._porcelain_paths("") == []
    assert provenance._porcelain_paths(None) == []


# -- the refusal ----------------------------------------------------------

def test_require_clean_refuses_a_dirty_tree():
    rec = {"dirty": 2, "dirty_paths": ["a.py", "b.py"], "commit": "deadbeef"}
    with pytest.raises(SystemExit) as exc:
        provenance.require_clean(rec)
    message = str(exc.value)
    assert "dirty worktree" in message
    # The message must say what to do, and why it matters.
    assert "a.py" in message and "--allow-dirty" in message
    assert "PR #30" in message


def test_require_clean_passes_a_clean_tree():
    rec = {"dirty": 0, "dirty_paths": [], "commit": "deadbeef"}
    assert provenance.require_clean(rec) is rec
    assert "allow_dirty" not in rec


def test_allow_dirty_marks_the_record_rather_than_hiding_the_problem(capsys):
    """An escape hatch that leaves no trace is how an unreproducible number
    gets quoted six weeks later."""
    rec = {"dirty": 1, "dirty_paths": ["a.py"], "commit": "deadbeef"}
    provenance.require_clean(rec, allow_dirty=True)
    assert rec["allow_dirty"] is True
    assert "not reproducible" in capsys.readouterr().err


def test_allow_dirty_can_be_set_by_environment(monkeypatch, capsys):
    monkeypatch.setenv("DRIFTSENSE_ALLOW_DIRTY", "1")
    rec = {"dirty": 1, "dirty_paths": ["a.py"], "commit": "deadbeef"}
    provenance.require_clean(rec)
    assert rec["allow_dirty"] is True
    capsys.readouterr()


# -- the record -----------------------------------------------------------

def test_record_captures_what_a_rerun_needs():
    rec = provenance.record(split="/tmp/x", seed=1234)
    for key in ("recorded_utc", "commit", "branch", "dirty", "dirty_paths",
                "command", "versions", "threads"):
        assert key in rec, key
    assert rec["seed"] == 1234 and rec["split"] == "/tmp/x"
    assert rec["versions"]["python"]
    # The thread caps must be read back, not assumed -- quoting a runtime
    # without them is how a 10-core box stands in for a 4-core one.
    assert "cpu_count" in rec["threads"]
    assert isinstance(rec["dirty"], int)


def test_record_hashes_the_weights_it_names(tmp_path):
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"not really a checkpoint")
    rec = provenance.record(weights=str(weights))
    assert rec["weights_sha256"] == provenance.sha256(str(weights))
    assert len(rec["weights_sha256"]) == 64


def test_sha256_of_a_missing_file_is_none():
    assert provenance.sha256("/nonexistent/file.pt") is None
    assert provenance.sha256(None) is None


def test_write_produces_readable_json(tmp_path):
    out = tmp_path / "sub" / "provenance.json"
    provenance.write(str(out), provenance.record(note="hello"))
    with io.open(str(out), encoding="utf-8") as fh:
        assert json.load(fh)["note"] == "hello"


# -- wiring ---------------------------------------------------------------

def test_eval_phase2_checks_provenance_before_it_decodes():
    """Order matters. Failing in the first second of a 2,250-pair run is
    cheap; failing after it is what blocked PR #30."""
    path = os.path.join(REPO_ROOT, "scripts", "eval_phase2.py")
    with io.open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    main = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == "main")

    def line_of(predicate):
        for node in ast.walk(main):
            if isinstance(node, ast.Call) and predicate(node):
                return node.lineno
        return None

    require_line = line_of(
        lambda n: isinstance(n.func, ast.Attribute)
        and n.func.attr == "require_clean")
    load_line = line_of(
        lambda n: isinstance(n.func, ast.Attribute)
        and n.func.attr == "load_model")

    assert require_line, "eval_phase2.py does not assert provenance at all"
    assert load_line, "eval_phase2.py no longer loads a model?"
    assert require_line < load_line, (
        "provenance.require_clean runs at line " + str(require_line)
        + " but the model loads at line " + str(load_line)
        + "; the check must come first, before any compute is spent")

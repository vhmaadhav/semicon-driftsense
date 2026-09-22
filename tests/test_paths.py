"""driftsense.paths.is_rooted, and both readers that depend on it.

A pairs.csv value is either a location or a path relative to the manifest.
Getting that wrong does not raise -- it silently reads a different file, or
none. These run on every platform on purpose: the whole class of bug is that
the answer used to depend on which platform, and which Python, you were
standing on.

Regression: Python 3.13 narrowed ntpath.isabs, so on Windows
"/abs/search/p001.png" stopped being absolute and was joined to the
manifest's directory, which splices the current drive on and yields
"C:/abs/search/p001.png".
"""

import csv
import os

import pytest

from driftsense.paths import is_rooted


ROOTED = [
    ("/data/search/p001.png", "POSIX absolute"),
    ("/x.png", "POSIX absolute, one segment"),
    ("//server/share/p001.png", "double slash; POSIX-absolute either way"),
    (r"\\server\share\p001.png", "UNC"),
    (r"\data\search\p001.png", "rooted on the current drive"),
    ("C:/data/search/p001.png", "Windows drive, forward slashes"),
    (r"C:\data\search\p001.png", "Windows drive, backslashes"),
    ("z:/data/p001.png", "lowercase drive letter"),
]

RELATIVE = [
    ("reference/p001.gds", "the ordinary manifest form"),
    ("./reference/p001.gds", "explicitly relative"),
    ("../sibling/p001.gds", "parent-relative"),
    ("p001.png", "bare filename"),
    ("a/b/c/d.png", "nested relative"),
]


@pytest.mark.parametrize("value,why", ROOTED, ids=[w for _, w in ROOTED])
def test_rooted_values_are_recognised(value, why):
    assert is_rooted(value) is True


@pytest.mark.parametrize("value,why", RELATIVE, ids=[w for _, w in RELATIVE])
def test_relative_values_are_not_rooted(value, why):
    assert is_rooted(value) is False


def test_graded_inputs_classify_exactly_as_isabs_did():
    """The compatibility guarantee, stated as a test.

    is_rooted deliberately diverges from os.path.isabs for Windows-authored
    values -- that is the fix. It must not diverge for the two kinds a graded
    run carries: a POSIX-absolute path and a relative one.
    """
    assert is_rooted("/data/search/p001.png") is True
    assert is_rooted("/x.png") is True
    for value, _ in RELATIVE:
        assert is_rooted(value) is os.path.isabs(value) is False


def test_empty_string_is_not_rooted():
    """pathval() short-circuits before calling this, but the predicate should
    not depend on that."""
    assert is_rooted("") is False


# --------------------------------------------------------------------------
# Both readers. Phase 3 goes through driftsense.pairs3.read_pairs; Phase 2
# through register.py's own resolve(). The bug was in both, so both are
# pinned here -- fixing one and leaving the graded Phase 2 path broken in the
# same way would have been half a fix.

PHASE3_HEADER = ["pair_id", "search_path", "reference_gds_path",
                 "search_gds_path", "reference_sem_path", "params_json_path"]
PHASE3_ROW = {
    "pair_id": "p001",
    "search_path": "search/p001.png",
    "reference_gds_path": "reference/p001.gds",
    "search_gds_path": "",
    "reference_sem_path": "",
    "params_json_path": "",
}


def _write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


@pytest.mark.parametrize("value", ["/abs/search/p001.png",
                                   "C:/data/search/p001.png"])
def test_phase3_reader_leaves_rooted_paths_alone(tmp_path, value):
    from driftsense.pairs3 import read_pairs

    row = dict(PHASE3_ROW, search_path=value)
    csv_path = _write_csv(tmp_path / "pairs.csv", PHASE3_HEADER, [row])

    got = read_pairs(csv_path)[0].search_path
    assert got == value
    assert str(tmp_path) not in got, (
        "the manifest directory was joined onto a rooted path: " + got)


def test_phase3_reader_still_resolves_relative_paths(tmp_path):
    """Guard the other direction: the fix must not root everything."""
    from driftsense.pairs3 import read_pairs

    csv_path = _write_csv(tmp_path / "pairs.csv", PHASE3_HEADER, [PHASE3_ROW])
    resolved = read_pairs(csv_path)[0].reference_gds_path
    assert os.path.isabs(resolved)
    assert os.path.dirname(resolved) == os.path.join(str(tmp_path),
                                                     "reference")


def test_register_resolve_uses_the_same_predicate():
    """register.py must classify a pairs.csv value the way pairs3 does.

    Checked as source rather than by running the entry point: resolve() is a
    closure inside main() and the graded path needs weights, a model load and
    real images to reach it. What matters is that it no longer branches on
    os.path.isabs.
    """
    import ast

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source = open(os.path.join(here, "register.py"), encoding="utf-8").read()

    assert "from driftsense.paths import is_rooted" in source, (
        "register.py should import the shared predicate")

    tree = ast.parse(source)
    resolve = next(
        (node for node in ast.walk(tree)
         if isinstance(node, ast.FunctionDef) and node.name == "resolve"),
        None)
    assert resolve is not None, "register.py no longer defines resolve()"

    called = {node.func.id for node in ast.walk(resolve)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert "is_rooted" in called, "resolve() should call is_rooted"

    attribute_calls = {
        ".".join(filter(None, [getattr(node.func.value, "attr", None),
                               node.func.attr]))
        for node in ast.walk(resolve)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not any(name.endswith("isabs") for name in attribute_calls), (
        "resolve() still branches on os.path.isabs: " + repr(attribute_calls))


def test_phase3_existing_does_not_reinterpret_a_rooted_path(tmp_path, monkeypatch):
    """phase3._existing falls back to a working-directory reading when the
    resolved path is missing. A rooted value must not qualify.

    Without the fix, "/data/p001.png" is not isabs on Windows under 3.13, so
    a same-named file happening to sit under the current drive root would be
    adopted -- silently registering against the wrong image.
    """
    import phase3

    rooted = "/data/definitely-absent-p001.png"
    monkeypatch.chdir(tmp_path)
    assert phase3._existing(rooted, rooted) == rooted


def test_phase3_existing_still_tries_the_cwd_for_relative_values(tmp_path, monkeypatch):
    """The behaviour the fix must preserve: a genuinely relative value that
    exists under the working directory is still adopted."""
    import phase3

    (tmp_path / "search").mkdir()
    real = tmp_path / "search" / "p001.png"
    real.write_bytes(b"")
    monkeypatch.chdir(tmp_path)

    missing = str(tmp_path / "elsewhere" / "search" / "p001.png")
    got = phase3._existing(missing, "search/p001.png")
    assert got == os.path.abspath("search/p001.png")
    assert os.path.exists(got)

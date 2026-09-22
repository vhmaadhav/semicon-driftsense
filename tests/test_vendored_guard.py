"""driftsense._vendored: the generator tree is found, or the failure explains.

driftsense.generate and driftsense.presets import the vendored generator's
`src` package. pyproject.toml packages `driftsense` alone -- the generator's
upstream terms are undetermined (NOTICE) -- so from an installed wheel there
is no tree beside them and the import cannot succeed.

That is intended. What is not acceptable is the bare symptom,
`ModuleNotFoundError: No module named 'src'`, which names neither the cause
nor the fix. These tests pin the guard that replaces it, and pin that the
inference path never depends on the tree in the first place.
"""

import os
import sys

import pytest

from driftsense import _vendored


def test_generator_root_sits_beside_the_package():
    """The tree is resolved relative to __file__, not to the cwd."""
    assert _vendored.GENERATOR_ROOT == os.path.join(
        _vendored.REPO_ROOT, "generator")
    assert os.path.basename(_vendored.REPO_ROOT) != "driftsense", (
        "REPO_ROOT should be the repository, one level above the package")


def test_found_in_a_checkout_and_put_on_the_path():
    """In this repository the tree exists, so the call succeeds."""
    used = _vendored.ensure_generator_on_path()
    assert used == _vendored.GENERATOR_ROOT
    assert os.path.isdir(os.path.join(used, "src"))
    assert used in sys.path


def test_idempotent():
    """A second call is a no-op: it must not append another entry.

    Asserting a global `count == 1` here would be wrong. Running the whole
    suite in one process, sys.path picks up fifteen copies of the generator
    path from the other insertion sites scattered across the repository --
    this function cannot dedupe what it did not add. The contract it does own
    is that calling it again changes nothing.
    """
    _vendored.ensure_generator_on_path()
    before = list(sys.path)
    occurrences = before.count(_vendored.GENERATOR_ROOT)
    assert occurrences >= 1

    _vendored.ensure_generator_on_path()

    assert sys.path == before, "a repeat call must not modify sys.path"
    assert sys.path.count(_vendored.GENERATOR_ROOT) == occurrences


def test_missing_tree_raises_an_actionable_importerror(monkeypatch, tmp_path):
    """The wheel case: no tree beside the package.

    The error has to be an ImportError subclass so callers that already guard
    imports keep working, and it has to say the three things the bare
    ModuleNotFoundError does not: where it looked, why the tree is absent, and
    that inference is unaffected.
    """
    absent = str(tmp_path / "no-generator-here")
    monkeypatch.setattr(_vendored, "GENERATOR_ROOT", absent)

    with pytest.raises(_vendored.VendoredGeneratorMissing) as caught:
        _vendored.ensure_generator_on_path()

    assert issubclass(_vendored.VendoredGeneratorMissing, ImportError)
    message = str(caught.value)
    assert absent in message, "must say where it looked"
    assert "NOTICE" in message, "must point at why the tree is not installed"
    assert "register.py" in message and "phase3.py" in message, (
        "must say the graded entry points do not need it")


def test_missing_tree_does_not_touch_sys_path(monkeypatch, tmp_path):
    absent = str(tmp_path / "no-generator-here")
    monkeypatch.setattr(_vendored, "GENERATOR_ROOT", absent)
    before = list(sys.path)
    with pytest.raises(_vendored.VendoredGeneratorMissing):
        _vendored.ensure_generator_on_path()
    assert sys.path == before


def test_the_inference_path_never_imports_the_vendored_modules():
    """The reason a wheel without generator/ is still a working localiser.

    driftsense/__init__.py must not pull in generate or presets, and the
    graded entry points must not import them either. Read as source rather
    than imported, so this test holds even where torch cannot load.
    """
    import ast

    repo = _vendored.REPO_ROOT
    vendored_dependents = {"driftsense.generate", "driftsense.presets"}

    for relative in ("driftsense/__init__.py", "register.py", "phase3.py"):
        path = os.path.join(repo, relative)
        with open(path, encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), path)

        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)

        assert not (imported & vendored_dependents), (
            relative + " imports " + repr(sorted(imported & vendored_dependents))
            + ", which needs the vendored generator tree and therefore cannot "
              "work from an installed wheel")

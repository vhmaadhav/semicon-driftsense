"""Every first-party module must import. Loudly.

This exists because of a specific near-miss. A linter autofix removed

    from src.presets import PRESETS

from driftsense/generate.py as "unused" -- correctly, as far as that file is
concerned. It was a re-export: driftsense/stream_dataset.py, train.py and
tests/test_decoy_pitch.py all import PRESETS from driftsense.generate, and all
three broke with ImportError.

The suite still reported zero failures. tests/test_stream_quota.py reaches the
module through `pytest.importorskip("driftsense.stream_dataset")`, and on
pytest 8 an ImportError inside importorskip is a *skip*, not a failure. Two
broken modules, including the training entry point, looked like a green run.
(pytest 9 turns that into an error and requirements.txt pins 9.1.1, so CI
would very likely have caught it -- but a test suite should not depend on the
runner's version to notice that the package no longer imports.)

So: import everything, and distinguish the one absence that is legitimate.
"""

import glob
import importlib
import os

import pytest

from driftsense._vendored import VendoredGeneratorMissing

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIRST_PARTY = sorted(
    "driftsense." + os.path.basename(p)[:-3]
    for p in glob.glob(os.path.join(REPO, "driftsense", "*.py"))
    if os.path.basename(p) != "__init__.py"
)

# Top-level modules that are part of the product rather than tooling.
ENTRY_POINTS = ["register", "phase3", "infer", "evaluate", "train",
                "generate_dataset", "generate_phase3_dataset"]


def test_the_package_has_modules_to_check():
    """Guard against the glob silently matching nothing."""
    assert len(FIRST_PARTY) >= 15, FIRST_PARTY
    assert "driftsense.matching" in FIRST_PARTY
    assert "driftsense.generate" in FIRST_PARTY


@pytest.mark.parametrize("name", FIRST_PARTY)
def test_first_party_module_imports(name):
    try:
        importlib.import_module(name)
    except VendoredGeneratorMissing:
        # The only acceptable failure: driftsense.generate and
        # driftsense.presets need the vendored generator tree, which ships
        # only in a source checkout (see NOTICE and driftsense/_vendored.py).
        pytest.skip(name + " needs the vendored generator tree")


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_entry_point_imports(name):
    try:
        importlib.import_module(name)
    except VendoredGeneratorMissing:
        pytest.skip(name + " needs the vendored generator tree")


def test_reexports_that_other_modules_depend_on():
    """Names imported from a module that does not itself use them.

    A linter cannot see these and will offer to delete them. Each entry is a
    name some other module imports from the module named on the left, so
    removing it is an ImportError somewhere else.
    """
    expected = {
        "driftsense.generate": ["PRESETS", "PoseSpec", "make_pairs"],
    }
    for module_name, names in expected.items():
        module = importlib.import_module(module_name)
        missing = [n for n in names if not hasattr(module, n)]
        assert not missing, (
            module_name + " no longer exports " + repr(missing)
            + " -- other modules import these from it; if the removal is "
              "intentional, update their imports and this list together")

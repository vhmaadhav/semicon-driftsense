"""The two import-closure walkers must agree.

This repository has TWO independent implementations of "which local modules
can register.py reach?":

* ``scripts/build_submission_zip.py:driftsense_imports`` decides what goes
  INTO the ZIP (via ``check_driftsense``), and
* ``scripts/check_submission_zip.py:imported_names`` decides what gets
  SCANNED for network calls (via ``transitive_local_modules``).

They disagreed. The builder handles ``from driftsense import vst``
explicitly (build_submission_zip.py, "from driftsense import x" branch);
the checker recorded only ``node.module``, so that form resolved to
``driftsense/__init__.py`` and ``driftsense/vst.py`` was never read. The
artifact audit reported "scanned 10 file(s)" with vst.py sitting in the
archive -- i.e. the *no network calls in entry-point import closure* check,
the one that backs the organizer's no-network requirement, had a hole in
exactly the shape of a module imported as ``from package import module``.

A weaker scanner is not a cosmetic bug: it passes silently. These tests pin
the invariant rather than the symptom, so the next module added this way
cannot reopen it.
"""

import ast
import importlib.util
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

# Both walkers start from the entry points the network audit actually uses.
NETWORK_AUDIT_ENTRIES = ("register.py", "infer.py")


def _load(script):
    path = os.path.join(REPO_ROOT, "scripts", script)
    spec = importlib.util.spec_from_file_location(
        script.replace(".py", "") + "_closure_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _load("build_submission_zip.py")


@pytest.fixture(scope="module")
def checker():
    return _load("check_submission_zip.py")


def _builder_closure(builder, entries):
    """driftsense module names the BUILDER's walker reaches from `entries`."""
    reached, queue = set(), []
    for entry in entries:
        path = os.path.join(REPO_ROOT, entry)
        if os.path.isfile(path):
            queue.extend(builder.driftsense_imports(path))
    while queue:
        name = queue.pop()
        if name in reached:
            continue
        reached.add(name)
        module = os.path.join(REPO_ROOT, "driftsense", name + ".py")
        if os.path.isfile(module):
            queue.extend(builder.driftsense_imports(module))
    return {n for n in reached
            if os.path.isfile(
                os.path.join(REPO_ROOT, "driftsense", n + ".py"))}


def _checker_closure(checker, entries):
    """driftsense module names the CHECKER's walker reaches from `entries`."""
    files = checker.transitive_local_modules(
        REPO_ROOT, [os.path.join(REPO_ROOT, e) for e in entries])
    names = set()
    for path in files:
        rel = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
        if rel.startswith("driftsense/") and rel.endswith(".py"):
            stem = rel[len("driftsense/"):-len(".py")]
            if stem != "__init__":
                names.add(stem)
    return names


def test_from_package_import_module_yields_the_dotted_module():
    """`from driftsense import vst` must offer `driftsense.vst` to
    resolve_local, not just `driftsense`. This is the specific defect."""
    checker = _load("check_submission_zip.py")
    names = checker.imported_names("from driftsense import vst\n")
    assert "driftsense.vst" in names, (
        "imported_names lost the submodule in `from pkg import mod`; got "
        + repr(sorted(names)))
    target = checker.resolve_local(REPO_ROOT, "driftsense.vst")
    assert target and os.path.basename(target) == "vst.py", (
        "driftsense.vst did not resolve to driftsense/vst.py; got "
        + repr(target))


def test_plain_from_import_of_a_name_still_resolves_to_its_module():
    """`from driftsense.config import SHIPPED_BAND` must keep resolving to
    driftsense/config.py -- the fix must not change this path."""
    checker = _load("check_submission_zip.py")
    names = checker.imported_names(
        "from driftsense.config import SHIPPED_BAND\n")
    target = checker.resolve_local(REPO_ROOT, "driftsense.config")
    assert "driftsense.config" in names
    assert target and os.path.basename(target) == "config.py", repr(target)


def test_checker_closure_covers_every_module_the_builder_ships(builder,
                                                               checker):
    """The invariant. Anything the builder considers reachable (and therefore
    ships) must also be scanned for network calls. A module in the ZIP that
    the network audit never reads is an unaudited file in the artifact."""
    shipped = _builder_closure(builder, NETWORK_AUDIT_ENTRIES)
    scanned = _checker_closure(checker, NETWORK_AUDIT_ENTRIES)
    missing = sorted(shipped - scanned)
    assert not missing, (
        "these driftsense modules are reachable from "
        + ", ".join(NETWORK_AUDIT_ENTRIES)
        + " and ship in the ZIP, but the network audit never scans them: "
        + ", ".join(missing)
        + ". The two walkers have drifted apart again -- see this module's "
          "docstring.")


def test_vst_is_both_shipped_and_scanned(builder, checker):
    """Regression pin for the specific module that exposed the gap."""
    if not os.path.isfile(os.path.join(REPO_ROOT, "driftsense", "vst.py")):
        pytest.skip("driftsense/vst.py has been removed")
    assert "vst" in builder.DRIFTSENSE_SHIP, (
        "driftsense/vst.py is imported at module scope by "
        "driftsense/matching.py; omitting it from DRIFTSENSE_SHIP makes the "
        "extracted artifact fail on import")
    assert "vst" in _checker_closure(checker, NETWORK_AUDIT_ENTRIES), (
        "driftsense/vst.py ships but is not in the scanned network closure")

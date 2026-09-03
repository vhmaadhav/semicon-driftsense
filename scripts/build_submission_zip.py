#!/usr/bin/env python3
"""Build the final submission ZIP from an explicit whitelist.

    python scripts/build_submission_zip.py -o dist/submission.zip
    python scripts/check_submission_zip.py dist/submission.zip

Why a builder exists (PR #48 review, item 3): `check_submission_zip.py` audits
an artifact that already exists. It cannot make a bad artifact good. A `git
archive` or a hand-zipped checkout carries everything tracked -- `.agents/`
benchmark CSVs and internal reports, every experimental checkpoint in
`weights/` (67 MB of which one file is shipped), `data/`, `results/` -- and
some of that material is internal. So the ZIP is *constructed*, never
*filtered*: nothing enters unless a rule below names it.

Three gates, and a file must pass all three:

1.  **Whitelist.** `INCLUDE` is the complete set of rules. A path not matched
    by a rule does not ship, full stop.
2.  **Import closure.** The local modules the entry points transitively import
    are computed with `check_submission_zip.transitive_local_modules` -- the
    same resolver the auditor uses. Every one of them must already be inside
    the whitelist. A needed module outside it is a **build failure**, not a
    silent addition: the whitelist is updated by a human who can see what
    changed.
3.  **Denylist.** A last-resort substring gate over the staged paths, so a
    careless future glob still cannot pull in `.agents/`, `data/`, `secrets/`
    or a stray checkpoint.

The archive is written deterministically -- sorted entries, fixed timestamps,
fixed permissions -- so rebuilding the same checkout produces a byte-identical
ZIP and the audited artifact is provably the uploaded one.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

sys.path.insert(0, HERE)
from check_submission_zip import transitive_local_modules  # noqa: E402

# --------------------------------------------------------------------------
# Gate 1 -- the whitelist. Each entry is either a literal file path or a
# (directory, glob) rule. Both are relative to the repository root, and the
# path inside the ZIP is identical, because the organizer command runs from
# the extraction root:
#     python register.py --input pairs.csv --output predictions.csv
# --------------------------------------------------------------------------
INCLUDE_FILES = [
    # Organizer entry points and the loader they share.
    "register.py",
    "infer.py",
    # Deliverables named by the Phase 2 spec table.
    "generate_dataset.py",
    "train.py",
    "evaluate.py",
    "requirements.txt",
    "README.md",
    "CITATIONS.md",
    "TRAINING.md",
    "FAILURE_ANALYSIS.md",
    "failure_analysis.pdf",
    # The ONE shipped checkpoint. weights/ holds 20 files and 67 MB of
    # experiment history; naming the file rather than the directory is the
    # whole point of the whitelist.
    os.path.join("weights", "driftsense.pt"),
]

INCLUDE_TREES = [
    # (directory, glob) -- recursive.
    ("driftsense", "*.py"),
    # The vendored upstream generator. driftsense/generate.py puts `generator/`
    # on sys.path and imports `src.*` from it, so generate_dataset.py does not
    # run without this -- the AST closure below cannot see it, because the
    # import is resolved through a sys.path insertion at runtime. Only `src/`
    # ships: baseline_solution/, tests/ and .claude/ are upstream extras and
    # internal material, and none of them is imported.
    (os.path.join("generator", "src"), "*.py"),
]

# Added only with --with-tests. The graders do not run the suite, and the
# tests need weights and generated data that are not in the artifact.
TEST_TREES = [("tests", "test_*.py")]
TEST_FILES = ["pytest.ini"]

ENTRY_POINTS = ["register.py", "infer.py", "generate_dataset.py",
                "train.py", "evaluate.py"]

# Extra import roots the shipped code puts on sys.path at runtime, mirroring
# `sys.path.insert(0, .../generator)` in driftsense/generate.py. The closure is
# resolved once per root so a `src.*` import cannot escape the whitelist.
EXTRA_IMPORT_ROOTS = ["generator"]

# --------------------------------------------------------------------------
# Gate 3 -- the denylist. Substring match on the ZIP-relative path, checked
# after staging. Nothing here can ship even if a whitelist rule matched it.
# --------------------------------------------------------------------------
DENY_SUBSTRINGS = [
    ".agents", "data/", "results/", "secrets", "venv", ".git",
    "__pycache__", ".pytest_cache", ".DS_Store", "graphify-out",
    "bundle_stage", "docs/plans", "kaggle",
]
DENY_SUFFIXES = [".pyc", ".pyo", ".log", ".csv", ".zip", ".ipynb"]

# The one checkpoint that may ship. Any other .pt/.pth is a build failure.
ALLOWED_WEIGHT = os.path.join("weights", "driftsense.pt")


def fail(msg):
    print(f"BUILD FAILED: {msg}", file=sys.stderr)
    raise SystemExit(2)


def expand_tree(root, directory, pattern):
    out = []
    base = os.path.join(root, directory)
    if not os.path.isdir(base):
        return out
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames
                       if d not in ("__pycache__", ".pytest_cache")]
        for name in sorted(filenames):
            if fnmatch.fnmatch(name, pattern):
                out.append(os.path.relpath(os.path.join(dirpath, name), root))
    return out


def whitelist_paths(root, with_tests):
    files = list(INCLUDE_FILES)
    trees = list(INCLUDE_TREES)
    if with_tests:
        files += TEST_FILES
        trees += TEST_TREES
    staged = []
    for rel in files:
        if not os.path.isfile(os.path.join(root, rel)):
            fail(f"whitelisted file is missing from the checkout: {rel}")
        staged.append(rel)
    for directory, pattern in trees:
        found = expand_tree(root, directory, pattern)
        if not found:
            fail(f"whitelisted tree matched nothing: {directory}/{pattern}")
        staged += found
    return sorted(set(staged))


def check_import_closure(root, staged):
    """Gate 2: every locally imported module must already be whitelisted."""
    needed = transitive_local_modules(root, [os.path.join(root, e)
                                             for e in ENTRY_POINTS])
    # Same question asked from each runtime sys.path root, seeded with the
    # staged files that live under it.
    for extra in EXTRA_IMPORT_ROOTS:
        sub = os.path.join(root, extra)
        seeds = [os.path.join(root, p) for p in staged
                 if p.startswith(extra + os.sep) and p.endswith(".py")]
        if seeds:
            needed += transitive_local_modules(sub, seeds)
    staged_abs = {os.path.abspath(os.path.join(root, p)) for p in staged}
    missing = sorted(os.path.relpath(p, root) for p in needed
                     if os.path.abspath(p) not in staged_abs)
    if missing:
        fail("the entry points import local modules that the whitelist does "
             "not ship, so the artifact would break on the grader's box:\n  "
             + "\n  ".join(missing)
             + "\nAdd them to INCLUDE_FILES/INCLUDE_TREES deliberately.")
    return len(needed)


def check_denylist(staged):
    """Gate 3: nothing internal, no stray checkpoint, no build droppings."""
    bad = []
    for rel in staged:
        posix = rel.replace(os.sep, "/")
        for token in DENY_SUBSTRINGS:
            if token in posix:
                bad.append(f"{rel}  (matches deny token '{token}')")
        for suffix in DENY_SUFFIXES:
            if posix.endswith(suffix):
                bad.append(f"{rel}  (denied extension '{suffix}')")
        if posix.endswith((".pt", ".pth")) and rel != ALLOWED_WEIGHT:
            bad.append(f"{rel}  (only {ALLOWED_WEIGHT} may ship)")
    if bad:
        fail("staged paths hit the denylist:\n  " + "\n  ".join(sorted(set(bad))))


def write_zip(root, staged, out):
    """Deterministic archive: sorted, fixed mtime, fixed mode."""
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for rel in staged:
            src = os.path.join(root, rel)
            info = zipfile.ZipInfo(rel.replace(os.sep, "/"),
                                   date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(src, "rb") as fh:
                z.writestr(info, fh.read())
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default=os.path.join(REPO, "dist", "submission.zip"))
    ap.add_argument("--with-tests", action="store_true",
                    help="also ship tests/ and pytest.ini (off by default: the "
                         "suite needs weights and generated data the artifact "
                         "does not carry)")
    ap.add_argument("--check", action="store_true",
                    help="run scripts/check_submission_zip.py on the artifact "
                         "just built and inherit its exit code")
    a = ap.parse_args()

    staged = whitelist_paths(REPO, a.with_tests)
    n_closure = check_import_closure(REPO, staged)
    check_denylist(staged)
    out = write_zip(REPO, staged, a.out)

    total = sum(os.path.getsize(os.path.join(REPO, r)) for r in staged)
    print(f"whitelist: {len(staged)} files, {total / 1e6:.1f} MB uncompressed")
    print(f"import closure: {n_closure} local modules, all whitelisted")
    for rel in staged:
        size = os.path.getsize(os.path.join(REPO, rel))
        print(f"  {size / 1024:9.1f} KB  {rel}")
    print(f"\nwrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")
    print(f"sha256 {sha256(out)}")

    checker = os.path.join(HERE, "check_submission_zip.py")
    if a.check:
        print(f"\n--- {os.path.basename(checker)} {out}\n", flush=True)
        raise SystemExit(subprocess.call([sys.executable, checker, out]))
    print(f"\nnow audit the exact artifact you will upload:\n"
          f"  python scripts/check_submission_zip.py {out}")


if __name__ == "__main__":
    main()

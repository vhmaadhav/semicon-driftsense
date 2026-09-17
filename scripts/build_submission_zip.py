#!/usr/bin/env python3
"""Build the Phase 2 submission ZIP from an explicit allow-list.

    python scripts/build_submission_zip.py --out dist/submission.zip

The graded artifact is a ZIP the organizers extract and run on their reference
machine (4-core x86, 8 GB, no GPU, no network, Python 3.11) -- see slide 5 in
.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md: "Weights ship inside the zip --
nothing downloads at run time". Whatever ships is exactly what gets graded, so
the ZIP is built from the ALLOW manifest below rather than from a `git archive`
of main. A repo-shaped archive would bundle phase1/ (a 15 MB archived
duplicate), .agents/ (36 MB of internal notes, one of them transcribed from a
deck marked "Applied Materials Confidential"), and 48 MB of unused legacy
checkpoints -- none of which belongs in what a judge opens.

Two independent guards protect that boundary:

*   ALLOW is a closed list. A path that is not named here, or reachable by
    walking a directory named here, cannot enter the ZIP.
*   DENY is checked twice -- while collecting, and again against the finished
    archive's own namelist. A DENY hit aborts the build and removes the
    output, so a file that must never leave the private repo cannot ship even
    if someone later adds it to ALLOW by mistake.

Missing ALLOW entries are fatal too: a rename that silently drops a required
file from the submission is exactly the failure this script exists to prevent.

Audit the result with the companion checker, which extracts the ZIP into a
temporary directory and audits only that extraction:

    python scripts/check_submission_zip.py dist/submission.zip
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import io
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

# driftsense/ is allow-listed one module at a time (see the ALLOW entry).
# Reachability from the graded entry points is verified at build time, so this
# list cannot silently drop a module register.py needs; it can only refuse to
# ship one nothing needs.
DRIFTSENSE_SHIP = [
    "__init__",
    # reachable from register.py / infer.py / generate_dataset.py
    "calibration", "config", "generate", "matching", "model",
    "policy", "presets", "subpixel", "verification", "vst",
    # vst ships even though SHIPPED_VST == "none": matching.py imports it at
    # module scope and calls vst.resolve_mode() on every coarse sweep, so an
    # extraction without it fails on import before the first pair decodes.
    # PR #20 landed driftsense/vst.py; PR #26 authored this list against a
    # tree that did not have it yet. Each was green alone and red together --
    # which is why check_driftsense() now runs in CI, not just at build time.

    # training-side library. Not reachable from the graded entry points
    # (train.py and evaluate.py are both DENY-listed), but shipped alongside
    # TRAINING.md historically. Named explicitly so that stays a decision
    # rather than a side effect of walking the directory.
    "dataset", "engine", "stream_dataset",
]

NL_BULLET = chr(10) + "  - "

# The files whose imports define what driftsense/ must contain.
ENTRY_POINTS = ["register.py", "infer.py", "generate_dataset.py"]

# Every entry is repo-relative. A file ships as itself; a directory ships as
# its whole subtree minus the PRUNE_* rules below. Grouped by why it ships.
ALLOW = [
    # -- organizer entry points (slide 5) ---------------------------------
    "register.py",           # THE entry point: --input pairs.csv --output ...
    "infer.py",              # ship loader; register.py imports it
    "generate_dataset.py",   # "generate_dataset.py documented" (slide 5)

    # -- the model --------------------------------------------------------
    # driftsense/ ships module by module, never as a directory walk. Research
    # code lands in this package while an experiment is live -- row refiners,
    # scan banks, confidence heads -- and a walk would put all of it in front
    # of a judge purely because it sits next to the shipped decode path. A
    # module absent from DRIFTSENSE_SHIP does not ship; check_driftsense()
    # then fails the build if an entry point actually needs one that is
    # missing, so the closed list cannot ship a hole either.
    *["driftsense/" + m + ".py" for m in DRIFTSENSE_SHIP],
    "driftsense/diagrams",   # slide4.png, the figure README.md references
    "weights/driftsense.pt",  # the ONLY checkpoint that ships (see DENY)

    # -- environment ------------------------------------------------------
    "requirements.txt",      # "requirements.txt from pip freeze" (slide 5)

    # -- graded documents -------------------------------------------------
    "failure_analysis.pdf",  # "failure_analysis.pdf max 2 pages" (slide 5)
    "README.md",
    "CITATIONS.md",          # cited sources (slide 9 / Phase 1 rules)
    "TRAINING.md",
    "FAILURE_ANALYSIS.md",

    # -- the generator deliverable (DOCX section 7) -----------------------
    # Section 7 names generate_phase2.py, baseline.py, score.py,
    # contact_sheet.py, src/ and REPORT.md. check_submission.py ships with
    # them because it is what validates a generated output/ directory, and
    # requirements.txt because the generator pins its own environment.
    "generator/generate_phase2.py",
    "generator/baseline.py",
    "generator/score.py",
    "generator/contact_sheet.py",
    "generator/check_submission.py",
    "generator/src",
    "generator/REPORT.md",
    "generator/README.md",
    "generator/requirements.txt",
    "generator/tests",
    # Section 7 names output/ itself as a deliverable: "output/ with pairs.csv
    # + ground_truth.csv + manifest.csv + baseline_calibration.txt +
    # contact_sheet.png + reference/ + search/". This is the fixed, audited
    # 20-pair package (A8/B6/C4/D2, seed 45045) -- 34 MB, and the single
    # largest thing in the ZIP after the checkpoint. It is NOT regenerable by
    # the judge from the ZIP alone in a way that reproduces these exact
    # labels, so it ships.
    "generator/output",

    # -- the tests a judge can actually run -------------------------------
    # A test ships when both are true: its subject ships, and it passes from
    # the extraction. That rules out the suite's coverage of train.py,
    # evaluate.py, phase1/ and scripts/*.py -- none of which ship, so those
    # files would fail on collection or on a missing path in a judge's
    # extraction. Verified by running the shipped suite from an extraction
    # outside the repo; tests/test_submission_manifest.py holds the line.
    "pytest.ini",
    "tests/conftest.py",
    "tests/test_cache_invalidation.py",
    "tests/test_calibration.py",
    "tests/test_checkpoint_safety.py",
    "tests/test_conv_bn_fusion.py",
    "tests/test_decoy_pitch.py",
    "tests/test_early_exit_gates.py",
    "tests/test_fallback_pose_space.py",
    "tests/test_generator.py",
    "tests/test_geometry.py",
    "tests/test_inference.py",
    "tests/test_labels.py",
    "tests/test_model_shapes.py",
    "tests/test_pose.py",
    "tests/test_pose_rotation_ranking.py",
    "tests/test_presets.py",
    "tests/test_register_runtime_meta.py",
    "tests/test_rejector_features.py",
    "tests/test_scale_semantics.py",
    "tests/test_search_feat_cache.py",
    "tests/test_stream_quota.py",
    "tests/test_subpixel.py",
    "tests/test_verification.py",
    "tests/test_write_split.py",
]

# Never ship, whatever ALLOW says. Matched against the archive-relative name
# of every member, with "/" separators.
DENY = [
    (".agents", ".agents/*",
     "internal notes; PHASE2_ADDENDUM.md is transcribed from a deck marked "
     "'Applied Materials Confidential' and must not leave the private repo"),
    ("phase1", "phase1/*",
     "archived Phase 1 duplicate, irrelevant to a Phase 2 submission"),
    (".git", ".git/*", "version control internals"),
    (".github", ".github/*", "CI configuration"),
    (".claude", "*.claude/*", "local agent configuration"),
    (".coderabbit.yaml", ".coderabbit.yaml", "review-bot configuration"),
    ("venv", "venv*/*", "local virtualenv"),
    ("data", "data/*", "generated training data (~15 GB)"),
    ("results", "results/*", "evaluation scratch output"),
    ("scripts", "scripts/*",
     "development tooling, including this builder, is not submission content"),
    ("train.py", "train.py", "training entry point, not part of a graded run"),
    ("evaluate.py", "evaluate.py", "development evaluation harness"),
]

# weights/: driftsense.pt and nothing else. Spelled out separately because the
# rule is "exactly one file in this directory", which a glob cannot express.
WEIGHTS_ONLY = "weights/driftsense.pt"

# Dropped while walking any allow-listed directory.
PRUNE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git",
              "output", "eval_results"}

# Repo-relative directories that survive PRUNE_DIRS anyway. output/ is a
# scratch name everywhere in this repo except one place: generator/output/ is
# the fixed 20-pair package DOCX section 7 names as a deliverable. Excepting
# it by full path, rather than dropping "output" from PRUNE_DIRS, keeps every
# unrelated output/ directory excluded.
PRUNE_EXCEPTIONS = {"generator/output"}
PRUNE_GLOBS = ["*.pyc", "*.pyo", "*.tmp", ".DS_Store", "*~", "*.orig", "*.rej",
               # Git plumbing means nothing in an extraction. generator/
               # output/.gitattributes exists only to keep the images out of
               # LFS in the repository; it is not submission content.
               ".gitattributes", ".gitignore", ".gitkeep"]

# Sanity floor for the shipped checkpoint: a truncated file or an unfetched
# LFS pointer is small, loads as garbage, and is easy to miss by eye.
MIN_WEIGHTS_BYTES = 1_000_000

# weights/driftsense.pt on disk carries training-resume state -- `optimizer`
# (8.22 MB) and `raw_model` (4.13 MB, the pre-EMA duplicate) -- neither of
# which the shipped inference path ever reads: infer.load_model consumes only
# `ckpt.get("model", ckpt)` and `ckpt.get("arch_kwargs")`, and
# check_submission_zip.py's own checkpoint check asserts only a `model` key.
# There is no weight-hash pin anywhere that a re-save would invalidate
# (verified: grepped for sha256/hexdigest against weights/ -- none).
#
# Stripped at PACKAGE time, not on disk: the file `train.py`/`checkpoint_soup.py`
# read for a training resume is untouched, so this cannot break that lineage.
# 16.50 MB -> 4.13 MB in the artifact, a 75% cut, for zero behaviour change.
STRIP_TRAINING_STATE_FROM = {WEIGHTS_ONLY}
CHECKPOINT_SHIP_KEYS = ("model", "arch_kwargs", "arch")

# generator/.gitattributes routes *.png through Git LFS, so the 41 images in
# generator/output/ are pointers in the repository and only become real bytes
# when LFS smudges them on checkout. Build on a machine where that did not
# happen -- no git-lfs installed, a `git archive`, GitHub's "Download ZIP" --
# and every image ships as ~130 bytes of text that a judge cannot open. The
# builder reads the working tree, so it is the last gate that can catch this.
LFS_POINTER_MAGIC = b"version https://git-lfs.github.com/spec/v1"
LFS_POINTER_MAX_BYTES = 1024

# Reproducible archives: a fixed DOS timestamp, so two builds of the same tree
# are byte-identical. 1980-01-01 is the ZIP epoch.
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def pruned(name):
    return any(fnmatch.fnmatch(name, pat) for pat in PRUNE_GLOBS)


def prune_dir(repo, abspath):
    """True when a directory is scratch and must not be walked."""
    name = os.path.basename(abspath)
    if name not in PRUNE_DIRS:
        return False
    rel = os.path.relpath(abspath, repo).replace(os.sep, "/")
    return rel not in PRUNE_EXCEPTIONS


def stripped_checkpoint_bytes(path):
    """Re-serialise a checkpoint keeping only what inference reads.

    Deferred import: torch is a heavy, optional-at-collection-time dependency,
    and every other path through this builder (--list, a DENY hit, a missing
    file) never needs it.
    """
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    minimal = {k: ckpt[k] for k in CHECKPOINT_SHIP_KEYS if k in ckpt}
    if "model" not in minimal:
        raise ValueError(f"{path}: no 'model' key -- refusing to strip a "
                         f"checkpoint that would ship broken")
    import io
    buf = io.BytesIO()
    torch.save(minimal, buf)
    return buf.getvalue()


def is_lfs_pointer(path):
    """True for an unfetched Git LFS pointer standing in for real content."""
    try:
        if os.path.getsize(path) > LFS_POINTER_MAX_BYTES:
            return False
        with open(path, "rb") as fh:
            return fh.read(len(LFS_POINTER_MAGIC)) == LFS_POINTER_MAGIC
    except OSError:
        return False


def deny_reason(arcname):
    """The reason an archive-relative name is forbidden, else None."""
    if fnmatch.fnmatch(arcname, "weights/*") and arcname != WEIGHTS_ONLY:
        return ("weights/: only " + WEIGHTS_ONLY + " ships; the other "
                "checkpoints and history files are unused legacy artifacts")
    for _, pattern, why in DENY:
        if fnmatch.fnmatch(arcname, pattern):
            return why
    return None


def driftsense_imports(path):
    """driftsense.* submodule names imported by one source file."""
    with io.open(path, encoding="utf-8") as handle:
        tree = ast.parse(handle.read(), filename=path)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:                      # from .x import y
                if node.module:
                    found.add(node.module.split(".")[0])
                else:
                    found.update(a.name for a in node.names)
            elif node.module and node.module.split(".")[0] == "driftsense":
                parts = node.module.split(".")
                if len(parts) > 1:
                    found.add(parts[1])
                else:                           # from driftsense import x
                    found.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[0] == "driftsense" and len(parts) > 1:
                    found.add(parts[1])
    return found


def check_driftsense(repo=REPO):
    """Reasons the driftsense/ manifest is wrong, else an empty list.

    Walks imports out from ENTRY_POINTS. Every driftsense module the graded
    entry points can actually reach has to be in DRIFTSENSE_SHIP -- that is
    the direction that would ship a broken artifact. The other direction is
    deliberate: a module nothing reaches simply does not ship.
    """
    pkg = os.path.join(repo, "driftsense")
    if not os.path.isdir(pkg):
        return ["driftsense/ is missing from the working tree"]

    reached, queue = set(), []
    for entry in ENTRY_POINTS:
        path = os.path.join(repo, entry)
        if os.path.isfile(path):
            queue.extend(driftsense_imports(path))
    while queue:
        name = queue.pop()
        if name in reached:
            continue
        reached.add(name)
        module = os.path.join(pkg, name + ".py")
        if os.path.isfile(module):
            queue.extend(driftsense_imports(module))

    shipped = set(DRIFTSENSE_SHIP)
    on_disk = {f[:-3] for f in os.listdir(pkg) if f.endswith(".py")}
    problems = []

    unshipped = sorted((reached & on_disk) - shipped)
    if unshipped:
        problems.append(
            "DRIFTSENSE_SHIP omits module(s) the entry points import: "
            + ", ".join(unshipped)
            + ". The submission would extract and fail on import.")

    phantom = sorted(shipped - on_disk)
    if phantom:
        problems.append(
            "DRIFTSENSE_SHIP names module(s) that do not exist: "
            + ", ".join(phantom) + ". Fix the list, do not ship a hole.")
    return problems


def collect(repo=REPO):
    """(arcname, abspath) pairs for the manifest, sorted, plus missing entries."""
    members = {}
    missing = []
    for entry in ALLOW:
        src = os.path.join(repo, entry.replace("/", os.sep))
        if os.path.isfile(src):
            members[entry] = src
        elif os.path.isdir(src):
            found = False
            for dirpath, dirnames, filenames in os.walk(src):
                dirnames[:] = sorted(
                    d for d in dirnames
                    if not prune_dir(repo, os.path.join(dirpath, d)))
                for filename in sorted(filenames):
                    if pruned(filename):
                        continue
                    abspath = os.path.join(dirpath, filename)
                    arcname = os.path.relpath(abspath, repo)
                    members[arcname.replace(os.sep, "/")] = abspath
                    found = True
            if not found:
                missing.append(entry + " (empty after pruning)")
        else:
            missing.append(entry)
    return sorted(members.items()), missing


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def human(n):
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return format(size, ".1f").rstrip("0").rstrip(".") + " " + unit
        size /= 1024.0
    return str(n)


def fail(message):
    print("ERROR: " + message, file=sys.stderr)
    return 1


def build(out_path, repo=REPO):
    broken = check_driftsense(repo)
    if broken:
        return fail("driftsense/ manifest is inconsistent:" + NL_BULLET
                    + NL_BULLET.join(broken))

    members, missing = collect(repo)
    if missing:
        return fail("manifest entries missing from the working tree:\n  - "
                    + "\n  - ".join(missing)
                    + "\nA renamed or deleted file has to be fixed in ALLOW, "
                      "not shipped as a hole in the submission.")

    blocked = [(a, deny_reason(a)) for a, _ in members]
    blocked = [(a, why) for a, why in blocked if why]
    if blocked:
        return fail("ALLOW would ship DENY-listed paths:\n  - "
                    + "\n  - ".join(a + ": " + w for a, w in blocked))

    pointers = [a for a, path in members if is_lfs_pointer(path)]
    if pointers:
        shown = pointers[:10]
        more = len(pointers) - len(shown)
        return fail(str(len(pointers)) + " file(s) are unfetched Git LFS "
                    "pointers, not real content:\n  - "
                    + "\n  - ".join(shown)
                    + ("\n  - ... and " + str(more) + " more" if more else "")
                    + "\nRun `git lfs pull` and rebuild. Shipping these would "
                      "put ~130 bytes of pointer text where a judge expects "
                      "an image.")

    weights = dict(members).get(WEIGHTS_ONLY)
    if weights is not None and os.path.getsize(weights) < MIN_WEIGHTS_BYTES:
        return fail(WEIGHTS_ONLY + " is only "
                    + human(os.path.getsize(weights)) + "; expected at least "
                    + human(MIN_WEIGHTS_BYTES)
                    + " -- a truncated checkpoint or an unfetched LFS pointer")

    parent = os.path.dirname(os.path.abspath(out_path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp_path = out_path + ".partial"
    # Bytes actually shipped per member -- not os.path.getsize(abspath), which
    # would report the on-disk (unstripped) checkpoint size and misstate the
    # archive's own uncompressed total in the report below.
    shipped_size = {}
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for arcname, abspath in members:
                info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                if arcname in STRIP_TRAINING_STATE_FROM:
                    try:
                        data = stripped_checkpoint_bytes(abspath)
                    except Exception as exc:  # noqa: BLE001
                        return fail(f"{arcname}: could not strip training "
                                   f"state for shipping ({type(exc).__name__}: "
                                   f"{exc}) -- shipping the full file unstripped "
                                   f"is safer than shipping nothing, so fix this "
                                   f"rather than silently falling back")
                    zf.writestr(info, data)
                    shipped_size[arcname] = len(data)
                else:
                    with open(abspath, "rb") as fh:
                        data = fh.read()
                    zf.writestr(info, data)
                    shipped_size[arcname] = len(data)

        # Second, independent pass: audit the finished archive's own namelist
        # instead of trusting the collection that produced it.
        with zipfile.ZipFile(tmp_path) as zf:
            names = zf.namelist()
        leaked = [(n, deny_reason(n)) for n in names]
        leaked = [(n, why) for n, why in leaked if why]
        if leaked:
            return fail("built archive contains DENY-listed paths:\n  - "
                        + "\n  - ".join(n + ": " + w for n, w in leaked))
        unsafe = [n for n in names
                  if n.startswith("/") or ".." in n.split("/") or "\\" in n]
        if unsafe:
            return fail("unsafe member names: " + ", ".join(unsafe))
        if "register.py" not in names:
            return fail("register.py is not at the archive root; the "
                        "organizer command would fail from the extraction "
                        "directory")

        os.replace(tmp_path, out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    report(out_path, members, shipped_size)
    return 0


def report(out_path, members, shipped_size):
    raw = sum(shipped_size[a] for a, _ in members)
    tops = {}
    for arcname, abspath in members:
        top = arcname.split("/")[0] if "/" in arcname else "(root files)"
        count, size = tops.get(top, (0, 0))
        tops[top] = (count + 1, size + shipped_size[arcname])

    print("built " + out_path)
    print("  " + str(len(members)) + " files, " + human(raw)
          + " uncompressed -> " + human(os.path.getsize(out_path)) + " zipped")
    print()
    width = max(len(t) for t in tops)
    for top in sorted(tops):
        count, size = tops[top]
        label = " file " if count == 1 else " files"
        print("  " + top.ljust(width) + "  " + str(count).rjust(4)
              + label + "  " + human(size).rjust(9))
    print()
    print("  excluded: " + ", ".join(label for label, _, _ in DENY)
          + ", weights/* except driftsense.pt")
    print()
    print("next: python scripts/check_submission_zip.py " + out_path)


def main():
    ap = argparse.ArgumentParser(
        description="Build the Phase 2 submission ZIP from an explicit "
                    "allow-list. See the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("dist", "submission.zip"),
                    help="output ZIP path (default: dist/submission.zip)")
    ap.add_argument("--list", action="store_true",
                    help="print the resolved manifest and exit without "
                         "writing a ZIP")
    args = ap.parse_args()

    if args.list:
        members, missing = collect()
        for arcname, _ in members:
            print(arcname)
        if missing:
            print("MISSING: " + ", ".join(missing), file=sys.stderr)
            return 1
        return 0

    return build(args.out)


if __name__ == "__main__":
    sys.exit(main())

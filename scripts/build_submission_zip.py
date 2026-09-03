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

The companion checker runs automatically on the finished archive: it extracts
the ZIP into a temporary directory and audits only that extraction, so what is
audited is exactly what a judge would open. Building and auditing are one
operation on purpose -- an unaudited artifact is not evidence of anything, and
the two steps drifting apart is how an unaudited ZIP reaches a judge. Its exit
status is this script's exit status. Use `--no-audit` to skip it (for
inspecting a deliberately partial build), then audit by hand with:

    python scripts/check_submission_zip.py dist/submission.zip
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# --------------------------------------------------------------------------
# the manifest
# --------------------------------------------------------------------------

# Every entry is repo-relative. A file ships as itself; a directory ships as
# its whole subtree minus the PRUNE_* rules below. Grouped by why it ships.
ALLOW = [
    # -- organizer entry points (slide 5) ---------------------------------
    "register.py",           # THE entry point: --input pairs.csv --output ...
    "infer.py",              # ship loader; register.py imports it
    "generate_dataset.py",   # "generate_dataset.py documented" (slide 5)

    # -- the model --------------------------------------------------------
    "driftsense",            # the package register.py / infer.py import
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
PRUNE_GLOBS = ["*.pyc", "*.pyo", "*.tmp", ".DS_Store", "*~", "*.orig", "*.rej"]

# Sanity floor for the shipped checkpoint: a truncated file or an unfetched
# LFS pointer is small, loads as garbage, and is easy to miss by eye.
MIN_WEIGHTS_BYTES = 1_000_000

# Reproducible archives: a fixed DOS timestamp, so two builds of the same tree
# are byte-identical. 1980-01-01 is the ZIP epoch.
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def pruned(name):
    return any(fnmatch.fnmatch(name, pat) for pat in PRUNE_GLOBS)


def deny_reason(arcname):
    """The reason an archive-relative name is forbidden, else None."""
    if fnmatch.fnmatch(arcname, "weights/*") and arcname != WEIGHTS_ONLY:
        return ("weights/: only " + WEIGHTS_ONLY + " ships; the other "
                "checkpoints and history files are unused legacy artifacts")
    for _, pattern, why in DENY:
        if fnmatch.fnmatch(arcname, pattern):
            return why
    return None


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
                dirnames[:] = sorted(d for d in dirnames
                                     if d not in PRUNE_DIRS)
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
    try:
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6) as zf:
            for arcname, abspath in members:
                info = zipfile.ZipInfo(arcname, date_time=ZIP_DATE_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(abspath, "rb") as fh:
                    zf.writestr(info, fh.read())

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

    report(out_path, members)
    return 0


def report(out_path, members):
    raw = sum(os.path.getsize(p) for _, p in members)
    tops = {}
    for arcname, abspath in members:
        top = arcname.split("/")[0] if "/" in arcname else "(root files)"
        count, size = tops.get(top, (0, 0))
        tops[top] = (count + 1, size + os.path.getsize(abspath))

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



def audit(out_path):
    """Audit the archive we just built, via the companion checker.

    The checker is imported rather than shelled out to so that the audit runs
    under the same interpreter that built the ZIP -- a subprocess would have
    to guess one, and guessing wrong turns a failed audit into a skipped one.
    """
    sys.path.insert(0, HERE)
    import check_submission_zip as checker

    print("auditing " + out_path + " -- check_submission_zip.py, "
          "artifact mode")
    print()
    with tempfile.TemporaryDirectory(prefix="submission-audit-") as tmp:
        try:
            checker.extract_zip(out_path, tmp)
        except zipfile.BadZipFile as exc:
            print("[ARTIFACT] [FAIL] ZIP is unreadable: " + str(exc))
            return 1
        return checker.run_audit(tmp, "ARTIFACT", artifact_mode=True)


def main():
    ap = argparse.ArgumentParser(
        description="Build the Phase 2 submission ZIP from an explicit "
                    "allow-list, then audit it. See the module docstring.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join("dist", "submission.zip"),
                    help="output ZIP path (default: dist/submission.zip)")
    ap.add_argument("--no-audit", action="store_true",
                    help="skip the automatic artifact audit of the built ZIP")
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

    rc = build(args.out)
    if rc or args.no_audit:
        return rc
    return audit(args.out)


if __name__ == "__main__":
    sys.exit(main())

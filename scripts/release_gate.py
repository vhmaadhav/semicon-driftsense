#!/usr/bin/env python3
"""One command that decides whether this tree may be released.

    python scripts/release_gate.py

Exit 0 means: the artifact this tree produces is buildable, reproducible,
free of anything that must not leave the private repository, and passes the
full artifact audit. Anything else exits non-zero and says which gate failed.

This exists because the repository already had every individual control and
still shipped a tree whose ZIP could not be built. `driftsense/vst.py` landed
in one PR; the closed ship manifest that has to name it was authored in
another; each was green alone and red together, and nothing ran the builder
between the merge and the discovery. The controls were fine. Nothing
*composed* them.

Five gates, ordered so the cheapest failure to diagnose comes first:

1. **Clean worktree.** A number measured against uncommitted edits cannot be
   reproduced by anyone, including its author. The gate records the HEAD SHA
   so the provenance record it writes is addressable.
2. **Import closure vs ship manifest** (`build_submission_zip.check_driftsense`).
   The direction that matters: a module the graded entry points can reach and
   the manifest omits produces an artifact that fails on import.
3. **Reproducible build.** Two builds, byte-identical SHA-256. A build that
   varies run to run cannot be audited, because the thing checked is not
   provably the thing shipped.
4. **Forbidden content, read back out of the ZIP.** `build_submission_zip`
   enforces its DENY list while writing; nothing re-read the finished archive
   to confirm it held. This gate does, because the builder is precisely the
   component whose bug this would be. `.agents/` is the one that matters: it
   carries a deck transcribed from material marked "Applied Materials
   Confidential" and an organizer SharePoint capability URL.
5. **Full artifact audit** (`check_submission_zip.py`), delegated unchanged.

Nothing here re-implements a check that already exists elsewhere.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)

# The assertion side of build_submission_zip.DENY -- stated as prefixes read
# back out of the finished ZIP, so a builder bug cannot conceal itself.
FORBIDDEN_PREFIXES = (
    ".agents/",      # confidential deck transcript + SharePoint capability URL
    "phase1/",       # archived Phase 1 duplicate
    ".git/",
    ".github/",
    "scripts/",      # development tooling, including this file
    "data/",
    "results/",
    "venv/",
)
FORBIDDEN_FILES = ("train.py", "evaluate.py", ".coderabbit.yaml")

# The single checkpoint the submission is allowed to carry.
ALLOWED_WEIGHT = "weights/driftsense.pt"

# A Git-LFS pointer is a ~130-byte text stub. Shipping one means the artifact
# carries a promise to fetch bytes over a network the graded machine does not
# have.
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

results = []


def gate(name, ok, detail=""):
    results.append({"name": name, "ok": bool(ok), "detail": detail})
    print("[GATE] [" + ("PASS" if ok else "FAIL") + "] " + name
          + (("  (" + detail + ")") if detail else ""))
    return bool(ok)


def sha256_file(path):
    digest = hashlib.sha256()
    with io.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    """Trailing newline only. `--porcelain` encodes status in the first two
    columns, so a leading space is data -- stripping it shifts every path by
    one character."""
    proc = subprocess.run(["git"] + list(args), cwd=REPO,
                          capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "").rstrip("\r\n")


def porcelain_paths(status):
    """Paths from `git status --porcelain`, handling renames and quoting."""
    paths = []
    for line in (status or "").splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip('"'))
    return paths


def gate_clean_worktree(allow_dirty=False):
    code, out = git("status", "--porcelain")
    if code != 0:
        return gate("clean worktree", False,
                    "git status failed; not a checkout?")
    dirty = porcelain_paths(out)
    if dirty and allow_dirty:
        return gate("clean worktree", True,
                    "DIRTY (" + str(len(dirty)) + " path(s)) -- allowed by "
                    "--allow-dirty; this run is NOT release evidence")
    return gate("clean worktree", not dirty,
                "clean" if not dirty else
                str(len(dirty)) + " uncommitted path(s): "
                + ", ".join(dirty[:5])
                + ("..." if len(dirty) > 5 else ""))


def gate_import_closure():
    import build_submission_zip as builder
    problems = builder.check_driftsense(REPO)
    return gate("import closure matches the ship manifest", not problems,
                "every reachable driftsense module is in DRIFTSENSE_SHIP"
                if not problems else "; ".join(problems))


def gate_reproducible_build(workdir):
    import build_submission_zip as builder
    paths = []
    for n in (1, 2):
        out = os.path.join(workdir, "build" + str(n) + ".zip")
        code = builder.build(out, REPO)
        if code != 0 or not os.path.isfile(out):
            gate("reproducible build", False,
                 "build " + str(n) + " failed (exit " + str(code) + ")")
            return None
        paths.append(out)
    digests = [sha256_file(p) for p in paths]
    ok = digests[0] == digests[1]
    gate("reproducible build (two builds, identical SHA-256)", ok,
         digests[0][:16] + "..." if ok else
         "build 1 " + digests[0][:16] + "... != build 2 "
         + digests[1][:16] + "...")
    return paths[0] if ok else None


def gate_forbidden_content(zip_path):
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        infos = dict((i.filename, i) for i in archive.infolist())

        leaked = sorted(
            n for n in names
            if n.startswith(FORBIDDEN_PREFIXES)
            or n in FORBIDDEN_FILES
            or ".agents" in n.split("/"))
        ok_paths = gate(
            "no denied path in the artifact", not leaked,
            "none of " + ", ".join(FORBIDDEN_PREFIXES + FORBIDDEN_FILES)
            if not leaked else
            str(len(leaked)) + " leaked: " + ", ".join(leaked[:8]))

        checkpoints = sorted(n for n in names
                             if n.endswith((".pt", ".pth", ".ckpt")))
        ok_ckpt = gate("exactly one checkpoint, and it is the shipped one",
                       checkpoints == [ALLOWED_WEIGHT],
                       ALLOWED_WEIGHT if checkpoints == [ALLOWED_WEIGHT]
                       else "found " + repr(checkpoints))

        pointers = []
        for name in names:
            info = infos.get(name)
            if info is None or info.is_dir() or info.file_size > 1024:
                continue
            with archive.open(name) as handle:
                if handle.read(len(LFS_MAGIC)) == LFS_MAGIC:
                    pointers.append(name)
        ok_lfs = gate("no Git-LFS pointer files", not pointers,
                      "none" if not pointers else ", ".join(pointers))

    return ok_paths and ok_ckpt and ok_lfs


def gate_artifact_audit(zip_path):
    proc = subprocess.run(
        [sys.executable,
         os.path.join(HERE, "check_submission_zip.py"), zip_path],
        cwd=REPO, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    if proc.stderr.strip():
        sys.stderr.write(proc.stderr)
    return gate("artifact audit (check_submission_zip.py)",
                proc.returncode == 0,
                "exit 0" if proc.returncode == 0
                else "exit " + str(proc.returncode))


def write_provenance(path, zip_path):
    _, sha = git("rev-parse", "HEAD")
    _, branch = git("rev-parse", "--abbrev-ref", "HEAD")
    _, dirty = git("status", "--porcelain")
    sha, branch = sha.strip(), branch.strip()
    record = {
        "commit": sha,
        "branch": branch,
        "dirty": len(porcelain_paths(dirty)),
        "zip_sha256": sha256_file(zip_path) if zip_path else None,
        "zip_bytes": os.path.getsize(zip_path) if zip_path else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "gates": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("\nprovenance -> " + path)
    return record


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="do not fail on an unclean worktree. Local "
                             "iteration only -- a run with this flag is not "
                             "release evidence, and the provenance record "
                             "says so.")
    parser.add_argument("--keep", metavar="PATH",
                        help="also write the verified ZIP here")
    parser.add_argument("--provenance", metavar="PATH",
                        default=os.path.join(REPO, "dist",
                                             "release_provenance.json"),
                        help="where to write the provenance record "
                             "(default %(default)s)")
    args = parser.parse_args()

    print("RELEASE GATE  " + REPO + "\n")
    ok = gate_clean_worktree(args.allow_dirty)
    ok = gate_import_closure() and ok

    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        zip_path = gate_reproducible_build(tmp)
        if zip_path is None:
            ok = False
        else:
            ok = gate_forbidden_content(zip_path) and ok
            ok = gate_artifact_audit(zip_path) and ok
            if args.keep:
                os.makedirs(
                    os.path.dirname(os.path.abspath(args.keep)) or ".",
                    exist_ok=True)
                with io.open(zip_path, "rb") as src:
                    data = src.read()
                with io.open(args.keep, "wb") as dst:
                    dst.write(data)
                print("\nkept -> " + args.keep)
        write_provenance(args.provenance, zip_path)

    failed = [r["name"] for r in results if not r["ok"]]
    print("\nRELEASE GATE: "
          + ("PASS" if ok and not failed else "FAIL -- " + "; ".join(failed)))
    return 0 if (ok and not failed) else 1


if __name__ == "__main__":
    sys.exit(main())

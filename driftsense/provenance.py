"""Everything needed to re-run a measurement -- or to know that you cannot.

A number without provenance is not evidence. PR #30 is the standing example:
its 2,500-pair headline run exists, but `.agents/G2_FULL_RERUN.log` records
`dirty: 2 files`, so the run cannot be reproduced from the SHA it names and
the whole campaign is blocked behind redoing it. That was caught by hand, in a
log, after the fact. Nothing in `scripts/eval_ext.py`, `scripts/eval_phase2.py`
or `scripts/grade_emulation.py` recorded a commit, a dirty flag, a seed or an
environment at all.

This module is the structured version of that note, emitted by the evaluator
itself at the moment of measurement:

    from driftsense import provenance
    record = provenance.record(split=args.split, weights=weights_path)
    provenance.require_clean(record)            # refuses a dirty tree
    provenance.write(os.path.join(args.split, "provenance.json"), record)

`require_clean` is the part that matters. A dirty-tree measurement is not
"slightly worse evidence" -- it is unreproducible, and the cost of discovering
that is paid weeks later by whoever has to re-run it on the machine that holds
the data. Failing at the start of the run is cheaper than failing after it.

It lives in `driftsense/` rather than `scripts/` because the evaluators are
scripts and this is a library. It is NOT reachable from register.py, infer.py
or generate_dataset.py, so `build_submission_zip.check_driftsense` does not
require it in DRIFTSENSE_SHIP and it correctly does not ship -- the same
arrangement `driftsense/rubric.py` uses. `tests/test_import_closure_parity.py`
pins that invariant.
"""
from __future__ import annotations

import datetime
import hashlib
import io
import json
import os
import platform
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _git(*args):
    """Trailing newline only. `--porcelain` encodes status in the first two
    columns, so a leading space is data: stripping it shifts every path by one
    character and quietly mangles the record this module exists to make
    trustworthy."""
    try:
        proc = subprocess.run(["git"] + list(args), cwd=REPO,
                              capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "").rstrip("\r\n")


def _git_value(*args):
    """A single-line git value, e.g. a SHA or a branch name."""
    out = _git(*args)
    return out.strip() if out is not None else None


def _porcelain_paths(status):
    """Paths from `git status --porcelain`, tolerating quoted and renamed
    entries. Format is `XY<space>PATH`, and `R` renames read `OLD -> NEW`."""
    paths = []
    for line in (status or "").splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if " -> " in path:                        # rename: keep the new name
            path = path.split(" -> ", 1)[1]
        paths.append(path.strip('"'))
    return paths


def sha256(path, limit=None):
    """Hex digest of a file, or None when it is absent/unreadable.

    `limit` caps how many bytes are read, for manifests large enough that
    hashing the whole file would dominate a short run. A capped digest is
    recorded as such so it is never mistaken for a whole-file hash.
    """
    if not path or not os.path.isfile(path):
        return None
    digest, read = hashlib.sha256(), 0
    try:
        with io.open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1048576), b""):
                if limit is not None and read + len(block) > limit:
                    digest.update(block[:limit - read])
                    return digest.hexdigest() + "(first %d bytes)" % limit
                digest.update(block)
                read += len(block)
    except OSError:
        return None
    return digest.hexdigest()


def _versions():
    out = {"python": platform.python_version(),
           "platform": platform.platform()}
    for name in ("torch", "numpy", "cv2", "pandas"):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception:                        # noqa: BLE001
            out[name] = None
    return out


def _threads():
    """The thread caps actually in force, read back rather than assumed.

    Quoting a runtime without this is how a 10-core development box comes to
    stand in for a 4-core reference machine (register.py cap_threads).
    """
    out = {}
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "KMP_DUPLICATE_LIB_OK"):
        out[key] = os.environ.get(key)
    try:
        import torch
        out["torch_num_threads"] = torch.get_num_threads()
        out["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:                            # noqa: BLE001
        pass
    try:
        import cv2
        out["cv2_num_threads"] = cv2.getNumThreads()
    except Exception:                            # noqa: BLE001
        pass
    out["cpu_count"] = os.cpu_count()
    return out


def record(**extra):
    """A provenance record for the measurement about to be taken.

    Any keyword is merged in verbatim, so a caller adds what only it knows --
    the split, the seed, the threshold, the decode config. Paths handed in as
    `weights` or `manifest` are additionally hashed.
    """
    dirty_paths = _porcelain_paths(_git("status", "--porcelain"))
    out = {
        "recorded_utc": datetime.datetime.now(
            datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": len(dirty_paths),
        "dirty_paths": dirty_paths[:20],
        "command": " ".join(sys.argv),
        "versions": _versions(),
        "threads": _threads(),
    }
    out.update(extra)
    if extra.get("weights"):
        out["weights_sha256"] = sha256(extra["weights"])
    if extra.get("manifest"):
        out["manifest_sha256"] = sha256(extra["manifest"], limit=64 * 1048576)
    return out


def require_clean(record_dict, allow_dirty=False):
    """Refuse to measure against an unreproducible tree.

    Set `allow_dirty` (or DRIFTSENSE_ALLOW_DIRTY=1) for local iteration; the
    record then carries `allow_dirty: true` so the resulting number can never
    be mistaken for evidence.
    """
    allow = allow_dirty or os.environ.get("DRIFTSENSE_ALLOW_DIRTY") == "1"
    if not record_dict.get("dirty"):
        return record_dict
    record_dict["allow_dirty"] = bool(allow)
    if allow:
        sys.stderr.write(
            "[provenance] WARNING: measuring against a DIRTY tree ("
            + str(record_dict["dirty"]) + " path(s)). This number is not "
            "reproducible from " + str(record_dict.get("commit"))[:12]
            + " and must not be quoted as evidence.\n")
        return record_dict
    raise SystemExit(
        "FATAL: refusing to measure against a dirty worktree -- "
        + str(record_dict["dirty"]) + " uncommitted path(s): "
        + ", ".join(record_dict.get("dirty_paths", [])[:5])
        + ".\nA number recorded here cannot be reproduced from the commit it "
          "names; that is the exact failure that blocked PR #30 (its headline "
          "2,500-pair run recorded 'dirty: 2 files').\nCommit or stash first, "
          "or pass --allow-dirty / DRIFTSENSE_ALLOW_DIRTY=1 for a run you will "
          "not quote.")


def write(path, record_dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(record_dict, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")
    return path

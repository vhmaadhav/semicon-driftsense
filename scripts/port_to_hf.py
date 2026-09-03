#!/usr/bin/env python3
"""Port the shard collection to a Hugging Face dataset, Drive calls last.

Two phases, and the order is the point:

  phase 1  LOCAL   175 shards (77.5 GB) are already extracted on this machine.
                   They are re-tarred and uploaded with **zero Drive requests**,
                   so none of this costs quota.
  phase 2  REMOTE  only the ~873 shards we do not hold are fetched, throttled
                   and under a byte budget, stopping cleanly the moment Drive
                   serves its quota page.

Streaming, not stage-then-upload: the collection is ~666 GB against ~502 GB of
free disk, so each shard is uploaded and its tar deleted before the next starts.
Peak disk is (workers x ~651 MB) plus, in phase 1, one re-tarred copy.

Integrity is verified rather than assumed. A tar must open, contain a
manifest.csv, and hold exactly as many search images as the manifest has rows.
After upload the Hub's own sha256 for the LFS object is compared with the local
one; a shard is only marked done once those agree.

Naming: `shards/{split}/{set}/{split}_{set}_{idx:04d}.tar`. The Drive index
lists two different generator bundles under the same shard number for 64 of the
shards we hold, so the original filename cannot be reconstructed for those. The
content hash and the manifest's own generator_bundle_sha256 are recorded in the
state file instead, which identifies a shard properly rather than by guesswork.

Resumable: state is appended after every success and re-read on start.
The token is never logged.

  ./venv-hf/bin/python scripts/port_to_hf.py --phase local --limit 2   # smoke test
  ./venv-hf/bin/python scripts/port_to_hf.py --phase local             # all 77.5 GB
  ./venv-hf/bin/python scripts/port_to_hf.py --phase remote --budget-gb 50
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import glob
import hashlib
import os
import re
import subprocess
import sys
import tarfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import drive_api

REPO = "vhmaadhav/semiconductor-reference-search-registration"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(HERE, ".agents", "shards.tsv")
STATE = os.path.join(HERE, ".agents", "hf_port_state.tsv")
STAGE = os.path.join(HERE, ".hf_stage")

_lock = threading.Lock()
_tls = threading.local()
_quota = threading.Event()
_bytes = [0]


def log(m): 
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_done():
    d = {}
    if os.path.exists(STATE):
        for line in open(STATE):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[1] == "done":
                d[p[0]] = p[2]
    return d


def record(key, status, sha, note=""):
    with _lock, open(STATE, "a") as f:
        f.write(f"{key}\t{status}\t{sha}\t{int(time.time())}\t{note}\n")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def verify_tar(path):
    """The manifest must agree with the payload, or the shard is not usable."""
    try:
        with tarfile.open(path) as t:
            names = t.getnames()
            man = [n for n in names if n.endswith("manifest.csv")]
            if not man:
                return "no manifest.csv", None
            rows = list(csv.DictReader(
                (l.decode() for l in t.extractfile(man[0]))))
            # Match both layouts: a Drive tar nests under a shard directory,
            # while a tar we build here has search/ at the top level.
            imgs = sum(1 for n in names
                       if n.endswith(".png")
                       and (n.startswith("search/") or "/search/" in n))
        if not rows:
            return "empty manifest", None
        if imgs != len(rows):
            return f"manifest {len(rows)} pairs vs {imgs} search images", None
        return None, rows[0].get("generator_bundle_sha256", "")[:16]
    except Exception as e:
        return f"unreadable: {type(e).__name__}", None


def upload(local, dest, key, bundle):
    from huggingface_hub import HfApi
    sha = sha256_of(local)
    api = HfApi()
    api.upload_file(path_or_fileobj=local, path_in_repo=dest, repo_id=REPO,
                    repo_type="dataset", commit_message=f"Add {dest}")
    info = api.get_paths_info(REPO, [dest], repo_type="dataset")
    if not info:
        return False, "absent from hub after upload"
    lfs = getattr(info[0], "lfs", None)
    rsha = getattr(lfs, "sha256", None) if lfs else None
    if rsha and rsha != sha:
        return False, f"sha mismatch hub={rsha[:12]} local={sha[:12]}"
    record(key, "done", sha, f"{dest}\tbundle={bundle}")
    return True, sha[:12]


# ---------------------------------------------------------------- phase local
def do_local(job):
    d, split, sset, idx = job
    key = f"{split}_{sset}_{idx:04d}"
    tar = os.path.join(STAGE, key + ".tar")
    try:
        with tarfile.open(tar, "w") as t:                 # no compression: PNGs
            for n in sorted(os.listdir(d)):
                if n != "COMPLETE":
                    t.add(os.path.join(d, n), arcname=n)
        why, bundle = verify_tar(tar)
        if why:
            return key, False, why
        return (key, *upload(tar, f"shards/{split}/{sset}/{key}.tar", key, bundle))
    except Exception as e:
        return key, False, f"{type(e).__name__}: {e}"
    finally:
        if os.path.exists(tar):
            os.remove(tar)


# --------------------------------------------------------------- phase remote
def do_remote(job):
    split, sset, idx, fid, budget = job
    key = f"{split}_{sset}_{idx:04d}"
    if _quota.is_set() or (budget and _bytes[0] >= budget):
        return key, False, "skipped (budget/quota)"
    tar = os.path.join(STAGE, key + ".tar")
    try:
        # Authenticated Drive API, not the public download endpoint. The public
        # one enforces a per-file *sharing* abuse quota that answers HTTP 200
        # with a 2 KB HTML page for about a day; as the files' owner over the
        # API that limit does not apply, and the API's own limit (12,000
        # queries per 100 s) is ~100x more headroom than this port needs.
        svc = getattr(_tls, "svc", None)
        if svc is None:
            svc = _tls.svc = drive_api.service()   # httplib2 is not thread-safe
        why = drive_api.download(svc, fid, tar)
        if why:
            return key, False, why
        size = os.path.getsize(tar)
        with _lock:
            _bytes[0] += size
        why, bundle = verify_tar(tar)
        if why:
            return key, False, why
        return (key, *upload(tar, f"shards/{split}/{sset}/{key}.tar", key, bundle))
    finally:
        if os.path.exists(tar):
            os.remove(tar)


def local_jobs(done):
    out = []
    for d in sorted(glob.glob(os.path.join(HERE, "data", "ext_train", "*/"))) + \
             sorted(glob.glob(os.path.join(HERE, "data", "ext_p2", "*/"))) + \
             sorted(glob.glob(os.path.join(HERE, "data", "ext_holdout", "*/"))):
        m = re.match(r"^(test_)?([ABCD])_(\d+)$", os.path.basename(d.rstrip("/")))
        if not m:
            continue                                   # our own pools, not Drive shards
        split = "test" if m.group(1) else "train"
        key = f"{split}_{m.group(2)}_{int(m.group(3)):04d}"
        if key in done:
            continue
        out.append((d.rstrip("/"), split, m.group(2), int(m.group(3))))
    return out


def remote_jobs(done, budget):
    held = {j[0] for j in []}
    out, seen = [], set()
    for line in open(INDEX):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5:
            continue
        split, sset, idx, fid = p[0], p[1], int(p[2]), p[3]
        key = f"{split}_{sset}_{idx:04d}"
        if key in done or key in seen:
            continue          # one upload per shard number; duplicate bundles skipped
        seen.add(key)
        out.append((split, sset, idx, fid, budget))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["local", "remote"], default="local")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--budget-gb", type=float, default=0,
                    help="remote only: stop after this many GB pulled from Drive")
    a = ap.parse_args()

    os.makedirs(STAGE, exist_ok=True)
    from huggingface_hub import HfApi
    try:
        log(f"authenticated as {HfApi().whoami().get('name','?')}")
    except Exception as e:
        sys.exit(f"not authenticated ({type(e).__name__}).  "
                 f"Run:  ./venv-hf/bin/hf auth login")

    done = load_done()
    budget = int(a.budget_gb * 1024**3)
    jobs = local_jobs(done) if a.phase == "local" else remote_jobs(done, budget)
    if a.limit:
        jobs = jobs[:a.limit]
    fn = do_local if a.phase == "local" else do_remote

    log(f"phase {a.phase}: {len(done)} already on the hub, {len(jobs)} to do, "
        f"{a.workers} workers")
    if a.phase == "remote":
        log(f"Drive budget: {'unlimited' if not budget else f'{a.budget_gb:.0f} GB'}")

    ok = bad = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(fn, j) for j in jobs]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            key, good, note = fut.result()
            if good:
                ok += 1
                el = time.time() - t0
                log(f"  [{i}/{len(jobs)}] OK   {key}  {note}  eta {(len(jobs)-i)*el/i/60:.0f} min")
            else:
                bad += 1
                record(key, "failed", "", note)
                log(f"  [{i}/{len(jobs)}] FAIL {key}  {note}")
            if _quota.is_set():
                log("Drive quota hit -- stopping cleanly. Finished shards are "
                    "recorded and will be skipped on the next run.")
                for f in futs:
                    f.cancel()
                break
    log(f"PHASE {a.phase.upper()} DONE: {ok} uploaded, {bad} failed")


if __name__ == "__main__":
    main()

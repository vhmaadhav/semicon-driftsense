#!/usr/bin/env python3
"""Check the Hugging Face dataset against what we believe we uploaded.

Run after scripts/port_to_hf.py. It answers three questions independently,
because "the upload script said OK" is not evidence:

  1. Is every shard the local state file calls done actually present on the Hub,
     with the sha256 the Hub itself reports matching what we recorded?
  2. Is anything on the Hub that is *not* in our state file (a partial or
     duplicated upload from an interrupted run)?
  3. How much of the Drive index is still missing?

Exit status is non-zero if anything is missing or mismatched, so it can gate a
cleanup step.

  ./venv-hf/bin/python scripts/verify_hf_port.py
  ./venv-hf/bin/python scripts/verify_hf_port.py --deep    # also re-list sizes
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "vhmaadhav/semiconductor-reference-search-registration"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INDEX = os.path.join(HERE, ".agents", "shards.tsv")
STATE = os.path.join(HERE, ".agents", "hf_port_state.tsv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    a = ap.parse_args()

    from huggingface_hub import HfApi
    api = HfApi()
    try:
        api.whoami()
    except Exception as e:
        sys.exit(f"not authenticated ({type(e).__name__}). Run: ./venv-hf/bin/hf auth login")

    claimed = {}
    if os.path.exists(STATE):
        for line in open(STATE):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[1] == "done":
                claimed[p[0]] = p[2]

    files = [f for f in api.list_repo_files(REPO, repo_type="dataset")
             if f.startswith("shards/") and f.endswith(".tar")]
    on_hub = {os.path.basename(f)[:-4]: f for f in files}

    want = set()
    for line in open(INDEX):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 5:
            want.add(f"{p[0]}_{p[1]}_{int(p[2]):04d}")

    missing_on_hub = sorted(set(claimed) - set(on_hub))
    orphan_on_hub = sorted(set(on_hub) - set(claimed))
    not_ported = sorted(want - set(on_hub))

    print(f"Drive index          : {len(want)} shards")
    print(f"state file says done : {len(claimed)}")
    print(f"actually on the hub  : {len(on_hub)}")
    print(f"still to port        : {len(not_ported)}")

    bad_sha = []
    if claimed:
        # Ask the Hub for its own hashes rather than trusting our record of them.
        paths = [on_hub[k] for k in claimed if k in on_hub]
        for i in range(0, len(paths), 200):
            for info in api.get_paths_info(REPO, paths[i:i + 200], repo_type="dataset"):
                key = os.path.basename(info.path)[:-4]
                lfs = getattr(info, "lfs", None)
                rsha = getattr(lfs, "sha256", None) if lfs else None
                if rsha and claimed.get(key) and rsha != claimed[key]:
                    bad_sha.append((key, claimed[key][:12], rsha[:12]))
                if a.deep:
                    print(f"   {key:<22} {getattr(info,'size',0)/1024**2:8.1f} MB  {(rsha or '')[:12]}")

    ok = True
    if missing_on_hub:
        ok = False
        print(f"\nMISSING: {len(missing_on_hub)} shards recorded done but absent from the hub")
        for k in missing_on_hub[:10]:
            print(f"   {k}")
    if bad_sha:
        ok = False
        print(f"\nSHA MISMATCH: {len(bad_sha)}")
        for k, l, r in bad_sha[:10]:
            print(f"   {k}  local {l}  hub {r}")
    if orphan_on_hub:
        print(f"\nnote: {len(orphan_on_hub)} files on the hub are not in the state file "
              f"(fine if uploaded from elsewhere; suspicious after an interrupted run)")
        for k in orphan_on_hub[:10]:
            print(f"   {k}")

    print("\nVERIFIED: every recorded shard is on the hub with a matching hash"
          if ok else "\nVERIFICATION FAILED -- see above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

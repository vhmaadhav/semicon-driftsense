#!/usr/bin/env python3
"""Pull more Set C training shards over the authenticated Drive API.

Set C is the absent-pair half, and it feeds the only bonus still in reach:
rejection F1 is 0.8893 against a 0.90 threshold worth +4. Set C is 16.8% of the
training pool and we hold 28 of the 198 shards that exist.

Uses the Drive API as the files' owner, not the public download endpoint. The
public one enforces a per-file sharing abuse cap that answers HTTP 200 with a
2 KB HTML page for about a day; the API's limit is 12,000 queries per 100 s,
which this comes nowhere near.

Downloads to a .part file and only renames on success, so an interrupted run
cannot leave a truncated tar that the next pass mistakes for complete.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="C")
    ap.add_argument("--count", type=int, default=60)
    ap.add_argument("--dest", default="data/ext_train")
    a = ap.parse_args()

    import drive_api
    svc = drive_api.service()
    dest = os.path.join(HERE, a.dest)
    os.makedirs(dest, exist_ok=True)

    want = []
    for line in open(os.path.join(HERE, ".agents", "shards.tsv")):
        p = line.rstrip("\n").split("\t")
        if len(p) < 5 or p[0] != "train" or p[1] != a.set:
            continue
        d = os.path.join(dest, f"{a.set}_{int(p[2]):04d}")
        if not os.path.isdir(d):
            want.append((int(p[2]), p[3], d))
    want = want[:a.count]
    print(f"{len(want)} set {a.set} shards to fetch", flush=True)

    ok = 0
    for i, (idx, fid, d) in enumerate(want, 1):
        tar = d + ".tar"
        why = drive_api.download(svc, fid, tar, chunk_mb=32)
        if why:
            print(f"  [{i}/{len(want)}] FAIL {a.set}_{idx:04d}: {why}", flush=True)
            continue
        os.makedirs(d, exist_ok=True)
        try:
            with tarfile.open(tar) as t:
                t.extractall(d)
        except Exception as e:
            print(f"  [{i}/{len(want)}] FAIL extract {a.set}_{idx:04d}: {e}", flush=True)
            subprocess.run(["rm", "-rf", d, tar]); continue
        os.remove(tar)
        man = os.path.join(d, "manifest.csv")
        n_rows = (sum(1 for _ in open(man)) - 1) if os.path.exists(man) else 0
        n_img = len(os.listdir(os.path.join(d, "search"))) if os.path.isdir(os.path.join(d, "search")) else 0
        if n_rows > 0 and n_rows == n_img:
            open(os.path.join(d, "COMPLETE"), "w").write(f"{n_rows} pairs\n")
            ok += 1
            print(f"  [{i}/{len(want)}] ok {a.set}_{idx:04d}  {n_rows} pairs", flush=True)
        else:
            subprocess.run(["rm", "-rf", d])
            print(f"  [{i}/{len(want)}] discarded {a.set}_{idx:04d} "
                  f"({n_img}/{n_rows})", flush=True)
    print(f"done: {ok}/{len(want)} shards added", flush=True)


if __name__ == "__main__":
    main()

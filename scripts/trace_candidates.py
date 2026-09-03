#!/usr/bin/env python3
"""Per-pair CANDIDATE TRACE: what did `pose_candidates` actually offer?

`eval_ext.py` scores the answer the decode returned. It cannot tell you *why*
a wrong answer was wrong, and for a candidate-generation change that
distinction is the whole question. A wrong tile has two very different causes:

* the near-GT pose basin was **never offered** -- a candidate-generation
  failure, which is what issue #37 (rotation-blind scale ranking) is about
  and what rotation-aware ranking can fix;
* the near-GT basin **was offered** and the native-ZNCC selector preferred a
  different hypothesis -- a selection failure (issue #5), which candidate
  generation cannot be credited or blamed for.

Without this trace a pair recovered by any candidate-side change could be
silently attributed to the wrong lever. Issue #37's A/B used it to establish
that rotation-aware ranking gained the true basin on 8 pairs and lost it on
zero, and that its three end-to-end regressions were all selector failures on
pairs whose basin was still offered (two of them ranked FIRST).

No network is loaded: `pose_candidates` needs only the two images, so this
pass is cheap enough to run over the full 2,250 in both arms of an A/B.

Mirrors the shipped decode: denoise=0 (so `search_corr` is `search`),
k=hypotheses=3, coarse_scales=17, band=SHIPPED_BAND.

  python scripts/trace_candidates.py data/ext_p2/*  --out .agents/cand.csv

Per-pair timings here are collected under process parallelism and are NOT
valid for the efficiency component; measure that with scripts/profile_pair.py,
single process at 4 threads.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Near-GT basin tolerances on the REFINED candidate. Three levels are reported
# rather than one so the recall figure cannot be read as a threshold chosen to
# flatter a result. `basin` is the primary: a hypothesis this close is in the
# right basin, and the refine/polish stages routinely carry it to the credit
# tiers from there.
TOLS = {
    # The rubric's top pose tier is scale <=1% AND rotation <=0.25 deg (the
    # 1.00-credit row); 0.5 deg is the SECOND rotation tier (0.60 credit),
    # not the top one -- do not relabel this back without re-checking the
    # published credit table.
    "tight": (0.01, 0.25),
    "basin": (0.02, 1.0),    # primary
    "loose": (0.05, 2.0),    # generous, still unambiguously not a wrong tile
}


def _assert_loaded_from(module, root: str) -> None:
    """Fail loudly if `module` did not actually load from under `root`.

    Import order is what silently broke --root before: once a package is in
    sys.modules, a later sys.path.insert(0, ...) does not move it. Every
    driftsense/infer import in this script is guarded by this check instead
    of trusting sys.path alone.
    """
    loaded_from = os.path.abspath(getattr(module, "__file__", ""))
    root = os.path.abspath(root)
    if os.path.commonpath([loaded_from, root]) != root:
        raise RuntimeError(
            f"{module.__name__} loaded from {loaded_from!r}, which is NOT "
            f"under the requested root {root!r} -- another copy of it is "
            "already cached in this process (check for a stray import "
            "earlier in the module, or PYTHONPATH). Refusing to silently "
            "trace the wrong tree.")


def _resolve_root_config(root: str):
    """Import driftsense.config from `root` specifically, verified.

    Must run before ANY other driftsense/infer import in this process --
    that is the whole fix. The old top-level `from driftsense.config import
    SHIPPED_BAND` ran before argparse had even parsed --root, on every
    process (the parent, and every forked/spawned worker), so it always
    loaded from HERE regardless of --root.
    """
    root = os.path.abspath(root)
    sys.path.insert(0, root)
    import driftsense.config as config
    _assert_loaded_from(config, root)
    return config.SHIPPED_BAND


def _worker(job):
    root, shard_dir, row, threads, k, coarse, band = job
    # Nothing under driftsense/infer may be imported above this line in this
    # file -- see _resolve_root_config's docstring.
    sys.path.insert(0, root)
    import torch
    torch.set_num_threads(threads)
    import cv2
    # OpenCV otherwise spawns a pool per *process*, so N workers oversubscribe
    # the machine N-fold.
    cv2.setNumThreads(threads)
    import infer as I
    import driftsense.matching as matching
    _assert_loaded_from(matching, root)
    pose_candidates = matching.pose_candidates

    ref = I.read_gray(os.path.join(shard_dir, row["reference_path"]))
    sea = I.read_gray(os.path.join(shard_dir, row["search_path"]))
    t0 = time.perf_counter()
    cands = pose_candidates(ref, sea, k=k, coarse_scales=coarse, band=band)
    dt = time.perf_counter() - t0

    gt_z, gt_r = float(row["magnification"]), float(row["rotation_deg"])
    out = {
        "pair_id": row["pair_id"], "set": row["phase2_set"],
        "severity": row.get("severity_level", -1),
        "architecture": row.get("architecture", ""),
        "gt_found": int(row["found"]), "gt_scale": gt_z, "gt_rot": gt_r,
        "n_cands": len(cands), "cand_secs": dt,
    }
    for i, (f, r, peak) in enumerate(cands):
        out[f"c{i}_scale"], out[f"c{i}_rot"], out[f"c{i}_peak"] = \
            float(f), float(r), float(peak)
    # Rank of the first hypothesis inside each tolerance; -1 = never offered.
    for name, (s_tol, r_tol) in TOLS.items():
        rank = -1
        for i, (f, r, _) in enumerate(cands):
            if abs(f - gt_z) / gt_z <= s_tol and abs(r - gt_r) <= r_tol:
                rank = i
                break
        out[f"rank_{name}"] = rank
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--root", default=HERE,
                    help="repo root whose driftsense/ is under test -- point "
                         "this at a second worktree to trace the other arm of "
                         "an A/B without swapping branches. Verified: the "
                         "loaded driftsense.config/matching modules must "
                         "actually resolve under this root, or the run "
                         "aborts rather than silently tracing the wrong tree.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hypotheses", type=int, default=3)
    ap.add_argument("--coarse-scales", type=int, default=17)
    ap.add_argument("--band", action="store_true",
                    help="opt in to the DoG band pre-filter; off by default, "
                         "which is SHIPPED_BAND")
    a = ap.parse_args()

    # Resolve --root and SHIPPED_BAND from it BEFORE any other driftsense
    # import in this process, in the parent -- see _resolve_root_config.
    shipped_band = _resolve_root_config(a.root)

    import multiprocessing as mp
    band = True if a.band else shipped_band
    tasks = []
    for d in a.shards:
        man = pd.read_csv(os.path.join(d, "manifest.csv"))
        for _, r in man.iterrows():
            tasks.append((a.root, d, r.to_dict(), a.threads,
                          a.hypotheses, a.coarse_scales, band))
    print(f"{len(tasks)} pairs, root={a.root}, {a.jobs} workers", flush=True)

    rows, t0 = [], time.perf_counter()
    with mp.Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            rows.append(r)
            if i % 250 == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(tasks)}  {el/i:.2f}s/pair wall  "
                      f"eta {(len(tasks)-i)*el/i/60:.1f} min", flush=True)
    df = pd.DataFrame(rows).sort_values("pair_id")
    df.to_csv(a.out, index=False)
    print(f"wrote {a.out}  ({len(df)} rows)")
    report(df, a.hypotheses)


def report(df, k):
    pres = df[df.gt_found == 1]
    print(f"\ncandidate-recall on {len(pres)} present pairs (k={k}):")
    for name in TOLS:
        rk = pres[f"rank_{name}"]
        line = "  ".join(f"@{K} {100*((rk >= 0) & (rk < K)).mean():5.2f}%"
                         for K in range(1, k + 1))
        print(f"  {name:<6} {line}   never-offered "
              f"{int((rk < 0).sum())}/{len(pres)}")
    for s in ("A", "B"):
        q = pres[pres["set"] == s]
        if len(q):
            rk = q["rank_basin"]
            print(f"  set {s}: recall@{k} {100*(rk >= 0).mean():5.2f}%  "
                  f"never-offered {int((rk < 0).sum())}/{len(q)}")
    print(f"\ncandidate-generation seconds (PARALLEL -- not the efficiency "
          f"number): median {df.cand_secs.median():.3f}  "
          f"p90 {df.cand_secs.quantile(.9):.3f}")


if __name__ == "__main__":
    main()

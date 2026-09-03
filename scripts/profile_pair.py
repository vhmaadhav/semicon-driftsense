#!/usr/bin/env python3
"""Where does a Phase 2 pair's wall-clock actually go?

Single process, fixed thread count, on an idle machine -- the only conditions
under which the efficiency component (5 pts) and the 20 s hard timeout mean
anything. Reports the total and the share taken by each stage, so an
optimisation targets the stage that is actually expensive rather than the one
that looks expensive.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shard")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "driftsense.pt"))
    ap.add_argument("--hypotheses", type=int, default=3)
    a = ap.parse_args()

    import torch
    torch.set_num_threads(a.threads)
    cv2.setNumThreads(a.threads)
    import infer as I
    import driftsense.matching as M

    # Instrument the stages by wrapping them.
    acc = defaultdict(float)

    def timed(name, fn):
        def wrapper(*args, **kw):
            t = time.perf_counter()
            try:
                return fn(*args, **kw)
            finally:
                acc[name] += time.perf_counter() - t
        return wrapper

    M.pose_candidates = timed("pose_candidates", M.pose_candidates)
    M.polish_pose = timed("polish_pose", M.polish_pose)
    M.refine_zncc = timed("refine_zncc", M.refine_zncc)
    M.locate = timed("locate (network)", M.locate)
    M.canonicalize_search = timed("canonicalize", M.canonicalize_search)

    model, device = I.load_model(a.weights)
    man = pd.read_csv(os.path.join(a.shard, "manifest.csv")).head(a.n)

    times = []
    for _, r in man.iterrows():
        ref = I.read_gray(os.path.join(a.shard, r.reference_path))
        sea = I.read_gray(os.path.join(a.shard, r.search_path))
        t0 = time.perf_counter()
        M.locate_phase2(model, ref, sea, device, refine=True, hypotheses=a.hypotheses)
        times.append(time.perf_counter() - t0)

    t = np.array(times)
    print(f"\n{a.shard}   n={len(t)}  threads={a.threads}  hypotheses={a.hypotheses}")
    print(f"  median {np.median(t):.2f}s   mean {t.mean():.2f}s   "
          f"p90 {np.percentile(t,90):.2f}s   max {t.max():.2f}s")
    print(f"  budget: median<=5s {'OK' if np.median(t)<=5 else 'FAIL'}   "
          f"max<20s {'OK' if t.max()<20 else 'FAIL'}")
    total = t.sum()
    print(f"\n  stage breakdown (share of {total:.1f}s total):")
    for k, v in sorted(acc.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22}{v:8.2f}s  {100*v/total:5.1f}%")
    other = total - sum(acc.values()) + acc.get("locate (network)", 0) * 0
    print(f"    {'(unattributed)':<22}{total-sum(v for k,v in acc.items() if k!='locate (network)')-acc.get('locate (network)',0):8.2f}s")


if __name__ == "__main__":
    main()

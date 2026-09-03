#!/usr/bin/env python3
"""How much of the Set B gap is the pose search, and how much is the network?

Runs each pair twice: once normally, and once with the *ground-truth* pose
handed to `locate_phase2`. The gap between them is the ceiling that fixing the
pose search could buy; whatever remains under the true pose is the network's
own error, and is the only part retraining can address.

This is the check that should precede any retrain. Run once before, not after,
committing a night of GPU time to it.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def _worker(job):
    shard, row, weights, threads = job
    import torch
    torch.set_num_threads(threads)
    import cv2
    cv2.setNumThreads(threads)
    import infer as I
    from driftsense.matching import locate_phase2

    global _M
    try:
        model, device = _M
    except NameError:
        _M = I.load_model(weights)
        model, device = _M

    ref = I.read_gray(os.path.join(shard, row["reference_path"]))
    sea = I.read_gray(os.path.join(shard, row["search_path"]))
    gx, gy = row["gt_x_corr"], row["gt_y_corr"]

    est = locate_phase2(model, ref, sea, device, refine=True)
    orc = locate_phase2(model, ref, sea, device, refine=True,
                        pose=(float(row["magnification"]), float(row["rotation_deg"])))
    return {
        "set": row["phase2_set"], "severity": row.get("severity_level", -1),
        "err_est": float(np.hypot(est["x"] - gx, est["y"] - gy)),
        "err_orc": float(np.hypot(orc["x"] - gx, orc["y"] - gy)),
        "s_err_est": abs(est["scale"] - row["magnification"]) / row["magnification"],
        "gt_scale": row["magnification"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "driftsense.pt"))
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--stride", type=int, default=10)
    a = ap.parse_args()

    import multiprocessing as mp
    tasks = []
    for d in a.shards:
        m = pd.read_csv(os.path.join(d, "manifest.csv")).iloc[::a.stride]
        for _, r in m.iterrows():
            if int(r["found"]) == 1:
                tasks.append((d, r.to_dict(), a.weights, a.threads))
    print(f"{len(tasks)} present pairs, {a.jobs} workers", flush=True)

    out, t0 = [], time.perf_counter()
    with mp.Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2), 1):
            out.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(tasks)}  {(time.perf_counter()-t0)/i:.2f}s/pair", flush=True)
    d = pd.DataFrame(out)

    print(f"\n{'set':<6}{'n':>5}{'est <=5px':>12}{'ORACLE <=5px':>15}{'gap':>8}"
          f"{'est med':>10}{'orc med':>10}")
    print("-" * 66)
    for s in sorted(d["set"].unique()):
        g = d[d["set"] == s]
        e, o = 100 * (g.err_est <= 5).mean(), 100 * (g.err_orc <= 5).mean()
        print(f"{s:<6}{len(g):>5}{e:>11.1f}%{o:>14.1f}%{o-e:>7.1f}{g.err_est.median():>10.2f}"
              f"{g.err_orc.median():>10.2f}")
    e, o = 100 * (d.err_est <= 5).mean(), 100 * (d.err_orc <= 5).mean()
    print(f"{'ALL':<6}{len(d):>5}{e:>11.1f}%{o:>14.1f}%{o-e:>7.1f}")

    print("\nOf the pairs that fail with the estimated pose:")
    f = d[d.err_est > 5]
    if len(f):
        fixed = (f.err_orc <= 5).mean()
        print(f"  {len(f)} failures; {100*fixed:.1f}% are fixed by the true pose alone.")
        print(f"  -> {100*fixed:.0f}% of the gap is the POSE SEARCH (fixable without retraining)")
        print(f"  -> {100*(1-fixed):.0f}% survives the true pose = the NETWORK's own error")
        print(f"  median scale error among failures: {100*f.s_err_est.median():.2f}%"
              f"   (successes: {100*d[d.err_est<=5].s_err_est.median():.2f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit the present/absent decision on features the pipeline already computes.

Rejection (15 pts) and calibration (10 pts) are both decided by one scalar per
pair, and the shipped scalar is `min(score, zncc)` -- two of the four signals
the pipeline produces, combined by a rule chosen by hand. This fits a small
logistic regression over all of them instead.

The features, and why each carries independent information:

* `score`     network peak confidence. Answers "which of these identical
              repeats is right". It is a *relative* judgement, so something
              always wins and it can be confident on a plausible decoy.
* `zncc`      full-resolution correlation at the chosen centre. Answers "is the
              reference really here". Degrades under heavy noise even when the
              answer is right.
* `peak_ratio` runner-up / winner among well-separated response peaks. A
              contested decision is weak evidence regardless of how high the
              winner scored.
* `pose_peak` coarse correlation of the winning pose hypothesis. Low when no
              pose explains the frame, which is the absent case.

Deliberately kept to a 5-parameter linear model on 4 features. The point is to
weight signals that already exist, not to learn a second detector, and a
logistic this small cannot memorise 20k pairs. Fitted on *training* shards and
scored on a held-out fold; the blind-set analogue is never touched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

FEATURES = ["score", "zncc", "peak_ratio", "pose_peak"]


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
    r = locate_phase2(model, ref, sea, device, refine=True)
    out = {f: float(r.get(f, 0.0)) for f in FEATURES}
    out["found"] = int(row["found"])
    if int(row["found"]) == 1:
        out["err"] = float(np.hypot(r["x"] - row["gt_x_corr"], r["y"] - row["gt_y_corr"]))
    else:
        out["err"] = np.nan
    return out


def extract(shards, weights, jobs, threads, per_shard, cache=None):
    import multiprocessing as mp
    tasks = []
    for d in shards:
        m = pd.read_csv(os.path.join(d, "manifest.csv"))
        if per_shard and len(m) > per_shard:
            m = m.iloc[:: max(len(m) // per_shard, 1)][:per_shard]
        for _, r in m.iterrows():
            tasks.append((d, r.to_dict(), weights, threads))
    # Shuffle before dispatch. Tasks are built shard by shard, so an
    # interrupted run would otherwise cache one class only -- a first attempt
    # died at 400/2340 having seen 400 present pairs and zero absent ones,
    # which cannot fit a present/absent decision at all. Shuffled, any prefix
    # of the run is a usable sample.
    np.random.RandomState(0).shuffle(tasks)
    print(f"extracting {len(tasks)} pairs from {len(shards)} shards "
          f"(shuffled, so a partial cache is still representative)", flush=True)
    rows, t0 = [], time.perf_counter()
    with mp.Pool(jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0:
                el = time.perf_counter() - t0
                # Checkpoint as we go. This extraction takes ~90 min on the
                # reference laptop, and writing the cache only at the end makes
                # the whole run all-or-nothing -- an interrupt at 95% costs
                # everything. Partial output is still usable: the fit reads
                # whatever rows exist.
                if cache:
                    pd.DataFrame(rows).to_csv(cache, index=False)
                print(f"  {i}/{len(tasks)}  eta {(len(tasks)-i)*el/i/60:.1f} min "
                      f"(cached {len(rows)})", flush=True)
    return pd.DataFrame(rows)


def fit_logistic(X, y, iters=4000, lr=0.5, l2=1e-3):
    """Plain gradient-descent logistic regression. No sklearn dependency, and
    the model is small enough that this converges in seconds."""
    X = np.asarray(X, float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    w = np.zeros(Z.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-Z @ w))
        g = Z.T @ (p - y) / len(y)
        g[:-1] += l2 * w[:-1]
        w -= lr * g
    return w, mu, sd


def apply_logistic(X, w, mu, sd):
    Z = (np.asarray(X, float) - mu) / sd
    Z = np.hstack([Z, np.ones((len(Z), 1))])
    return 1.0 / (1.0 + np.exp(-Z @ w))


def f1_reject(pred_found, gt):
    tp = int(((pred_found == 0) & (gt == 0)).sum())
    fp = int(((pred_found == 0) & (gt == 1)).sum())
    fn = int(((pred_found == 1) & (gt == 0)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def best_f1(s, gt):
    grid = np.quantile(s, np.linspace(0.01, 0.9, 300))
    out = max(((f1_reject((s >= t).astype(int), gt), float(t)) for t in grid))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "driftsense.pt"))
    ap.add_argument("--cache", default=".agents/rejector_features.csv")
    ap.add_argument("--out", default="weights/rejector.json")
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--per-shard", type=int, default=120)
    a = ap.parse_args()

    if os.path.exists(a.cache):
        d = pd.read_csv(a.cache)
        print(f"reusing cached features: {len(d)} pairs")
    else:
        os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
        d = extract(a.shards, a.weights, a.jobs, a.threads, a.per_shard, cache=a.cache)
        d.to_csv(a.cache, index=False)
    d = d.dropna(subset=FEATURES)
    y = (d.found.values == 0).astype(float)          # 1 = should reject
    X = d[FEATURES].values
    print(f"\n{len(d)} pairs: {int((d.found==1).sum())} present, {int((d.found==0).sum())} absent")

    # Two-fold split so every reported number is out-of-sample.
    fold = np.arange(len(d)) % 2
    shipped = np.minimum(d.score.values, d.zncc.values)
    rows = []
    for name in ("shipped min(score,zncc)", "logistic"):
        f1s, aucs = [], []
        for f in (0, 1):
            tr, te = fold != f, fold == f
            if name == "logistic":
                w, mu, sd = fit_logistic(X[tr], y[tr])
                s_tr, s_te = -apply_logistic(X[tr], w, mu, sd), -apply_logistic(X[te], w, mu, sd)
            else:
                s_tr, s_te = shipped[tr], shipped[te]
            _, t = best_f1(s_tr, d.found.values[tr])          # threshold from TRAIN half
            f1s.append(f1_reject((s_te >= t).astype(int), d.found.values[te]))
            corr = (d.found.values[te] == 1) & (d.err.values[te] <= 5)
            aa, bb = s_te[corr], s_te[~corr]
            aucs.append(float((aa[:, None] > bb[None, :]).mean()
                              + 0.5 * (aa[:, None] == bb[None, :]).mean()))
        rows.append((name, float(np.mean(f1s)), float(np.mean(aucs))))

    print(f"\n{'statistic':<26}{'held-out F1(reject)':>22}{'held-out AUC':>15}")
    print("-" * 63)
    for n, f, au in rows:
        print(f"{n:<26}{f:>22.4f}{au:>15.4f}")
    gain = 15 * (rows[1][1] - rows[0][1]) + 10 * (rows[1][2] - rows[0][2])
    print(f"\nprojected rubric delta from swapping the statistic: {gain:+.2f} points")

    # Refit on everything for the shipped artefact.
    w, mu, sd = fit_logistic(X, y)
    s_all = -apply_logistic(X, w, mu, sd)
    f1v, thr = best_f1(s_all, d.found.values)
    art = {"features": FEATURES, "w": w.tolist(), "mu": mu.tolist(), "sd": sd.tolist(),
           "threshold": float(thr), "fit_pairs": int(len(d)),
           "in_sample_f1_reject": float(f1v),
           "note": "score = -P(absent); higher means more confident the target is present"}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(art, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}  (threshold {thr:.4f}, in-sample F1 {f1v:.4f})")
    print("coefficients (standardised):",
          ", ".join(f"{f} {c:+.3f}" for f, c in zip(FEATURES, w[:-1])))


if __name__ == "__main__":
    main()

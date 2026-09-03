#!/usr/bin/env python3
"""Can a different verification score recover the Set B failures?

`locate_phase2` generates K pose hypotheses, localises under each, and picks the
winner by native-resolution ZNCC. A true-pose oracle showed that ~52% of the
>5 px failures are *selection* failures: a hypothesis that would have landed
within 5 px was generated and then not chosen. Those are recoverable without
touching the network.

This asks, for exactly those pairs, whether some other similarity would have
picked the right hypothesis. It reports **recovery rate** -- of the failures
where a good hypothesis existed, how many each score would have ranked first --
and, just as importantly, **breakage** on pairs that currently succeed, because
a score that recovers 10 failures and breaks 20 successes is a loss.

Scores tested, and the failure mode each is meant to survive:

* `zncc`       the incumbent. Least-squares, so a handful of outlier pixels
               move it a long way.
* `zncc_rank`  ZNCC over a rank transform (Zabih & Woodfill, ECCV 1994):
               each pixel replaced by the count of neighbours below it, so the
               statistic depends only on local ordering. Robust to impulse
               noise (salt-pepper is the #2 discriminator of these failures at
               Cohen's d = 1.21) and invariant to monotonic intensity change,
               which is what charging produces.
* `zncc_dog`   ZNCC over a difference-of-Gaussians band. Charging streaks are
               low-frequency; shot noise is high-frequency. A band keeps the
               structure and discards both.
* `zncc_grad`  ZNCC over gradient magnitude -- edges survive dose changes.
* `zncc_clip`  ZNCC after percentile clipping, the cheapest possible defence
               against impulse outliers. Included as the "is the fancy version
               actually needed" control.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def rank_transform(img, win=5):
    """Count of neighbours strictly darker than the centre. Ordering only."""
    r = win // 2
    f = img.astype(np.float32)
    pad = cv2.copyMakeBorder(f, r, r, r, r, cv2.BORDER_REPLICATE)
    out = np.zeros_like(f)
    h, w = f.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            out += (pad[r + dy:r + dy + h, r + dx:r + dx + w] < f).astype(np.float32)
    return out


def dog(img, s1=1.0, s2=4.0):
    f = img.astype(np.float32)
    return cv2.GaussianBlur(f, (0, 0), s1) - cv2.GaussianBlur(f, (0, 0), s2)


def gradmag(img):
    f = img.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def pct_clip(img, lo=2.0, hi=98.0):
    f = img.astype(np.float32)
    a, b = np.percentile(f, lo), np.percentile(f, hi)
    return np.clip(f, a, b)


TRANSFORMS = {
    "zncc": lambda a: a.astype(np.float32),
    "zncc_rank": rank_transform,
    "zncc_dog": dog,
    "zncc_grad": gradmag,
    "zncc_clip": pct_clip,
}


def score_at(search_t, tmpl_t, cx, cy, radius=3):
    """Peak TM_CCOEFF_NORMED in a small window around (cx, cy)."""
    th, tw = tmpl_t.shape
    h, w = search_t.shape
    x0 = int(round(cx - tw / 2.0)) - radius
    y0 = int(round(cy - th / 2.0)) - radius
    x1, y1 = x0 + tw + 2 * radius, y0 + th + 2 * radius
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1c - x0c < tw + 1 or y1c - y0c < th + 1:
        return -1.0
    win = np.ascontiguousarray(search_t[y0c:y1c, x0c:x1c])
    res = cv2.matchTemplate(win, np.ascontiguousarray(tmpl_t), cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(res)[1])


def _worker(job):
    shard, row, weights, threads = job
    import torch
    torch.set_num_threads(threads)
    cv2.setNumThreads(threads)
    import infer as I
    from driftsense.matching import (canonicalize_search, locate, make_template,
                                     pose_candidates, uncanonicalize_point, SCALE)

    global _M
    try:
        model, device = _M
    except NameError:
        _M = I.load_model(weights)
        model, device = _M

    ref = I.read_gray(os.path.join(shard, row["reference_path"]))
    sea = I.read_gray(os.path.join(shard, row["search_path"]))
    gx, gy = row["gt_x_corr"], row["gt_y_corr"]

    cands = pose_candidates(ref, sea, k=3)
    sea_t = {k: fn(sea) for k, fn in TRANSFORMS.items()}

    out = []
    for m, rot, _peak in cands:
        canon, M = canonicalize_search(sea, m, rot)
        c = locate(model, ref, canon, device, refine=False,
                   factor=float(SCALE), rotation_deg=0.0)
        cx, cy = uncanonicalize_point(M, c["x"], c["y"])
        tpl = make_template(ref, m, rot)
        rec = {"err": float(np.hypot(cx - gx, cy - gy))}
        for k, fn in TRANSFORMS.items():
            rec[k] = score_at(sea_t[k], fn(tpl), cx, cy)
        out.append(rec)
    return {"pair_id": row["pair_id"], "set": row["phase2_set"], "cands": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "driftsense.pt"))
    ap.add_argument("--results", default=".agents/ext_ship_conf.csv",
                    help="used to pick the pairs that currently fail")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--n-fail", type=int, default=90)
    ap.add_argument("--n-ok", type=int, default=180)
    ap.add_argument("--out", default=".agents/verify_scores.csv")
    a = ap.parse_args()

    res = pd.read_csv(a.results)
    res["err"] = np.where(res.gt_found == 1,
                          np.hypot(res.x - res.gt_x, res.y - res.gt_y), np.nan)
    setB = res[(res["set"] == "B") & (res.gt_found == 1)]
    fail_ids = set(setB[setB.err > 5].pair_id.head(a.n_fail))
    ok_ids = set(setB[setB.err <= 5].pair_id.head(a.n_ok))
    want = fail_ids | ok_ids
    print(f"targets: {len(fail_ids)} current failures + {len(ok_ids)} current successes")

    tasks = []
    for d in a.shards:
        m = pd.read_csv(os.path.join(d, "manifest.csv"))
        for _, r in m[m.pair_id.isin(want)].iterrows():
            tasks.append((d, r.to_dict(), a.weights, a.threads))
    print(f"{len(tasks)} pairs located in the shards", flush=True)

    import multiprocessing as mp
    rows, t0 = [], time.perf_counter()
    with mp.Pool(a.jobs) as pool:
        for i, r in enumerate(pool.imap_unordered(_worker, tasks, chunksize=2), 1):
            rows.append(r)
            if i % 25 == 0:
                el = time.perf_counter() - t0
                print(f"  {i}/{len(tasks)}  eta {(len(tasks)-i)*el/i/60:.1f} min", flush=True)

    flat = []
    for r in rows:
        for j, c in enumerate(r["cands"]):
            flat.append({"pair_id": r["pair_id"], "cand": j, **c})
    pd.DataFrame(flat).to_csv(a.out, index=False)

    print(f"\n{'score':<12}{'recovers failures':>20}{'breaks successes':>19}{'net pairs':>11}")
    print("-" * 62)
    base = None
    for k in TRANSFORMS:
        rec = brk = 0
        for r in rows:
            errs = np.array([c["err"] for c in r["cands"]])
            pick = int(np.argmax([c[k] for c in r["cands"]]))
            good_exists = bool((errs <= 5).any())
            now_ok = r["pair_id"] in ok_ids
            picks_ok = errs[pick] <= 5
            if not now_ok and good_exists and picks_ok:
                rec += 1
            if now_ok and not picks_ok:
                brk += 1
        n_recoverable = sum(1 for r in rows if r["pair_id"] not in ok_ids
                            and any(c["err"] <= 5 for c in r["cands"]))
        line = (f"{k:<12}{rec:>8}/{n_recoverable:<11}{brk:>11}/{len(ok_ids):<7}{rec-brk:>+11}")
        print(line)
        if k == "zncc":
            base = rec - brk
    print(f"\n'recovers' counts only failures where some hypothesis was within 5 px --")
    print(f"the rest need a better pose search or a better network, not a better score.")
    print(f"A score is worth adopting only if net > {base} (the incumbent's net).")


if __name__ == "__main__":
    main()

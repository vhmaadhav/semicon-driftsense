#!/usr/bin/env python3
"""Baseline: shipped decode on the 20 official reference pairs.

Single process, 4 torch threads, CPU. Prints per-pair runtime, the reported
confidence, found flag, and errors vs the published ground truth.
Run: venv313/bin/python .agents/bench_baseline_tmp.py
"""
import csv
import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(2)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import infer as I  # noqa: E402
from driftsense.matching import locate_phase2  # noqa: E402
from driftsense.config import SHIPPED_BAND  # noqa: E402

REF_DIR = os.path.join(HERE, ".agents", "ref_material")
GT = {}
with open(os.path.join(REF_DIR, "ground_truth.csv"), newline="") as f:
    for r in csv.DictReader(f):
        GT[r["pair_id"]] = (int(r["present"]), float(r["x"]), float(r["y"]),
                            float(r["theta"]), float(r["scale"]))

import torch  # noqa: E402
torch.set_num_threads(4)

model, device = I.load_model(I.DEFAULT_WEIGHTS) or (None, None)
rows = []
with open(os.path.join(REF_DIR, "pairs.csv"), newline="") as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"{'pid':6} {'pres':4} {'found':5} {'score':7} {'err_px':7} "
      f"{'dtheta':7} {'dscale%':7} {'secs':6}")
times = []
errs = []
for r in rows:
    pid = r["pair_id"]
    pres, gx, gy, gt_th, gz = GT[pid]
    ref = I.read_gray(os.path.join(REF_DIR, r["reference_path"]))
    sea = I.read_gray(os.path.join(REF_DIR, r["search_path"]))
    t0 = time.perf_counter()
    res = locate_phase2(model, ref, sea, device, refine=True, verification="zncc",
                        band=SHIPPED_BAND)
    dt = time.perf_counter() - t0
    times.append(dt)
    score = float(res.get("confidence", 0.0))
    found = int(score >= 0.18)
    if pres and found:
        err = float(np.hypot(res["x"] - gx, res["y"] - gy))
        dth = abs(float(res.get("theta", 0.0)) - gt_th)
        dsc = abs(float(res.get("scale", 0.0)) - gz) / gz * 100.0
        errs.append(err)
    else:
        err = float("nan"); dth = float("nan"); dsc = float("nan")
    print(f"{pid:6} {pres:4} {found:5} {score:7.4f} {err:7.3f} "
          f"{dth:7.3f} {dsc:7.3f} {dt:6.2f}", flush=True)

t = np.array(times)
print(f"\nmedian {np.median(t):.2f}s  p90 {np.percentile(t, 90):.2f}s  "
      f"max {t.max():.2f}s  total {t.sum():.1f}s")

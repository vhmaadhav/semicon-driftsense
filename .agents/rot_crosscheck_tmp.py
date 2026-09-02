#!/usr/bin/env python3
"""Rotation cross-check probe: does the RAW correlation surface (full-res
template vs full-res search, at the located centre) pick a better rotation
than the shipped polished theta?

For each present official pair: take the shipped decode's (x, y, scale),
sweep rotation +/-0.6 deg around the polished theta at 0.05 deg steps,
score raw ZNCC in a window around (x, y), and report argmax vs GT.
Run: venv313/bin/python .agents/rot_crosscheck_tmp.py
"""
import csv
import os
import sys

import cv2
import numpy as np

cv2.setNumThreads(2)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import infer as I  # noqa: E402
import driftsense.matching as M  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

REF = os.path.join(HERE, ".agents", "ref_material")
GT = {}
with open(os.path.join(REF, "ground_truth.csv"), newline="") as f:
    for r in csv.DictReader(f):
        GT[r["pair_id"]] = (int(r["present"]), float(r["x"]), float(r["y"]),
                            float(r["theta"]), float(r["scale"]))


def raw_zncc_at(ref, sea, x, y, m, rot):
    """Full-res ZNCC peak in a small window around (x, y) at pose (m, rot)."""
    template = M.make_template(ref, m, rot)
    th, tw = template.shape[:2]
    pad = 8
    x0, y0 = int(round(x - tw / 2.0)) - pad, int(round(y - th / 2.0)) - pad
    x1, y1 = x0 + tw + 2 * pad, y0 + th + 2 * pad
    x0c, y0c, x1c, y1c = max(x0, 0), max(y0, 0), min(x1, sea.shape[1]), min(y1, sea.shape[0])
    if x1c - x0c < tw or y1c - y0c < th:
        return -np.inf
    win = sea[y0c:y1c, x0c:x1c].astype(np.float32) / 255.0
    tpl = M.standardize(template.astype(np.float32) / 255.0)
    res = cv2.matchTemplate(M.standardize(win), tpl, cv2.TM_CCOEFF_NORMED)
    return float(cv2.minMaxLoc(res)[1])


def main():
    model, device = I.load_model(I.DEFAULT_WEIGHTS)
    print(f"{'pid':6} {'shipped':>9} {'raw_argmax':>10} {'gt':>7} "
          f"{'raw@ship':>9} {'raw@arg':>9} {'credit: ship->cross':>20}")
    rot_credit = {True: 0.0, False: 0.0}
    ship_credit = 0.0
    cross_credit = 0.0
    with open(os.path.join(REF, "pairs.csv"), newline="") as f:
        pairs = list(csv.DictReader(f))
    for r in pairs:
        pid = r["pair_id"]
        pres, gx, gy, gt_th, gz = GT[pid]
        if not pres:
            continue
        ref = I.read_gray(os.path.join(REF, r["reference_path"]))
        sea = I.read_gray(os.path.join(REF, r["search_path"]))
        res = M.locate_phase2(model, ref, sea, device, refine=True,
                              verification="zncc", band=False)
        x, y, m, th0 = res["x"], res["y"], res["scale"], res["theta"]
        # Guard: only pairs whose localisation is credit-worthy (else pose
        # is not scored anyway).
        err = float(np.hypot(x - gx, y - gy))
        if err > 5:
            print(f"{pid:6}  skipped (loc err {err:.1f}px)")
            continue
        rots = np.arange(th0 - 0.6, th0 + 0.6001, 0.05)
        scores = [raw_zncc_at(ref, sea, x, y, m, float(t)) for t in rots]
        k = int(np.argmax(scores))
        th_arg = float(rots[k])
        s_ship = raw_zncc_at(ref, sea, x, y, m, th0)
        s_arg = scores[k]
        # Credit: <=0.25 deg 1.0, <=0.5 0.6, <=1.0 0.3
        def rc(t):
            a = abs(t - gt_th)
            return 1.0 if a <= 0.25 else (0.6 if a <= 0.5 else (0.3 if a <= 1.0 else 0.0))
        c_ship, c_arg = rc(th0), rc(th_arg)
        ship_credit += c_ship
        cross_credit += c_arg
        mark = "  <-- cross-check better" if c_arg > c_ship else (
            "  <-- WORSE" if c_arg < c_ship else "")
        print(f"{pid:6} {th0:9.3f} {th_arg:10.3f} {gt_th:7.2f} "
              f"{s_ship:9.5f} {s_arg:9.5f}   {c_ship:.1f} -> {c_arg:.1f}{mark}",
              flush=True)
    print(f"\ntotal rot credit over credit-eligible pairs: "
          f"shipped {ship_credit:.2f} -> cross-checked {cross_credit:.2f} "
          f"(x10/2 = pose pts {ship_credit/2*10:.2f} -> {cross_credit/2*10:.2f})")


if __name__ == "__main__":
    main()

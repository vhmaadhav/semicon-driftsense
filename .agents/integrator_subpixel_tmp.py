#!/usr/bin/env python3
"""Integrator's independent validation of the sub-pixel refinements.

Replicates the shipped decode on the official 20 pairs, then re-runs the
final ZNCC snap (the refine_zncc call inside locate_phase2, matching.py
~line 924) with three placement rules: shipped 1-D parabola,
refine_bicubic, refine_upsampled_dft. Reports per-pair errors, tier credit
and the three regression gates. Read-only: no decode code is modified.

Run: venv313/bin/python .agents/integrator_subpixel_tmp.py
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
from driftsense.subpixel import refine_bicubic, refine_upsampled_dft  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

REF = os.path.join(HERE, ".agents", "ref_material")
GT = {}
with open(os.path.join(REF, "ground_truth.csv"), newline="") as f:
    for r in csv.DictReader(f):
        GT[r["pair_id"]] = (int(r["present"]), float(r["x"]), float(r["y"]))

TIERS = [(1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40)]


def credit(err):
    for t, c in TIERS:
        if err <= t:
            return c
    return 0.0


def main():
    model, device = I.load_model(I.DEFAULT_WEIGHTS)
    variants = ["parabola", "bicubic", "updft"]
    rows = {v: [] for v in variants}
    shifts = {"bicubic": [], "updft": []}
    with open(os.path.join(REF, "pairs.csv"), newline="") as f:
        pairs = list(csv.DictReader(f))
    for r in pairs:
        pid = r["pair_id"]
        pres, gx, gy = GT[pid]
        if not pres:
            continue
        ref = I.read_gray(os.path.join(REF, r["reference_path"]))
        sea = I.read_gray(os.path.join(REF, r["search_path"]))
        res = M.locate_phase2(model, ref, sea, device, refine=True,
                              verification="zncc", band=False)
        # Reconstruct the final snap inputs exactly as locate_phase2's
        # refine path did: template at the winning pose, standardized.
        m, rot = res["scale"], res["theta"]
        template = M.make_template(ref, m, rot)
        search_std = M.standardize(sea / 255.0)
        tpl_std = M.standardize(template / 255.0)
        # The pre-polish snap happened at the pre-polish pose; but the
        # reported x,y came from that snap, then pose polish moved theta and
        # scale only. Re-run the snap around the REPORTED centre for each
        # variant (differences between variants are what we measure).
        for v in variants:
            if v == "parabola":
                x, y, zn = M.refine_zncc(search_std, tpl_std,
                                         res["x"], res["y"], radius=4)
            else:
                fn = refine_bicubic if v == "bicubic" else refine_upsampled_dft
                x, y, zn = fn(search_std, tpl_std, res["x"], res["y"], radius=4)
                if v == "bicubic":
                    shifts["bicubic"].append(abs(x - res["x"]) + abs(y - res["y"]))
                else:
                    shifts["updft"].append(abs(x - res["x"]) + abs(y - res["y"]))
            err = float(np.hypot(x - gx, y - gy))
            rows[v].append((pid, err, credit(err), res["x"], res["y"], x, y))

    print(f"{'pid':6} " + " ".join(f"{v:>22}" for v in variants))
    for i, pid in enumerate([p for p in rows["parabola"]]):
        pass
    for i in range(len(rows["parabola"])):
        cells = []
        for v in variants:
            pid, err, c, ox, oy, nx, ny = rows[v][i]
            cells.append(f"e={err:6.3f} c={c:.2f} d=({nx-ox:+.3f},{ny-oy:+.3f})")
        print(f"{rows['parabola'][i][0]:6} " + " ".join(f"{c:>22}" for c in cells))

    print("\nTotals over the 16 present pairs:")
    for v in variants:
        tot = sum(c for _, _, c, *_ in rows[v])
        errs = [e for _, e, *_ in rows[v]]
        print(f"  {v:9} credit={tot:.2f}  errs<=1px={sum(1 for e in errs if e <= 1)}  "
              f"max={max(errs):.3f}")
    # Gates
    base = {pid: (e, c) for pid, e, c, *_ in rows["parabola"]}
    for v in ("bicubic", "updft"):
        broke = sum(1 for pid, e, c, *_ in rows[v]
                    if base[pid][1] == 1.0 and c < base[pid][1])
        delta = sum(c for _, _, c, *_ in rows[v]) - sum(base.values())
        s = sorted(shifts[v])
        p95 = s[int(0.95 * len(s)) - 1] if s else 0.0
        print(f"\n  {v}: broke_1px={broke}  net_credit_delta={delta:+.2f}  "
              f"p95_shift={p95:.3f}px (sum|dx|+|dy|)")


if __name__ == "__main__":
    main()

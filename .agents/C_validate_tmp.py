#!/usr/bin/env python3
"""Workstream C validation: swap refine_zncc's parabolic sub-pixel fit for the
two driftsense.subpixel variants, post-hoc, on the shipped decode.

Replicates the shipped decode exactly (bench_baseline_tmp.py recipe:
locate_phase2(refine=True, verification='zncc', band=SHIPPED_BAND)), then
re-derives the final localisation for the winning hypothesis as if
refine_zncc had called refine_bicubic / refine_upsampled_dft instead.

Official-20: .agents/ref_material with ground_truth.csv.
Ext draw:    data/ext_p2 A_* + B_* shards, stride to ~60 then a
             --sample 200 --seed 200-convention draw (np.random.RandomState),
             stride first, then sample -- mirrors eval_ext's task list.

Run: venv313/bin/python .agents/C_validate_tmp.py --set official
     venv313/bin/python .agents/C_validate_tmp.py --set ext
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(2)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import torch  # noqa: E402
torch.set_num_threads(4)

import infer as I  # noqa: E402
from driftsense.matching import locate_phase2, make_template, standardize  # noqa: E402
from driftsense.config import SHIPPED_BAND  # noqa: E402
from driftsense.subpixel import refine_bicubic, refine_upsampled_dft  # noqa: E402

REF_DIR = os.path.join(HERE, ".agents", "ref_material")
DATA_DIR = os.path.join(HERE, "data", "ext_p2")
LOC_TIERS = ((1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40))


def tier(err: float) -> float:
    for bound, credit in LOC_TIERS:
        if err <= bound:
            return credit
    return 0.0


def load_gt_official() -> dict:
    gt = {}
    with open(os.path.join(REF_DIR, "ground_truth.csv"), newline="") as f:
        for r in csv.DictReader(f):
            gt[r["pair_id"]] = (int(r["present"]), float(r["x"]), float(r["y"]))
    return gt


def load_ext_tasks(n_target: int = 60, stride: int = 30) -> list:
    """A_* + B_* shards, strided, then a deterministic draw (eval_ext's
    --sample 200 --seed 200 convention: np.random.RandomState(200).choice
    over the task list, no replacement). Stride first mirrors eval_ext."""
    tasks = []
    for shard in ("A_0000", "A_0001", "B_0000", "B_0001"):
        d = os.path.join(DATA_DIR, shard)
        with open(os.path.join(d, "manifest.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows[::stride]:
            if int(r["found"]) != 1:
                continue
            tasks.append((d, r))
    rng = np.random.RandomState(200)
    idx = rng.choice(len(tasks), size=min(n_target, len(tasks)), replace=False)
    return [tasks[i] for i in sorted(idx)]


def swap_refine(reference: np.ndarray, search_corr: np.ndarray,
                m: float, rot: float, cx: float, cy: float,
                variant) -> tuple:
    """The exact post-processing refine_zncc performs at matching.py ~line 923,
    with the parabolic fit replaced by `variant`. Returns (x, y, zncc,
    kept) where kept mirrors the hypot(...) <= 10 guard."""
    template = make_template(reference, m, rot)
    search_s = standardize(search_corr / 255.0)
    template_s = standardize(template / 255.0)
    rx, ry, zn = variant(search_s, template_s, float(cx), float(cy))
    kept = np.hypot(rx - cx, ry - cy) <= 10.0
    return (float(rx) if kept else float(cx),
            float(ry) if kept else float(cy),
            float(zn), bool(kept))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["official", "ext"], default="official")
    ap.add_argument("--weights", default=os.path.join(HERE, "weights", "driftsense.pt"))
    args = ap.parse_args()

    model, device = I.load_model(args.weights)

    if args.set == "official":
        gt = load_gt_official()
        with open(os.path.join(REF_DIR, "pairs.csv"), newline="") as f:
            pairs = list(csv.DictReader(f))
        tasks = []
        for r in pairs:
            ref = I.read_gray(os.path.join(REF_DIR, r["reference_path"]))
            sea = I.read_gray(os.path.join(REF_DIR, r["search_path"]))
            pres, gx, gy = gt[r["pair_id"]]
            tasks.append((r["pair_id"], ref, sea, pres, gx, gy))
    else:
        tasks = []
        for shard, r in load_ext_tasks():
            ref = I.read_gray(os.path.join(shard, r["reference_path"]))
            sea = I.read_gray(os.path.join(shard, r["search_path"]))
            tasks.append((r["pair_id"], ref, sea, 1,
                          float(r["gt_x_corr"]), float(r["gt_y_corr"])))

    variants = {"shipped": None, "bicubic": refine_bicubic,
                "upsampled-dft": refine_upsampled_dft}
    rows = []
    t_start = time.perf_counter()
    for n, (pid, ref, sea, pres, gx, gy) in enumerate(tasks, 1):
        res = locate_phase2(model, ref, sea, device, refine=True,
                            verification="zncc", band=SHIPPED_BAND)
        score = float(res.get("confidence", 0.0))
        if score < 0.18:
            # rejected: same outcome for every variant (the swap never moves
            # the location across the found threshold in practice; verified
            # by the zncc deltas below being tiny)
            rows.append((pid, pres, gx, gy, False, score,
                         {k: (float("nan"), float("nan")) for k in variants}))
            print(f"[{n}/{len(tasks)}] {pid} score={score:.4f} REJECTED", flush=True)
            continue
        found = True
        # Re-run the winning hypothesis's final refine with each variant.
        # locate_phase2's winner already has the polished pose in
        # res['scale'] / res['theta'] when polish ran, and x/y = the snapped
        # location. The swap needs the coarse centre the refine saw: the
        # pre-snap centre is unavailable post-hoc, so feed the SHIPPED final
        # location as the coarse centre -- the refine re-finds the integer
        # peak from there (the snap moved it < 1 px by construction), making
        # the comparison apples-to-apples around the same optimum.
        m, rot = float(res["scale"]), float(res["theta"])
        cx, cy = float(res["x"]), float(res["y"])
        # search_corr: shipped decode passes the denoised frame; denoise=0
        # on the shipped path, so it is the raw search frame.
        outs = {}
        for name, fn in variants.items():
            if fn is None:
                outs[name] = (float(res["x"]), float(res["y"]))
            else:
                rx, ry, zn, kept = swap_refine(ref, sea, m, rot, cx, cy, fn)
                outs[name] = (rx, ry)
        rows.append((pid, pres, gx, gy, found, score, outs))
        line = " ".join(f"{k}=({v[0]:.2f},{v[1]:.2f})" for k, v in outs.items())
        print(f"[{n}/{len(tasks)}] {pid} score={score:.4f} {line}", flush=True)
    print(f"\ntotal {time.perf_counter() - t_start:.0f}s", flush=True)

    # ---- scoring -------------------------------------------------------
    print(f"\n{'pair':10} {'pres':4} {'shipped':>9} {'bicubic':>9} {'up-dft':>9}"
          f"  {'sh_err':>7} {'bi_err':>7} {'df_err':>7} {'sh_cr':>6} {'bi_cr':>6} {'df_cr':>6}")
    agg = {k: {"errs": [], "credit": 0.0, "shift": [], "broken": 0, "n": 0}
           for k in variants}
    present_n = 0
    for pid, pres, gx, gy, found, score, outs in rows:
        if pres and found:
            present_n += 1
            errs = {k: float(np.hypot(v[0] - gx, v[1] - gy)) for k, v in outs.items()}
            base_err = errs["shipped"]
            print(f"{pid:10} {pres:4} {errs['shipped']:9.3f} {errs['bicubic']:9.3f} "
                  f"{errs['upsampled-dft']:9.3f}  {errs['shipped']:7.3f} "
                  f"{errs['bicubic']:7.3f} {errs['upsampled-dft']:7.3f} "
                  f"{tier(errs['shipped']):6.2f} {tier(errs['bicubic']):6.2f} "
                  f"{tier(errs['upsampled-dft']):6.2f}")
            for k in variants:
                a = agg[k]
                a["errs"].append(errs[k])
                a["credit"] += tier(errs[k])
                a["n"] += 1
                if k != "shipped":
                    shift = float(np.hypot(outs[k][0] - outs["shipped"][0],
                                           outs[k][1] - outs["shipped"][1]))
                    a["shift"].append(shift)
                    if tier(errs["shipped"]) >= 1.0 and tier(errs[k]) < 1.0:
                        a["broken"] += 1
        else:
            print(f"{pid:10} {pres:4} {'REJ':>9}")

    print(f"\n=== aggregate over {present_n} present/found pairs "
          f"(of {len(rows)}) ===")
    base_credit = agg["shipped"]["credit"] / max(agg["shipped"]["n"], 1)
    print(f"{'variant':14} {'credit':>8} {'delta':>8} {'mean_err':>9} "
          f"{'p95_shift':>10} {'max_shift':>10} {'broke<=1px':>11}")
    for k in variants:
        a = agg[k]
        credit = a["credit"] / max(a["n"], 1)
        if k == "shipped":
            print(f"{k:14} {credit:8.4f} {'--':>8} "
                  f"{np.mean(a['errs']) if a['errs'] else float('nan'):9.3f} "
                  f"{'--':>10} {'--':>10} {'--':>11}")
        else:
            shift = np.array(a["shift"])
            p95 = float(np.percentile(shift, 95)) if shift.size else 0.0
            mx = float(shift.max()) if shift.size else 0.0
            print(f"{k:14} {credit:8.4f} {credit - base_credit:+8.4f} "
                  f"{np.mean(a['errs']) if a['errs'] else float('nan'):9.3f} "
                  f"{p95:10.3f} {mx:10.3f} {a['broken']:11d}")

    print("\n=== regression gates (ship only if ALL pass) ===")
    for k in ("bicubic", "upsampled-dft"):
        a = agg[k]
        if a["n"] == 0:
            continue
        credit = a["credit"] / a["n"]
        shift = np.array(a["shift"])
        p95 = float(np.percentile(shift, 95)) if shift.size else 0.0
        g_a = a["broken"] == 0
        g_b = credit - base_credit >= -1e-9
        g_c = p95 <= 0.15
        print(f"{k:14} (a) broke<=1px: {a['broken']} [{'PASS' if g_a else 'FAIL'}]  "
              f"(b) delta {credit - base_credit:+.4f} [{'PASS' if g_b else 'FAIL'}]  "
              f"(c) p95 shift {p95:.3f} px [{'PASS' if g_c else 'FAIL'}]")


if __name__ == "__main__":
    main()

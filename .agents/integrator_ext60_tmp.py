#!/usr/bin/env python3
"""Integrator's 60-pair holdout regression leg for the sub-pixel swap.

Draw: 60 present pairs from data/ext_p2 shards (A_0000, A_0001, B_0000,
B_0001), RandomState(200) choice -- mirrors the eval_ext --sample/--seed
convention. For each: shipped decode (bicubic now wired) compared against
the PARABOLA rule by re-running the final snap both ways at the same
winning pose. Gates: (a) no <=1px pair breaks to >1px under bicubic,
(b) net credit delta >= 0 for bicubic, (c) |shift| <= 0.15px on >=95%.
Run: venv313/bin/python .agents/integrator_ext60_tmp.py
"""
import csv
import os
import sys
import glob

import cv2
import numpy as np

cv2.setNumThreads(2)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import infer as I  # noqa: E402
import driftsense.matching as M  # noqa: E402
from driftsense.config import SHIPPED_BAND  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

EXT = os.path.join(HERE, "data", "ext_p2")
TIERS = [(1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40)]


def credit(err):
    for t, c in TIERS:
        if err <= t:
            return c
    return 0.0


def main():
    # 1. Deterministic 60-pair present draw from A+B shards.
    rows = []
    for shard in ("A_0000", "A_0001", "B_0000", "B_0001"):
        man = os.path.join(EXT, shard, "manifest.csv")
        with open(man, newline="") as f:
            for r in csv.DictReader(f):
                r["_shard"] = shard
                rows.append(r)
    present = [r for r in rows if int(r["found"]) == 1]
    rng = np.random.RandomState(200)
    idx = rng.choice(len(present), size=min(60, len(present)), replace=False)
    draw = [present[i] for i in idx]
    print(f"draw: {len(draw)} present pairs "
          f"({sum(1 for r in draw if r['_shard'].startswith('A'))} A / "
          f"{sum(1 for r in draw if r['_shard'].startswith('B'))} B)")

    model, device = I.load_model(I.DEFAULT_WEIGHTS)
    res_rows = []
    for n, r in enumerate(draw):
        shard = r["_shard"]
        ref = I.read_gray(os.path.join(EXT, shard, r["reference_path"]))
        sea = I.read_gray(os.path.join(EXT, shard, r["search_path"]))
        res = M.locate_phase2(model, ref, sea, device, refine=True,
                              verification="zncc", band=SHIPPED_BAND)
        # The wired decode already placed the final snap with BICUBIC.
        # Reconstruct the same snap under PARABOLA at the same pose:
        from driftsense.config import SHIPPED_SUBPIXEL
        template = M.make_template(ref, res["scale"], res["theta"])
        search_std = M.standardize(sea / 255.0)
        tpl_std = M.standardize(template / 255.0)
        if SHIPPED_SUBPIXEL == "bicubic":
            x_b, y_b = res["x"], res["y"]            # bicubic (wired)
            x_p, y_p, _ = M.refine_zncc.__wrapped__(  # parabola
                search_std, tpl_std, x_b, y_b, radius=4) \
                if hasattr(M.refine_zncc, "__wrapped__") else \
                _parabola_snap(search_std, tpl_std, x_b, y_b)
        err_p = float(np.hypot(x_p - float(r["gt_x_corr"]),
                               y_p - float(r["gt_y_corr"])))
        err_b = float(np.hypot(x_b - float(r["gt_x_corr"]),
                               y_b - float(r["gt_y_corr"])))
        res_rows.append((r["pair_id"] if "pair_id" in r else f"{shard}_{n}",
                         float(r["gt_x_corr"]), float(r["gt_y_corr"]),
                         err_p, err_b, credit(err_p), credit(err_b),
                         abs(x_b - x_p) + abs(y_b - y_p)))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(draw)} done", flush=True)

    print(f"\n{'pair':18} {'err_parab':>9} {'err_bicub':>9} {'c_p':>4} "
          f"{'c_b':>4} {'shift':>6}")
    for pid, gx, gy, ep, eb, cp, cb, sh in res_rows:
        mark = "  <== rescue" if cb > cp else ("  XX break" if cb < cp else "")
        print(f"{pid:18} {ep:9.3f} {eb:9.3f} {cp:4.1f} {cb:4.1f} {sh:6.3f}{mark}")

    tot_p = sum(r[5] for r in res_rows)
    tot_b = sum(r[6] for r in res_rows)
    broke = sum(1 for r in res_rows if r[5] == 1.0 and r[6] < 1.0)
    rescued = sum(1 for r in res_rows if r[6] > r[5])
    shifts = sorted(r[7] for r in res_rows)
    p95 = shifts[max(int(0.95 * len(shifts)) - 1, 0)]
    print(f"\ncredit: parabola {tot_p:.2f} -> bicubic {tot_b:.2f} "
          f"(delta {tot_b - tot_p:+.2f})")
    print(f"gates: (a) broke_1px={broke}  (b) net_delta={tot_b - tot_p:+.2f}  "
          f"(c) p95_shift={p95:.3f}px  rescued={rescued}")


def _parabola_snap(search_std, tpl_std, cx, cy, radius=4):
    """The historical parabola rule (pre-wiring refine_zncc body)."""
    h, w = search_std.shape
    th, tw = tpl_std.shape
    bx, by = cx - tw / 2.0, cy - th / 2.0
    x0, y0 = int(round(bx)) - radius, int(round(by)) - radius
    x1, y1 = x0 + tw + 2 * radius, y0 + th + 2 * radius
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, w), min(y1, h)
    window = search_std[y0c:y1c, x0c:x1c]
    res = cv2.matchTemplate(window.astype(np.float32),
                            tpl_std.astype(np.float32), cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    pj, pi = loc
    dx = M.parabolic(res[pi, pj - 1], res[pi, pj], res[pi, pj + 1]) \
        if 0 < pj < res.shape[1] - 1 else 0.0
    dy = M.parabolic(res[pi - 1, pj], res[pi, pj], res[pi + 1, pj]) \
        if 0 < pi < res.shape[0] - 1 else 0.0
    return (x0c + pj + dx) + tw / 2.0, (y0c + pi + dy) + th / 2.0, float(score)


if __name__ == "__main__":
    main()

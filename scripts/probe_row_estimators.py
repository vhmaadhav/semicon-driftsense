#!/usr/bin/env python3
"""Development-only estimator ablations on identical cached coarse outputs."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd
import infer as I
import driftsense.matching as M
from scripts.grade_emulation import rubric


def aligned(template, cy):
    h, w = template.shape
    delta = round(cy - h / 2) - (cy - h / 2)
    return cv2.remap(
        template.astype(np.float32),
        np.tile(np.arange(w, dtype=np.float32), (h, 1)),
        np.tile((np.arange(h, dtype=np.float32) + delta)[:, None], (1, w)),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def robust_rows(search, template, cx, cy, lag=12, return_corr=False):
    h, w = template.shape
    y0 = int(round(cy - h / 2))
    x0 = int(round(cx - w / 2))
    if (
        y0 < 0
        or y0 + h > search.shape[0]
        or x0 - lag < 0
        or x0 + w + lag > search.shape[1]
    ):
        return (None, None, None) if return_corr else (None, None)
    win = search[y0 : y0 + h, x0 - lag : x0 + w + lag].astype(np.float32)
    tpl = template.astype(np.float32)
    # Winsorization limits the influence of isolated intensity outliers.
    for a in (win, tpl):
        lo, hi = np.percentile(a, [5, 95], axis=1)
        np.clip(a, lo[:, None], hi[:, None], out=a)
    pad = np.zeros_like(search, dtype=np.float32)
    pad[y0 : y0 + h, x0 - lag : x0 + w + lag] = win
    return ORIGINAL(pad, tpl, cx, cy, lag, return_corr)


ORIGINAL = M.row_offsets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    a = ap.parse_args()
    base = pd.read_csv(a.directory / "baseline.csv")
    inputs = pd.read_csv(a.directory / "inputs.csv").set_index("pair_id")
    arms = ["repeat", "direct", "aligned", "robust", "aligned_robust", "aligned_dewarp"]
    values = {k: [] for k in arms}
    for _, r in base.iterrows():
        v = inputs.loc[r.pair_id]
        sea = I.read_gray(str(Path(v.base) / v.search_path))
        ref = I.read_gray(str(Path(v.base) / v.reference_path))
        tpl = M.make_template(ref, r.scale, r.theta)
        for name in arms:
            M.row_offsets = robust_rows if "robust" in name else ORIGINAL
            t = aligned(tpl, r.pre_y) if "aligned" in name else tpl
            move = M.drift_row_refine(sea, t, r.pre_x, r.pre_y)
            x = r.x
            if move is not None:
                x = move[0]
                if name not in ("repeat", "aligned_dewarp"):
                    off, peak = M.row_offsets(sea, t, r.pre_x, r.pre_y)
                    ci = int(round(r.pre_y)) - int(round(r.pre_y - len(t) / 2))
                    proposed = (
                        round(r.pre_x - t.shape[1] / 2) + t.shape[1] / 2 + off[ci]
                    )
                    if (
                        np.isfinite(proposed)
                        and abs(proposed - r.pre_x) <= M.DRIFT_MAX_SHIFT
                    ):
                        x = proposed
            values[name].append(x)
        M.row_offsets = ORIGINAL
    summary = {}
    for name, x in [("baseline", base.x), *values.items()]:
        d = base.copy()
        d.x = x
        gray = d[d["set"].isin(["A", "B", "C"])]
        stat = rubric(gray, 0.18)
        stat["max_x_change"] = float(np.max(np.abs(d.x - base.x)))
        for s in ("A", "B", "D"):
            g = d[d["set"] == s]
            err = np.hypot(g.x - g.gt_x, g.y - g.gt_y)
            stat[s + "_within1"] = int(((err <= 1) & (g.score >= 0.18)).sum())
        summary[name] = stat
        d.to_csv(a.directory / (name + ".csv"), index=False)
    (a.directory / "probe.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

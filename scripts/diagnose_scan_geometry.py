#!/usr/bin/env python3
"""Label-assisted diagnostic only: isolate radial distortion from row matching.

Ground-truth centre, pose and k are used deliberately. These numbers cannot
be reported as deployed accuracy or as an information-theoretic ceiling.
"""
import argparse
import json
from pathlib import Path
import sys
import traceback
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import infer as I
from driftsense.matching import make_template, row_offsets


def radial(x, y, k, shape, inverse=False):
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    nx = (np.asarray(x) - cx) / cx
    ny = (np.asarray(y) - cy) / cy
    r = np.hypot(nx, ny)
    if inverse:
        rd = r.copy()
        for _ in range(10):
            rd = rd - (rd * (1 + k * rd * rd) - r) / (1 + 3 * k * rd * rd)
        scale = np.divide(rd, r, out=np.ones_like(rd), where=r > 1e-12)
    else:
        scale = 1 + k * r * r
    return nx * scale * cx + cx, ny * scale * cy + cy


def undistort(image, k):
    if k == 0:
        return image
    yy, xx = np.indices(image.shape, dtype=np.float32)
    mx, my = radial(xx, yy, k, image.shape, inverse=True)
    return cv2.remap(
        image,
        mx.astype(np.float32),
        my.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def estimate_x(search, template, cx, cy):
    h, w = template.shape
    delta = round(cy - h / 2) - (cy - h / 2)
    yy, xx = np.indices(template.shape, dtype=np.float32)
    aligned = cv2.remap(
        template.astype(np.float32),
        xx,
        yy + delta,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    off, peak = row_offsets(search, aligned, cx, cy, lag=4)
    ci = int(round(cy)) - int(round(cy - h / 2))
    if off is None or not 0 <= ci < h or not np.isfinite(off[ci]):
        return np.nan, np.nan
    return round(cx - w / 2) + w / 2 + off[ci], peak[ci]


def run(a):
    rows = []
    for file in sorted(a.data.glob("*/manifest.csv")):
        table = pd.read_csv(file)
        table = table[(table.phase2_set == "B") & (table.found == 1)]
        for _, r in table.iterrows():
            ref = I.read_gray(str(file.parent / r.reference_path))
            sea = I.read_gray(str(file.parent / r.search_path))
            gx, gy = float(r.gt_x_corr), float(r.gt_y_corr)
            k = float(r.barrel_distortion_k)
            tpl = make_template(ref, r.magnification, r.rotation_deg)
            nx, npk = estimate_x(sea, tpl, gx, gy)
            sx, sy = radial(gx, gy, k, sea.shape)
            flat = undistort(sea, k)
            rt = make_template(undistort(ref, k * 0.3), r.magnification, r.rotation_deg)
            fx, fpk = estimate_x(flat, rt, float(sx), float(sy))
            corrected_x, corrected_y = radial(fx, sy, k, sea.shape, inverse=True)
            rows.append(
                dict(
                    pair_id=r.pair_id,
                    severity=int(r.severity_level),
                    native_error=abs(nx - gx),
                    undistorted_error=float(
                        np.hypot(corrected_x - gx, corrected_y - gy)
                    ),
                    native_peak=npk,
                    undistorted_peak=fpk,
                    k=k,
                )
            )
    d = pd.DataFrame(rows)
    d.to_csv(a.output / "diagnostics.csv", index=False)
    out = {
        "scope": "oracle diagnostic using true centre, pose and barrel k; NOT deployable accuracy",
        "n": len(d),
        "groups": {},
    }
    for name, g in [
        ("all", d),
        *[(str(s), d[d.severity == s]) for s in sorted(d.severity.unique())],
    ]:
        out["groups"][name] = {"n": len(g)}
        for arm in ["native", "undistorted"]:
            e = g[arm + "_error"]
            out["groups"][name][arm] = {
                "within1": int((e < 1).sum()),
                "fraction": float((e < 1).mean()),
                "unavailable": int(e.isna().sum()),
                "median_available_error": float(e.median()),
            }
    (a.output / "results.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        run(a)
    except Exception:
        (a.output / "FAILED.txt").write_text(traceback.format_exc())
        raise

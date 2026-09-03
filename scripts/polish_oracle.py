#!/usr/bin/env python3
"""Is the pose-polish objective's optimum at the true pose, or somewhere else?

Scale credit is 8.99/10 and rotation 8.98/10, and both tiers are unforgiving:
1.00 within 1% / 0.25 deg, then 0.60. Set B sits at a median 0.721% scale error
with 38.8% of pairs outside the 1% tier, so a little more accuracy is worth
about a point on each axis. There are only two reasons it could be missing:

  * the search does not find the optimum of its objective  -> fixable by
    searching better (wider bands, more iterations, better seed);
  * the objective's optimum is not at the true pose        -> searching harder
    makes it *worse*, and the objective itself has to change.

This distinguishes them. For pairs already localised correctly, it evaluates the
same windowed ZNCC that `polish_pose` maximises at both the polished pose and
the ground-truth pose, and compares. No network is involved -- the match
location is taken from an existing eval CSV -- so it costs only OpenCV.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def objective(reference, search, x, y, m, r):
    """The exact quantity polish_pose maximises, at a pinned canvas."""
    from driftsense.matching import make_template
    h, w = reference.shape[:2]
    hi = m * 1.03
    canvas = (max(int(np.floor(h / hi)), 1), max(int(np.floor(w / hi)), 1))
    th, tw = canvas
    pad = int(max(th, tw) * 0.6)
    y0 = max(int(round(y - th / 2.0)) - pad, 0)
    x0 = max(int(round(x - tw / 2.0)) - pad, 0)
    win = search[y0:int(round(y + th / 2.0)) + pad, x0:int(round(x + tw / 2.0)) + pad]
    t = make_template(reference, m, r, canvas=canvas)
    if t.shape[0] >= win.shape[0] or t.shape[1] >= win.shape[1]:
        return -np.inf
    return float(cv2.minMaxLoc(cv2.matchTemplate(win, t, cv2.TM_CCOEFF_NORMED))[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--set", default="B")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)
    import infer as I

    man = pd.concat([pd.read_csv(m).assign(_d=os.path.dirname(m))
                     for m in glob.glob("data/ext_p2/*/manifest.csv")])
    d = pd.read_csv(a.csv)
    d = d[(d.gt_found == 1) & (d["set"] == a.set)]
    d["err"] = np.hypot(d.x - d.gt_x, d.y - d.gt_y)
    d = d[d.err <= 5]
    j = d.merge(man[["pair_id", "reference_path", "search_path", "_d",
                     "magnification", "rotation_deg"]], on="pair_id")
    j = j.iloc[:: max(len(j) // a.n, 1)][:a.n]

    rows = []
    for _, r in j.iterrows():
        ref = I.read_gray(os.path.join(r._d, r.reference_path))
        sea = I.read_gray(os.path.join(r._d, r.search_path))
        o_pol = objective(ref, sea, r.x, r.y, r.scale, r.theta)
        o_gt = objective(ref, sea, r.x, r.y, r.magnification, r.rotation_deg)
        rows.append(dict(
            s_err=abs(r.scale - r.magnification) / r.magnification * 100,
            r_err=abs(r.theta - r.rotation_deg),
            o_pol=o_pol, o_gt=o_gt))
    t = pd.DataFrame(rows)

    print(f"set {a.set}, {len(t)} correctly-located pairs\n")
    print(f"  scale error   median {t.s_err.median():.3f}%   within 1%: {100*(t.s_err<=1).mean():.1f}%")
    print(f"  rot   error   median {t.r_err.median():.3f}deg within 0.25: {100*(t.r_err<=.25).mean():.1f}%")
    wins = (t.o_pol >= t.o_gt).mean()
    print(f"\n  objective at polished pose >= at TRUE pose: {100*wins:.1f}% of pairs")
    print(f"  mean objective  polished {t.o_pol.mean():.5f}   true {t.o_gt.mean():.5f}"
          f"   diff {t.o_pol.mean()-t.o_gt.mean():+.5f}")
    print()
    if wins > 0.6:
        print("  => The search IS finding a better objective value than the truth scores.")
        print("     The objective's optimum is displaced from the true pose. Searching")
        print("     harder cannot help; only a different objective can.")
    else:
        print("  => The true pose scores higher than what the search returned.")
        print("     This is a SEARCH failure and is fixable: wider bands, more")
        print("     iterations, or a better seed.")


if __name__ == "__main__":
    main()

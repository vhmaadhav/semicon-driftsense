#!/usr/bin/env python3
"""Can the pose-polish objective be made less noisy?

Established first (scripts/polish_oracle.py): the polish search is *not*
failing. In 82% of pairs it finds a higher objective value than the true pose
scores, so it locates its optimum correctly -- the optimum is simply displaced
from the truth. Established second: the displacement is zero-mean, +0.078%
against a 1.337% standard deviation on set B, so there is no bias to calibrate
away. The error is scatter.

Scatter is a variance problem, and the credit tiers are hard walls at 1% and
0.25 deg, so halving the scatter is worth roughly a point on each pose axis:
set A already achieves std 0.535% and 93.8% inside the tier, set B has std
1.337% and 63.5%.

Three ways to reduce it, all cheap and none changing the architecture:

  band     correlate difference-of-Gaussians images instead of raw ones. The
           coarse hypothesis sweep already does this; the polish does not. Low
           frequencies (vignette, charging, dose drift) carry no pose
           information and only add variance to the peak height.
  quad     fit a parabola through samples of the objective near the optimum and
           take the vertex, instead of returning golden section's final probe.
           Averaging several samples suppresses the noise that displaces any
           single argmax.
  both     the two together.

Run with the match location taken from an existing eval CSV, so no network is
involved and this measures the polish alone.
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


def make_fit(reference, search, x, y, m0, band):
    from driftsense.matching import _band, make_template
    h, w = reference.shape[:2]
    hi = m0 * 1.03
    canvas = (max(int(np.floor(h / hi)), 1), max(int(np.floor(w / hi)), 1))
    th, tw = canvas
    pad = int(max(th, tw) * 0.6)
    y0 = max(int(round(y - th / 2.0)) - pad, 0)
    x0 = max(int(round(x - tw / 2.0)) - pad, 0)
    win = search[y0:int(round(y + th / 2.0)) + pad, x0:int(round(x + tw / 2.0)) + pad]
    winf = _band(win) if band else win.astype(np.float32)

    def fit(mm, rr):
        t = make_template(reference, mm, rr, canvas=canvas)
        if t.shape[0] >= winf.shape[0] or t.shape[1] >= winf.shape[1]:
            return -np.inf
        tf = _band(t) if band else t.astype(np.float32)
        return float(cv2.minMaxLoc(cv2.matchTemplate(winf, tf, cv2.TM_CCOEFF_NORMED))[1])
    return fit


def quad_vertex(f, c, half, n=7):
    """Sample the objective on a grid around `c` and return the parabola vertex.

    Uses every sample rather than only the best one, so independent noise on the
    samples partly cancels instead of choosing the location of the luckiest.
    """
    xs = np.linspace(c - half, c + half, n)
    ys = np.array([f(v) for v in xs])
    ok = np.isfinite(ys)
    if ok.sum() < 3:
        return c
    xs, ys = xs[ok], ys[ok]
    # Weight towards the top of the peak; the tails are not parabolic.
    keep = ys >= ys.max() - 0.5 * (ys.max() - ys.min() + 1e-12)
    if keep.sum() >= 3:
        xs, ys = xs[keep], ys[keep]
    a, b, _ = np.polyfit(xs, ys, 2)
    if a >= -1e-12:
        return float(xs[int(np.argmax(ys))])
    return float(np.clip(-b / (2 * a), c - half, c + half))


def polish(fit, m, r, mode, scale_band=0.03, rot_band=0.8, rounds=2, iters=7):
    from driftsense.matching import _golden_max
    ds, dr = m * scale_band, rot_band
    for _ in range(rounds):
        if mode in ("quad", "both"):
            m = quad_vertex(lambda v: fit(v, r), m, ds)
            r = quad_vertex(lambda v: fit(m, v), r, dr)
        else:
            m, _ = _golden_max(lambda v: fit(v, r), m - ds, m + ds, iters)
            r, _ = _golden_max(lambda v: fit(m, v), r - dr, r + dr, iters)
        ds, dr = ds / 3.0, dr / 3.0
    return m, r


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

    # Seed every variant from the same starting point the shipped polish gets,
    # backed out of the coarse estimate, so this compares polishes and not seeds.
    modes = ["shipped", "band", "quad", "both"]
    out = {k: {"s": [], "r": []} for k in modes}
    for _, row in j.iterrows():
        ref = I.read_gray(os.path.join(row._d, row.reference_path))
        sea = I.read_gray(os.path.join(row._d, row.search_path))
        for mode in modes:
            fit = make_fit(ref, sea, row.x, row.y, row.scale, band=mode in ("band", "both"))
            m, r = polish(fit, row.scale, row.theta, mode)
            out[mode]["s"].append(abs(m - row.magnification) / row.magnification * 100)
            out[mode]["r"].append(abs(r - row.rotation_deg))

    print(f"set {a.set}, {len(j)} correctly-located pairs. "
          f"Polish re-run from the shipped estimate.\n")
    print(f"{'variant':<10}{'scale med%':>12}{'<=1%':>9}{'scale std':>11}"
          f"{'rot med':>10}{'<=0.25':>9}{'rot std':>10}")
    print("-" * 71)
    base = None
    for mode in modes:
        s = np.array(out[mode]["s"]); r = np.array(out[mode]["r"])
        print(f"{mode:<10}{np.median(s):>12.3f}{100*(s<=1).mean():>8.1f}%{s.std():>11.3f}"
              f"{np.median(r):>10.3f}{100*(r<=.25).mean():>8.1f}%{r.std():>10.3f}")
        if mode == "shipped":
            base = (100*(s<=1).mean(), 100*(r<=.25).mean())
    print()
    for mode in modes[1:]:
        s = np.array(out[mode]["s"]); r = np.array(out[mode]["r"])
        ds = (100*(s<=1).mean() - base[0]) / 100 * 10
        dr = (100*(r<=.25).mean() - base[1]) / 100 * 10
        print(f"  {mode:<6} projected on this set: scale {ds:+.2f} pts, rotation {dr:+.2f} pts"
              f"  (set {a.set} only, before the A/B mix)")


if __name__ == "__main__":
    main()

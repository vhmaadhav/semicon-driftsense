#!/usr/bin/env python3
"""Can we recover the centre-row drift and win back the sub-pixel tier?

The generator distorts the search frame row by row -- `out(y, x) = in(y, x +
row_shift[y])` -- and then labels the pair with `correct_gt`, which subtracts
**`row_shift[centre_row]`**, the shift of the single row the target centre
falls on.

Template matching aligns the whole ~100-row template at once, so what it
recovers is approximately the *mean* shift across those rows. The residual is
therefore

    error_x  ~=  row_shift[centre] - mean(row_shift over the template)

which for i.i.d. jitter of scale sigma has standard deviation ~sigma. That is
precisely the measured behaviour: localisation error runs at a near-constant
0.72x the per-pair `drift_jitter_px` across a fourfold range of drift.

So the <=1 px tier is not blocked by a precision floor. It is blocked by
estimating the wrong statistic: the mean, where the label wants one row.

This probe measures three things on pairs with known ground truth:

1. the oracle -- correct x by the *true* centre-row deviation, which bounds
   what any estimator of it could buy;
2. a real estimator -- recover the per-row deviation by correlating each row
   of the matched window against the corresponding template row;
3. whether the estimator's residual is small enough to be worth shipping.

**Caveat that decides shipping, not accuracy:** this correction is tied to the
labelling convention, not to the physics. If the organisers' generator labels
the undistorted position (or the mean), applying it would *add* error. Treat a
positive result here as evidence the mechanism is real, and the decision to
enable it as a separate bet on the convention.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)


def parabolic(a, b, c):
    d = a - 2.0 * b + c
    return 0.0 if abs(d) < 1e-12 else float(np.clip(0.5 * (a - c) / d, -1, 1))


def estimate_row_deviation(search, template, x, y, radius=4, smooth=3):
    """Per-row horizontal deviation from the template's global alignment.

    Returns (deviation_at_centre_row, profile). Each row of the matched window
    is correlated against the corresponding template row over +/-radius px and
    interpolated to sub-pixel; the profile is then de-meaned, because only the
    *deviation* from the global alignment is what the label disagrees with.
    """
    th, tw = template.shape
    x0 = int(round(x - tw / 2.0))
    y0 = int(round(y - th / 2.0))
    h, w = search.shape
    if x0 - radius < 0 or y0 < 0 or x0 + tw + radius > w or y0 + th > h:
        return 0.0, None

    win = search[y0:y0 + th, x0 - radius:x0 + tw + radius].astype(np.float32)
    tpl = template.astype(np.float32)

    # Vectorised over rows and shifts together. The scalar version was a
    # Python loop of th x (2r+1) normalised dot products per pair, which made
    # the probe slower than the inference it was measuring.
    tpl_c = tpl - tpl.mean(axis=1, keepdims=True)
    tpl_n = np.linalg.norm(tpl_c, axis=1)                      # (th,)
    shifts = 2 * radius + 1
    # (th, shifts, tw) view over the window without copying
    from numpy.lib.stride_tricks import sliding_window_view
    segs = sliding_window_view(win, tw, axis=1)                # (th, shifts, tw)
    segs_c = segs - segs.mean(axis=2, keepdims=True)
    segs_n = np.linalg.norm(segs_c, axis=2)                    # (th, shifts)
    num = np.einsum("rst,rt->rs", segs_c, tpl_c)
    den = segs_n * tpl_n[:, None]
    scores = np.where(den > 1e-9, num / np.maximum(den, 1e-9), -1.0)

    bi = scores.argmax(axis=1)
    rows_ix = np.arange(th)
    devs = (bi - radius).astype(np.float32)
    inner = (bi > 0) & (bi < shifts - 1)
    if inner.any():
        a = scores[rows_ix[inner], bi[inner] - 1]
        b = scores[rows_ix[inner], bi[inner]]
        c = scores[rows_ix[inner], bi[inner] + 1]
        d = a - 2.0 * b + c
        sub = np.where(np.abs(d) < 1e-12, 0.0, np.clip(0.5 * (a - c) / np.where(np.abs(d) < 1e-12, 1.0, d), -1, 1))
        devs[inner] += sub.astype(np.float32)
    devs[tpl_n < 1e-6] = 0.0

    if smooth > 1:
        k = np.ones(smooth, dtype=np.float32) / smooth
        devs = np.convolve(devs, k, mode="same")
    devs = devs - devs.mean()          # only the deviation from global matters
    return float(devs[th // 2]), devs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--results", default=".agents/ext_ship_conf.csv")
    ap.add_argument("--limit", type=int, default=120)
    a = ap.parse_args()

    import pandas as pd
    import infer as I
    from driftsense.matching import locate_phase2, make_template

    res = pd.read_csv(a.results)
    res["err"] = np.where(res.gt_found == 1,
                          np.hypot(res.x - res.gt_x, res.y - res.gt_y), np.nan)
    ok = set(res[(res.gt_found == 1) & (res.err <= 5)].pair_id)

    model, device = I.load_model(os.path.join(HERE, "weights", "driftsense.pt"))
    rows = []
    for d in a.shards:
        man = pd.read_csv(os.path.join(d, "manifest.csv"))
        man = man[(man.found == 1) & (man.pair_id.isin(ok))].head(a.limit)
        for _, r in man.iterrows():
            ref = I.read_gray(os.path.join(d, r.reference_path))
            sea = I.read_gray(os.path.join(d, r.search_path))
            o = locate_phase2(model, ref, sea, device, refine=True)
            tpl = make_template(ref, o["scale"], o["theta"])
            dev, _ = estimate_row_deviation(sea, tpl, o["x"], o["y"])
            ex = o["x"] - r.gt_x_corr
            rows.append({"set": r.phase2_set, "drift": r.drift_jitter_px,
                         "err": float(np.hypot(ex, o["y"] - r.gt_y_corr)),
                         "ex": float(ex), "dev": dev,
                         "err_corr": float(np.hypot(ex - dev, o["y"] - r.gt_y_corr))})
            # Write as we go. A previous run of this probe was killed at 900 s
            # and lost everything because it only wrote at the end -- the same
            # mistake already fixed in fit_rejector.py.
            if len(rows) % 20 == 0:
                pd.DataFrame(rows).to_csv(".agents/row_shift_probe.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(".agents/row_shift_probe.csv", index=False)

    print(f"{len(df)} pairs (already within 5 px, so this is purely the precision tier)\n")
    print(f"{'':<26}{'median err px':>16}{'<=1px':>9}{'<=2px':>9}")
    print("-" * 60)
    for name, col in (("as shipped", "err"), ("row-shift corrected", "err_corr")):
        print(f"{name:<26}{df[col].median():>16.3f}{100*(df[col]<=1).mean():>8.0f}%"
              f"{100*(df[col]<=2).mean():>8.0f}%")
    print(f"\ncorr(estimated deviation, actual x error) = "
          f"{np.corrcoef(df.dev, df.ex)[0,1]:+.3f}")
    print("A correlation near +1 would mean the estimator recovers the residual;")
    print("near 0 means the per-row signal is too weak to measure at this noise.")
    for s in sorted(df["set"].unique()):
        g = df[df["set"] == s]
        print(f"  set {s}: median {g.err.median():.3f} -> {g.err_corr.median():.3f} px, "
              f"<=1px {100*(g.err<=1).mean():.0f}% -> {100*(g.err_corr<=1).mean():.0f}%")


if __name__ == "__main__":
    main()

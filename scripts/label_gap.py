#!/usr/bin/env python3
"""Is the sub-pixel error a property of the label, or of our matcher?

Localisation error measures a near-constant 0.73x the frame's raster drift. Two
explanations predict that equally well and imply opposite work:

  definitional  raster drift wobbles each scan row differently, so the imaged
                pattern is not a rigid copy of the reference. The best rigid
                alignment is a blend of the per-row displacements; the label is
                the *centre's* displacement. They differ by ~the drift scale,
                and no matcher closes that.
  ours          the label sits at the correlation optimum and we simply do not
                find it, in which case it is worth real points.

The test: take the TRUE scale and rotation from the manifest, so pose error is
removed entirely, and evaluate ZNCC on a fine sub-pixel grid around the label.
Where the maximum lands decides it.

  max at the label            -> our error, closable
  max displaced ~0.73*jitter  -> definitional, no matcher closes it

Also evaluates a centre-weighted ZNCC. If the label is the centre row's
displacement, weighting the correlation towards the template centre should pull
the optimum back towards the label -- that would be a cheap way to close part of
the gap.
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


def zncc(t, p, w=None):
    t = t.astype(np.float32); p = p.astype(np.float32)
    if w is None:
        a, b = t - t.mean(), p - p.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / d) if d > 1e-9 else 0.0
    sw = w.sum()
    a, b = t - (t * w).sum() / sw, p - (p * w).sum() / sw
    d = np.sqrt((w * a * a).sum() * (w * b * b).sum())
    return float((w * a * b).sum() / d) if d > 1e-9 else 0.0


def grid_argmax(search, tmpl, cx, cy, w=None, half=2.0, step=0.2):
    th, tw = tmpl.shape
    offs = np.arange(-half, half + 1e-9, step)
    best, bxy = -2.0, (0.0, 0.0)
    for dy in offs:
        for dx in offs:
            p = cv2.getRectSubPix(search, (tw, th), (float(cx + dx), float(cy + dy)))
            v = zncc(tmpl, p, w)
            if v > best:
                best, bxy = v, (float(dx), float(dy))
    return bxy, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="B")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    cv2.setNumThreads(a.threads)
    import infer as I
    from driftsense.matching import make_template

    man = pd.concat([pd.read_csv(m).assign(_d=os.path.dirname(m))
                     for m in glob.glob(f"data/ext_p2/*{a.set}*/manifest.csv")])
    man = man[man.found == 1]
    # Random, not strided. Manifests cycle severity 1,2,3,4, so any stride
    # sharing a factor with 4 samples a biased subset -- that aliasing has
    # invalidated three experiments on this project already.
    man = man.sample(n=min(a.n, len(man)), random_state=0)

    rows = []
    for _, r in man.iterrows():
        ref = I.read_gray(os.path.join(r._d, r.reference_path))
        sea = I.read_gray(os.path.join(r._d, r.search_path)).astype(np.float32)
        t = make_template(ref, float(r.magnification), float(r.rotation_deg)).astype(np.float32)
        th, tw = t.shape
        yy, xx = np.mgrid[0:th, 0:tw]
        sig = max(th, tw) / 4.0
        wgt = np.exp(-(((yy - th / 2) ** 2 + (xx - tw / 2) ** 2) / (2 * sig * sig))).astype(np.float32)

        (dx, dy), _ = grid_argmax(sea, t, r.gt_x_corr, r.gt_y_corr)
        (wx, wy), _ = grid_argmax(sea, t, r.gt_x_corr, r.gt_y_corr, w=wgt)
        rows.append(dict(jit=float(r.drift_jitter_px), sev=int(r.severity_level),
                         d=float(np.hypot(dx, dy)), dw=float(np.hypot(wx, wy))))
    t = pd.DataFrame(rows)

    print(f"set {a.set}, {len(t)} present pairs, TRUE pose, sub-pixel ZNCC grid +/-2px @0.2px\n")
    print("Distance from the label to the ZNCC optimum (pose error removed):")
    print(f"  plain ZNCC          median {t.d.median():.3f} px   mean {t.d.mean():.3f}")
    print(f"  centre-weighted     median {t.dw.median():.3f} px   mean {t.dw.mean():.3f}")
    print(f"  drift jitter        median {t.jit.median():.3f} px")
    print(f"  optimum/jitter ratio       {(t.d/t.jit).median():.3f}")
    print(f"\n  our shipped pipeline on set B: median error 0.744 px, err/jitter 0.730")
    print("\nby severity:")
    print(t.groupby("sev").agg(n=("d", "size"), jitter=("jit", "median"),
                               opt_dist=("d", "median"), weighted=("dw", "median")).round(3).to_string())
    print()
    ratio = (t.d / t.jit).median()
    if ratio > 0.4:
        print(f"=> The ZNCC optimum itself sits {ratio:.2f}x the drift away from the label,")
        print("   with the true pose supplied. The gap is DEFINITIONAL: correlation and")
        print("   the label measure different things, and no matcher closes it.")
    else:
        print(f"=> The ZNCC optimum sits close to the label ({ratio:.2f}x drift). The error")
        print("   is OURS, not the label's, and is worth chasing.")
    if t.dw.median() < t.d.median() * 0.9:
        print(f"   Centre-weighting helps: {t.d.median():.3f} -> {t.dw.median():.3f} px. Worth wiring in.")


if __name__ == "__main__":
    main()

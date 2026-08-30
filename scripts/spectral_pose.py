#!/usr/bin/env python3
"""Estimate magnification and rotation from the reciprocal lattice, not by search.

The coarse pose sweep correlates a rendered template against the search frame
at each candidate scale. That is expensive, and on degraded Set B frames it is
noise-dominated: measured on 90 Set B failures, **76% never had a correct
candidate generated at all**, so no amount of better ranking can rescue them.

This is a different route to the same quantity, and it exploits the one thing
the domain guarantees -- DRAM and FinFET layouts are periodic by construction,
so their power spectra carry sharp reciprocal-lattice peaks.

For a lattice of physical pitch P nm:

* the reference is 1 nm/px, so the pitch spans P px and appears at
  N/P cycles per N-pixel image;
* the search is m nm/px for magnification m, so the same pitch spans P/m px
  and appears at N*m/P cycles.

The ratio of the two peak radii is therefore **exactly m**, and the angular
offset between the two peak constellations is the rotation. Both fall out of
the magnitude spectrum, which is *translation invariant* -- so unlike the
correlation sweep this needs no localisation first, which is what makes it a
genuinely independent estimate rather than a cheaper version of the same thing.

Classical basis: the Fourier-Mellin / log-polar construction, in which rotation
becomes a shift in theta and scale a shift in log-r.

* Reddy, B. S. and Chatterji, B. N. "An FFT-Based Technique for Translation,
  Rotation and Scale-Invariant Image Registration", IEEE TIP 5(8), 1996.
* Zokai, S. and Wolberg, G. "Image Registration Using Log-Polar Mappings for
  Recovery of Large-Scale Similarity and Projective Transformations",
  IEEE TIP 14(10), 2005.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def _spectrum(img: np.ndarray, n: int = 512) -> np.ndarray:
    """Log power spectrum, Hann-windowed and centred."""
    f = img.astype(np.float32)
    f = cv2.resize(f, (n, n), interpolation=cv2.INTER_AREA)
    f = f - f.mean()
    w = np.outer(np.hanning(n), np.hanning(n)).astype(np.float32)
    F = np.fft.fftshift(np.abs(np.fft.fft2(f * w)))
    return np.log1p(F)


def _lattice_peaks(S: np.ndarray, k: int = 12, r_min: int = 6):
    """Strongest spectral peaks outside the DC disc, as (radius, angle, value).

    Angles are folded to [0, pi) because a reciprocal lattice is centro-
    symmetric: a peak and its opposite carry the same information.
    """
    n = S.shape[0]
    c = n // 2
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(yy - c, xx - c)
    mask = (r >= r_min) & (r < c - 2)
    work = np.where(mask, S, -np.inf)
    # Non-maximum suppression so one broad ridge does not supply every peak.
    pooled = cv2.dilate(np.where(mask, S, 0).astype(np.float32), np.ones((5, 5), np.uint8))
    cand = np.argwhere((work >= pooled - 1e-6) & mask)
    if not len(cand):
        return []
    vals = S[cand[:, 0], cand[:, 1]]
    order = np.argsort(-vals)[:k]
    out = []
    for i in order:
        y, x = cand[i]
        rad = float(np.hypot(y - c, x - c))
        ang = float(np.arctan2(y - c, x - c)) % np.pi
        out.append((rad, ang, float(vals[i])))
    return out


def spectral_pose(reference: np.ndarray, search: np.ndarray,
                  scale_bounds=(8.0, 12.0), rot_bounds=(-5.0, 5.0), n: int = 512):
    """Return (magnification, rotation_deg, confidence).

    Both spectra are computed on an n x n resample, so a peak radius is in
    cycles-per-resampled-image and the *ratio* is what carries the physics --
    the resample factor cancels as long as both images have the same pixel
    dimensions, which the contract guarantees (1000 x 1000).
    """
    Sr, Ss = _spectrum(reference, n), _spectrum(search, n)
    pr, ps = _lattice_peaks(Sr), _lattice_peaks(Ss)
    if not pr or not ps:
        return float(np.mean(scale_bounds)), 0.0, 0.0

    lo, hi = scale_bounds
    votes = []
    for rr, ar, vr in pr:
        for rs, as_, vs in ps:
            if rs <= 1e-6:
                continue
            m = rr / rs                      # reference cycles / search cycles
            if not (lo <= m <= hi):
                continue
            d = np.degrees(((ar - as_ + np.pi / 2) % np.pi) - np.pi / 2)
            if not (rot_bounds[0] <= d <= rot_bounds[1]):
                continue
            votes.append((m, d, vr * vs))
    if not votes:
        return float(np.mean(scale_bounds)), 0.0, 0.0

    v = np.array(votes)
    w = v[:, 2] / v[:, 2].sum()
    m_hat = float((v[:, 0] * w).sum())
    r_hat = float((v[:, 1] * w).sum())
    # Confidence: how concentrated the surviving votes are in scale.
    spread = float(np.sqrt(((v[:, 0] - m_hat) ** 2 * w).sum()))
    conf = float(np.exp(-spread / 0.5)) * min(len(votes) / 20.0, 1.0)
    return m_hat, r_hat, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--n", type=int, default=512)
    a = ap.parse_args()

    HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, HERE)
    import pandas as pd
    import infer as I

    rows = []
    for d in a.shards:
        m = pd.read_csv(os.path.join(d, "manifest.csv"))
        m = m[m.found == 1].head(a.limit)
        for _, r in m.iterrows():
            ref = I.read_gray(os.path.join(d, r.reference_path))
            sea = I.read_gray(os.path.join(d, r.search_path))
            mh, rh, c = spectral_pose(ref, sea, n=a.n)
            rows.append({"set": r.phase2_set, "sev": r.get("severity_level", -1),
                         "gt_m": r.magnification, "gt_r": r.rotation_deg,
                         "m": mh, "r": rh, "conf": c,
                         "m_err": abs(mh - r.magnification) / r.magnification,
                         "r_err": abs(rh - r.rotation_deg)})
    df = pd.DataFrame(rows)
    df.to_csv(".agents/spectral_pose.csv", index=False)

    print(f"{len(df)} present pairs\n")
    print(f"{'set':<5}{'n':>5}{'med |m| err %':>16}{'<=1%':>8}{'<=2%':>8}{'<=5%':>8}"
          f"{'med |rot| err':>15}{'<=1deg':>9}")
    print("-" * 74)
    for s in sorted(df["set"].unique()):
        g = df[df["set"] == s]
        print(f"{s:<5}{len(g):>5}{100*g.m_err.median():>15.2f}%{100*(g.m_err<=.01).mean():>7.0f}%"
              f"{100*(g.m_err<=.02).mean():>7.0f}%{100*(g.m_err<=.05).mean():>7.0f}%"
              f"{g.r_err.median():>14.2f}{100*(g.r_err<=1).mean():>8.0f}%")
    print("\nA sweep-free pose is useful if it lands inside the +/-3% window the")
    print("local refine can close, even when it is not accurate enough to report.")
    print(f"within 3% of true magnification: {100*(df.m_err<=0.03).mean():.0f}% of pairs")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""What distinguishes the pairs we get wrong from the pairs we get right?

The last Phase 2 gain came from a diagnostic, not a guess: every localisation
failure turned out to be a wrong scale-basin lock-on, which redirected the work
from retraining to the pose search. This script exists so the *next* change is
chosen the same way.

For each candidate explanatory variable it reports the value distribution among
failures against successes, plus a standardised effect size, so a variable that
merely looks different because failures are rare does not get mistaken for a
cause.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--manifest-glob", default="data/ext_p2/*/manifest.csv",
                    help="joined on pair_id to recover generator parameters")
    ap.add_argument("--tol", type=float, default=5.0)
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    d["err"] = np.where(d.gt_found == 1, np.hypot(d.x - d.gt_x, d.y - d.gt_y), np.nan)

    # Bring in the generator parameters that our results CSV does not carry.
    import glob
    mans = [pd.read_csv(p) for p in sorted(glob.glob(a.manifest_glob))]
    if mans:
        man = pd.concat(mans, ignore_index=True)
        keep = [c for c in ("pair_id", "polygon_scale_fraction_requested",
                            "drift_jitter_px", "shear_amplitude_px",
                            "detector_noise_sigma_search", "dose_search",
                            "speckle_sigma", "charging_streak_prob",
                            "barrel_distortion_k", "edge_brightening",
                            "beam_spot_size_nm", "astigmatism_ratio",
                            "salt_pepper_prob", "gamma", "collapse_threshold_nm",
                            "mat_size_nm", "linewidth_bias_nm") if c in man.columns]
        d = d.merge(man[keep], on="pair_id", how="left")

    p = d[d.gt_found == 1].copy()
    p["absrot"] = p.gt_rot.abs()
    p["scale_off"] = (p.gt_scale - 10.0).abs()
    p["fail"] = p.err > a.tol

    nf = int(p.fail.sum())
    print(f"{len(p)} present pairs, {nf} beyond {a.tol} px ({100*nf/len(p):.1f}%)\n")
    if nf == 0:
        print("no failures to explain")
        return

    cand = ["absrot", "scale_off", "gt_scale", "gt_rot", "severity",
            "polygon_scale_fraction_requested", "drift_jitter_px",
            "shear_amplitude_px", "detector_noise_sigma_search", "dose_search",
            "speckle_sigma", "charging_streak_prob", "barrel_distortion_k",
            "edge_brightening", "beam_spot_size_nm", "astigmatism_ratio",
            "salt_pepper_prob", "collapse_threshold_nm", "linewidth_bias_nm",
            "score", "zncc"]
    cand = [c for c in cand if c in p.columns]

    rows = []
    for c in cand:
        f, s = p.loc[p.fail, c], p.loc[~p.fail, c]
        if f.notna().sum() < 2:
            continue
        rows.append((c, f.median(), s.median(), cohen_d(f, s)))
    rows.sort(key=lambda r: -abs(r[3] if np.isfinite(r[3]) else 0))

    print(f"{'variable':<36}{'fail med':>11}{'ok med':>11}{'effect d':>11}")
    print("-" * 69)
    for c, fm, sm, dd in rows:
        flag = "  <<<" if abs(dd) >= 0.5 else ""
        print(f"{c:<36}{fm:>11.4f}{sm:>11.4f}{dd:>11.3f}{flag}")

    print("\nFailure rate by set / severity / architecture:")
    for key in ("set", "severity", "architecture"):
        if key not in p.columns:
            continue
        g = p.groupby(key).fail.agg(["mean", "size"]).sort_values("mean", ascending=False)
        print(f"  by {key}:")
        for k, r in g.iterrows():
            print(f"    {str(k):<22}{100*r['mean']:6.1f}%   n={int(r['size'])}")

    print("\nInterpretation guide: |d| >= 0.5 is a real separation worth acting on;")
    print("|d| < 0.2 means that variable does not explain these failures.")


if __name__ == "__main__":
    main()

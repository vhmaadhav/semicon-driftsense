#!/usr/bin/env python3
"""Lattice-ambiguity analysis: how much of our residual error is group theory?

Issue #19. Every failure in this project traces back to one informal statement:
"the layout repeats, so a 100 x 100 template correlates almost equally well at
dozens of positions." That is a group-theoretic claim -- a periodic layout is
invariant under a lattice translation subgroup, and the positions genuinely
indistinguishable from the truth form a coset of it. This script measures that
subgroup instead of asserting it.

Three outputs:

1. LATTICE BASIS. `driftsense.matching.row_pitch` already recovers the dominant
   *horizontal* period from the template's mean row autocorrelation (resolvable
   on 94% of pairs, median 9.9 px). `lattice_basis` generalises that same
   Wiener-Khinchin estimator to a full 2-D basis (v1, v2). The recovered |v1_x|
   is cross-checked against `row_pitch` so the generalisation is anchored to the
   number already in the repo.

2. AMBIGUITY SCORE. For each present pair, the ZNCC between the posed template
   and the search frame at the true centre, and at its lattice-translated
   neighbours a*v1 + b*v2. The gap between the two IS the difficulty of the
   pair, derived from the data rather than from our model's own confidence --
   a different and more honest axis than anything in calibration.py.

3. SYMMETRY-BREAKING BUDGET. The mat/strip composition is what breaks the
   wallpaper symmetry at a larger scale (CITATIONS.md section 5 calls it "the
   strongest globally unique cue available to the matcher"). The manifest
   carries mat_size_nm and strip_width_nm, so the fraction of crops that can be
   globally disambiguated at all is computable rather than assumed.

Ships a report, not a code path: nothing here is imported by the decode.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import zlib
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from driftsense.matching import make_template, row_pitch


def autocorr2d(patch: np.ndarray) -> np.ndarray:
    """Zero-mean 2-D autocorrelation, origin-centred and peak-normalised.

    Circular (unpadded), matching `row_pitch`'s 1-D estimator so the 2-D basis
    and the shipped horizontal pitch are the same statistic in two dimensions.
    """
    t = patch.astype(np.float32)
    t = t - t.mean()
    f = np.fft.rfft2(t)
    ac = np.fft.irfft2(f * np.conj(f), s=t.shape)
    ac = np.fft.fftshift(ac)
    peak = ac.max()
    return ac / peak if peak > 0 else ac


def lattice_basis(patch: np.ndarray, min_lag: int = 4, max_lag: int = 40,
                  rel_thresh: float = 0.25):
    """Two shortest independent lattice vectors from the 2-D autocorrelation.

    Returns (v1, v2, ac) with each v as (dx, dy) floats, or None where the
    lattice is not resolvable. Autocorrelation is symmetric, so only the
    upper half-plane is searched; the mirrored vector is the same translation.
    """
    ac = autocorr2d(patch)
    h, w = ac.shape
    oy, ox = h // 2, w // 2

    # Local maxima on a 5x5 stencil, the same neighbourhood find_peaks uses.
    pooled = cv2.dilate(ac, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero((ac >= pooled - 1e-9) & (ac > rel_thresh))

    cands = []
    for y, x in zip(ys, xs):
        dy, dx = float(y - oy), float(x - ox)
        r = np.hypot(dx, dy)
        if r < min_lag or r > max_lag:
            continue
        # Half-plane: keep one of each +/- pair.
        if dy < 0 or (dy == 0 and dx < 0):
            continue
        cands.append((r, dx, dy, float(ac[y, x])))

    if not cands:
        return None, None, ac
    cands.sort()

    r1, v1x, v1y, _ = cands[0]
    v1 = (v1x, v1y)

    # v2 is the shortest candidate not collinear with v1. |cross| / (|a||b|) is
    # sin(angle); 0.25 keeps anything more than ~14 degrees off the v1 axis.
    v2 = None
    for r, dx, dy, _ in cands[1:]:
        sin_ang = abs(v1x * dy - v1y * dx) / (r1 * r)
        if sin_ang > 0.25:
            v2 = (dx, dy)
            break

    return v1, v2, ac


def zncc_at(search: np.ndarray, template: np.ndarray,
            cx: float, cy: float) -> float:
    """ZNCC between the template and the search frame at centre (cx, cy)."""
    th, tw = template.shape
    x0 = int(round(cx - tw / 2.0))
    y0 = int(round(cy - th / 2.0))
    h, w = search.shape
    if x0 < 0 or y0 < 0 or x0 + tw > w or y0 + th > h:
        return float("nan")
    win = search[y0:y0 + th, x0:x0 + tw].astype(np.float32)
    t = template.astype(np.float32)
    win = win - win.mean()
    t = t - t.mean()
    denom = np.sqrt((win * win).sum() * (t * t).sum())
    return float((win * t).sum() / denom) if denom > 0 else float("nan")


# The coset of lattice translations tested around the truth. (a, b) are the
# integer coefficients on (v1, v2); one full ring is enough to establish
# whether the neighbours are indistinguishable.
NEIGHBOURS = [(1, 0), (-1, 0), (0, 1), (0, -1),
              (1, 1), (1, -1), (-1, 1), (-1, -1)]


def lattice_coords(err: tuple[float, float], v1, v2):
    """Express an error vector in the lattice basis: err = a*v1 + b*v2.

    Returns (a, b, residual_px) where residual is the distance from the error
    to the nearest lattice point round(a)*v1 + round(b)*v2. A gross failure
    that is "one full repeat over" has integer (a, b) and a small residual;
    ordinary estimation error has a near-zero (a, b) instead.
    """
    m = np.array([[v1[0], v2[0]], [v1[1], v2[1]]], float)
    if abs(np.linalg.det(m)) < 1e-9:
        return float("nan"), float("nan"), float("nan")
    a, b = np.linalg.solve(m, np.array(err, float))
    snap = m @ np.array([round(a), round(b)], float)
    return float(a), float(b), float(np.hypot(*(np.array(err) - snap)))


def analyse_pair(row, root: Path, model=None, device=None):
    ref = cv2.imread(str(root / row.reference_path), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(str(root / row.search_path), cv2.IMREAD_GRAYSCALE)
    if ref is None or search is None:
        return None

    # Pose the template into the search frame with the ground-truth geometry,
    # so the pose search is out of the experiment (the #22 convention).
    tmpl = make_template(ref, factor=float(row.magnification),
                         rotation_deg=float(row.rotation_deg))

    out = {"id": row.id, "architecture": row.architecture,
           "severity": float(row.severity_continuous),
           "found": int(row.found)}

    # Symmetry-breaking geometry, in SEARCH pixels. The reference is fixed at
    # 1 nm/px and the search sweeps 8-12 nm/px (generator/README.md), so
    # `magnification` is exactly nm-per-search-pixel and the 100 px template
    # footprint follows from the pixel-size ratio.
    z = float(row.magnification)
    out["mat_px"] = float(row.mat_size_nm) / z
    out["strip_px"] = float(row.strip_width_nm) / z
    out["period_px"] = out["mat_px"] + out["strip_px"]

    v1, v2, _ = lattice_basis(tmpl)
    out["pitch_1d"] = row_pitch(tmpl)
    out["v1_resolved"] = v1 is not None
    out["v2_resolved"] = v2 is not None
    if v1 is not None:
        out["v1x"], out["v1y"] = v1
        out["v1_len"] = float(np.hypot(*v1))
    if v2 is not None:
        out["v2x"], out["v2y"] = v2
        out["v2_len"] = float(np.hypot(*v2))
    if v1 is not None and v2 is not None:
        # Cell area = |v1 x v2|: how much of the frame one repeat covers.
        out["cell_area"] = abs(v1[0] * v2[1] - v1[1] * v2[0])
        # The ambiguity coset: how many lattice translations fit inside one mat
        # before a strip breaks the symmetry. This is the number of positions a
        # matcher with no global context has to choose between.
        out["coset"] = out["mat_px"] / float(np.hypot(*v1))

    if not row.found or v1 is None or v2 is None:
        return out

    # gt_*_corr, not gt_*: the raw label is the crop centre mapped through the
    # render affine, which still carries the raster-drift sample of its centre
    # row. Measured here on 15 pairs, the drift-corrected label sits on the
    # local ZNCC optimum (median 0.67) while the raw one is 2-4 px off it
    # (median 0.37) -- scoring the ambiguity against the raw label would
    # attribute that offset to lattice competition.
    cx, cy = float(row.gt_x_corr), float(row.gt_y_corr)
    true_z = zncc_at(search, tmpl, cx, cy)
    out["zncc_true"] = true_z

    best_decoy, best_ab = -np.inf, None
    for a, b in NEIGHBOURS:
        dx = a * v1[0] + b * v2[0]
        dy = a * v1[1] + b * v2[1]
        z = zncc_at(search, tmpl, cx + dx, cy + dy)
        if np.isfinite(z) and z > best_decoy:
            best_decoy, best_ab = z, (a, b)
    if best_ab is None:
        return out

    out["zncc_decoy"] = best_decoy
    out["decoy_a"], out["decoy_b"] = best_ab
    # The ambiguity: how much correlation actually separates truth from the
    # nearest repeat. <= 0 means the lattice neighbour wins outright.
    out["margin"] = true_z - best_decoy
    out["decoy_ratio"] = best_decoy / true_z if true_z > 0 else float("nan")

    # DISTANCE-MATCHED CONTROL. Without this the result is unfalsifiable: a
    # smooth image correlates with itself at ANY small displacement, so a high
    # decoy score need not be the lattice's doing. Each lattice neighbour is
    # paired with a random direction at the SAME radius, so the only difference
    # between the two populations is whether the displacement is a lattice
    # translation. If the lattice is real, its score must beat this.
    # crc32, not hash(): Python randomises string hashing per process, so
    # seeding from hash() gave a different control on every run (it moved the
    # measured lattice excess between +0.104 and +0.122 on identical data).
    # This script's numbers have to be reproducible from the script alone.
    rng = np.random.default_rng(zlib.crc32(str(row.id).encode()))
    best_rand = -np.inf
    for a, b in NEIGHBOURS:
        r = np.hypot(a * v1[0] + b * v2[0], a * v1[1] + b * v2[1])
        th = rng.uniform(0, 2 * np.pi)
        z = zncc_at(search, tmpl, cx + r * np.cos(th), cy + r * np.sin(th))
        if np.isfinite(z) and z > best_rand:
            best_rand = z
    if np.isfinite(best_rand):
        out["zncc_rand"] = best_rand
        out["margin_rand"] = true_z - best_rand
        # The quantity the whole issue turns on: how much of the decoy's
        # correlation is periodicity rather than ordinary image smoothness.
        out["lattice_excess"] = best_decoy - best_rand

    if model is not None:
        from driftsense.config import (SHIPPED_BAND, SHIPPED_SUBPIXEL_ROWS,
                                       SHIPPED_VERIFICATION)
        from driftsense.matching import locate_phase2
        res = locate_phase2(model, ref, search, device, refine=True,
                            band=SHIPPED_BAND, verification=SHIPPED_VERIFICATION,
                            subpixel_rows=SHIPPED_SUBPIXEL_ROWS)
        ex, ey = res["x"] - cx, res["y"] - cy
        out["pred_x"], out["pred_y"] = res["x"], res["y"]
        out["err_px"] = float(np.hypot(ex, ey))
        a_, b_, resid = lattice_coords((ex, ey), v1, v2)
        out["lat_a"], out["lat_b"], out["lat_resid"] = a_, b_, resid
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/holdout_p2",
                    help="dataset root holding manifest.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="analyse only the first N pairs (0 = all)")
    ap.add_argument("--out", default="", help="write per-pair CSV here")
    ap.add_argument("--decode", action="store_true",
                    help="also run the shipped decode and decompose its error "
                         "onto the lattice basis (slow: ~3 s/pair)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--report-only", metavar="CSV",
                    help="re-print the report from an existing CSV, including "
                         "the partial one an interrupted run leaves behind")
    a = ap.parse_args()

    if a.report_only:
        report(pd.read_csv(a.report_only))
        return

    root = Path(a.data)
    man = pd.read_csv(root / "manifest.csv")
    if a.limit:
        man = man.head(a.limit)

    model = device = None
    if a.decode:
        import torch
        import infer as I
        torch.set_num_threads(a.threads)
        model, device = I.load_model(I.DEFAULT_WEIGHTS)

    rows = []
    for i, row in enumerate(man.itertuples(index=False)):
        r = analyse_pair(row, root, model, device)
        if r is not None:
            rows.append(r)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(man)}", file=sys.stderr, flush=True)
            # Checkpoint. --decode takes ~3 s/pair, so a full pass is long
            # enough that it will sometimes be interrupted; writing only at the
            # end means an interrupted run yields nothing at all, which is how
            # one 200-pair decode was already lost.
            if a.out and rows:
                pd.DataFrame(rows).to_csv(a.out, index=False)

    d = pd.DataFrame(rows)
    if a.out:
        d.to_csv(a.out, index=False)
        print(f"wrote {a.out}  ({len(d)} rows)")
    report(d)


def report(d: pd.DataFrame):
    n = len(d)
    print(f"\n=== LATTICE RESOLVABILITY (n={n}) ===")
    print(f"v1 resolved       {d.v1_resolved.mean():6.1%}")
    print(f"v1 and v2         {d.v2_resolved.mean():6.1%}")
    ok = d[d.v2_resolved]
    if len(ok):
        print(f"|v1| median       {ok.v1_len.median():6.2f} px")
        print(f"|v2| median       {ok.v2_len.median():6.2f} px")
        print(f"cell area median  {ok.cell_area.median():6.1f} px^2")
    # Cross-check against the shipped 1-D estimator this generalises.
    both = d[d.v1_resolved & d.pitch_1d.notna()]
    if len(both):
        print(f"row_pitch median  {both.pitch_1d.median():6.2f} px  "
              f"(shipped 1-D estimator, reported 9.9)")

    ok2 = d[d.v2_resolved]
    if len(ok2):
        print(f"\n=== SYMMETRY-BREAKING BUDGET (n={len(ok2)}) ===")
        print(f"mat width      median {ok2.mat_px.median():6.1f} px "
              f"(search frame is 1000 px)")
        print(f"strip width    median {ok2.strip_px.median():6.1f} px")
        print(f"mat+strip      median {ok2.period_px.median():6.1f} px  "
              f"-> {1000 / ok2.period_px.median():.1f} periods per frame")
        print(f"strip fraction median {(ok2.strip_px / ok2.period_px).median():6.1%} "
              f"of the layout carries a globally unique cue")
        print(f"AMBIGUITY COSET median {ok2.coset.median():6.1f} lattice "
              f"translations fit inside one mat")
        print(f"  a 100 px template spans "
              f"{(100 / ok2.period_px).median():.2f} of a mat+strip period, so a "
              f"crop landing mid-mat sees no boundary at all")

    pres = d[(d.found == 1) & d.get("margin", pd.Series(dtype=float)).notna()] \
        if "margin" in d else pd.DataFrame()
    if len(pres):
        print(f"\n=== AMBIGUITY (present pairs, n={len(pres)}) ===")
        print(f"ZNCC at truth       median {pres.zncc_true.median():6.3f}")
        print(f"ZNCC at best decoy  median {pres.zncc_decoy.median():6.3f}")
        print(f"margin              median {pres.margin.median():6.3f}  "
              f"p10 {pres.margin.quantile(0.10):6.3f}")
        print(f"decoy/true ratio    median {pres.decoy_ratio.median():6.3f}")
        print(f"lattice neighbour beats truth: {(pres.margin <= 0).mean():6.1%}")
        for thr in (0.02, 0.05, 0.10):
            print(f"margin < {thr:.2f}  {(pres.margin < thr).mean():6.1%}")

        if "zncc_rand" in pres.columns and pres.zncc_rand.notna().any():
            c = pres[pres.zncc_rand.notna()]
            print(f"\n-- distance-matched random control (n={len(c)}) --")
            print(f"ZNCC at lattice decoy  median {c.zncc_decoy.median():6.3f}")
            print(f"ZNCC at random, same r median {c.zncc_rand.median():6.3f}")
            print(f"LATTICE EXCESS         median {c.lattice_excess.median():6.3f}"
                  f"   positive in {(c.lattice_excess > 0).mean():.1%} of pairs")
            print(f"random displacement beats truth: "
                  f"{(c.margin_rand <= 0).mean():6.1%}   "
                  f"(vs {(c.margin <= 0).mean():.1%} for the lattice)")

        if pres.severity.nunique() > 1:
            print("\n-- by severity quartile --")
            q = pd.qcut(pres.severity, 4, duplicates="drop")
            print(pres.groupby(q, observed=True)
                  .agg(n=("margin", "size"), margin=("margin", "median"),
                       beaten=("margin", lambda s: (s <= 0).mean())).to_string())
        else:
            print(f"\n-- severity is constant at {pres.severity.iloc[0]:g} "
                  f"in this split; no ladder to slice --")

        print("\n-- by architecture --")
        print(pres.groupby("architecture")
              .agg(n=("margin", "size"), margin=("margin", "median"),
                   beaten=("margin", lambda s: (s <= 0).mean())).to_string())

    if "lat_a" in d.columns:
        dec = d[d.lat_a.notna()]
        gross = dec[dec.err_px > 5.0]
        print(f"\n=== ERROR ON THE LATTICE (decoded n={len(dec)}) ===")
        print(f"localisation err median {dec.err_px.median():.2f} px   "
              f"gross (>5 px) {len(gross)} = {len(gross) / len(dec):.1%}")
        if len(gross):
            # The falsifiable claim: a gross failure is "one full repeat over",
            # i.e. its error vector is an integer combination of v1 and v2.
            r = 2.0
            on_lat = gross.lat_resid < r
            k, n_g = int(on_lat.sum()), len(gross)
            print(f"gross failures whose error is a lattice point "
                  f"(residual < {r:g} px): {k}/{n_g} = {on_lat.mean():.1%}")
            print(f"  |a| median {gross.lat_a.abs().median():.2f}   "
                  f"|b| median {gross.lat_b.abs().median():.2f}   "
                  f"residual median {gross.lat_resid.median():.2f} px")

            # CHANCE RATE, not a "non-failure" control. Errors that are not
            # gross sit near the origin, and the origin IS a lattice node, so
            # comparing against them is degenerate -- it reports ~96% and means
            # nothing. The honest null is a vector landing near a node BY
            # ACCIDENT: for a point uniform in the fundamental cell, that
            # probability is pi*r^2 / |v1 x v2|.
            p = float(np.clip(np.pi * r * r / gross.cell_area, 0, 1).mean())
            print(f"  chance rate if the error were unrelated to the lattice: "
                  f"{p:.1%}  (pi*r^2 / cell area)")
            # One-sided binomial tail, exact, no scipy dependency.
            tail = sum(math.comb(n_g, i) * p ** i * (1 - p) ** (n_g - i)
                       for i in range(k, n_g + 1))
            print(f"  P(>= {k} of {n_g} by chance) = {tail:.4f}"
                  f"{'   <-- lattice confirmed' if tail < 0.05 else ''}")

            # Robustness: pairs barely over the 5 px threshold have a small
            # error vector and could reach a node cheaply, so the headline is
            # re-tested on unambiguous basin failures.
            for cut in (10.0, 50.0):
                sub = gross[gross.err_px > cut]
                if len(sub) < 3:
                    continue
                kk, nn = int((sub.lat_resid < r).sum()), len(sub)
                pp = float(np.clip(np.pi * r * r / sub.cell_area, 0, 1).mean())
                tt = sum(math.comb(nn, i) * pp ** i * (1 - pp) ** (nn - i)
                         for i in range(kk, nn + 1))
                print(f"  restricted to err > {cut:g} px: {kk}/{nn} = "
                      f"{kk / nn:.1%}  chance {pp:.1%}  P={tt:.4f}")

            # Is the jump specifically MAT-scale? |a| ~ 44 periods against a
            # ~321 px mat+strip period invites that reading, but it is a
            # strictly sharper claim than "some lattice node" and has to be
            # tested separately rather than inferred from the median.
            v1len = np.hypot(gross.v1x, gross.v1y)
            mat_units = gross.lat_a * v1len / gross.period_px
            mat_resid = (((mat_units + 0.5) % 1.0) - 0.5).abs() * gross.period_px
            print("  -- is it the MAT period rather than the lattice? --")
            for mr in (10.0, 20.0):
                kk = int((mat_resid < mr).sum())
                pp = float(np.clip(2 * mr / gross.period_px, 0, 1).mean())
                tt = sum(math.comb(n_g, i) * pp ** i * (1 - pp) ** (n_g - i)
                         for i in range(kk, n_g + 1))
                print(f"     within {mr:g} px of an integer mat multiple: "
                      f"{kk}/{n_g} = {kk / n_g:.1%}  chance {pp:.1%}  P={tt:.4f}")


if __name__ == "__main__":
    main()

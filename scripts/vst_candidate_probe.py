#!/usr/bin/env python3
"""Does variance stabilisation generate the correct pose candidate more often?

Issue #13's named kill criterion. `driftsense.matching._band` records the
diagnostic this script measures: of Set B's >5 px failures, **76% never had a
correct pose hypothesis generated at all**, so the coarse sweep -- not the
ranking and not the network -- is what fails on degraded frames. A change that
is supposed to work by making the coarse correlation a better-founded statistic
must move *that* number. If it does not, no downstream score change is
attributable to the mechanism, and the hypothesis is wrong.

So this probe deliberately does NOT run the network, score the rubric, or touch
`locate_phase2`. It runs `pose_candidates` alone -- the classical stage the VST
acts on -- under each mode, on the same pairs, and asks one question per pair:

    did any of the k returned hypotheses land inside the window that the
    downstream local refine can actually close?

`polish_pose` re-fits within POLISH_SCALE_BAND (3% of magnification) and
POLISH_ROT_BAND (0.8 deg) of the hypothesis it is given, so those are the
defaults here rather than the rubric's credit tiers: the question is
recoverability, not final accuracy.

Paired by construction -- same pairs, same seed, same code path, only
DRIFTSENSE_VST_COARSE differs -- so the per-pair outcomes support a McNemar
test on the discordant pairs rather than a comparison of two independent rates.

Usage:
    python scripts/vst_candidate_probe.py SPLIT [SPLIT ...] \
        --modes none anscombe gat --limit 120 --out results/vst_probe.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import infer as I                                             # noqa: E402
from driftsense.matching import (                             # noqa: E402
    POLISH_ROT_BAND, POLISH_SCALE_BAND, pose_candidates)


def probe_split(split: str, mode: str, limit: int, k: int,
                scale_tol: float, rot_tol: float, band: bool) -> list[dict]:
    """Run pose_candidates over one split under one VST mode.

    `band` mirrors the shipped decode, which passes band=False: the DoG
    pre-filter measured negative on the full 2,250 pairs (-0.45 rubric points)
    and is off (`config.SHIPPED_BAND`, and the `locate_phase2` docstring).
    That makes the shipped coarse score a half-resolution correlation on *raw*
    intensities with no noise conditioning at all, which is the gap this issue
    is aimed at.
    """
    os.environ["DRIFTSENSE_VST_COARSE"] = mode
    man = pd.read_csv(os.path.join(split, "manifest.csv"))
    man = man[man.found == 1]
    if limit:
        man = man.head(limit)

    rows = []
    for _, r in man.iterrows():
        ref = I.read_gray(os.path.join(split, r.reference_path))
        sea = I.read_gray(os.path.join(split, r.search_path))
        t0 = time.perf_counter()
        cands = pose_candidates(ref, sea, k=k, band=band)
        dt = time.perf_counter() - t0

        gm, gr = float(r.magnification), float(r.rotation_deg)
        # Relative scale error and absolute rotation error, per candidate.
        s_err = [abs(c[0] - gm) / gm for c in cands]
        r_err = [abs(c[1] - gr) for c in cands]
        hit = [s <= scale_tol and d <= rot_tol for s, d in zip(s_err, r_err)]

        rows.append(dict(
            split=os.path.basename(os.path.normpath(split)),
            id=r.id, mode=mode,
            severity=float(r.get("severity_continuous", np.nan)),
            gt_m=gm, gt_rot=gr,
            # Did ANY hypothesis land in the recoverable window?
            generated=bool(any(hit)),
            # Would committing to the top-ranked one have been enough?
            top1=bool(hit[0]) if hit else False,
            # Rank of the first usable hypothesis (-1 = none).
            first_hit=int(np.argmax(hit)) if any(hit) else -1,
            best_s_err=float(min(s_err)) if s_err else np.nan,
            best_r_err=float(min(r_err)) if r_err else np.nan,
            band=band, n_cands=len(cands), secs=dt))
    return rows


def mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired boolean outcomes.

    Returns (b_only, a_only, p). The discordant pairs are the whole evidence:
    pairs both modes get right or both get wrong say nothing about which is
    better, and pooling them into two rates throws the pairing away.
    """
    from math import comb
    b_only = int(np.sum(~a & b))     # fixed by b
    a_only = int(np.sum(a & ~b))     # broken by b
    n = b_only + a_only
    if n == 0:
        return b_only, a_only, 1.0
    k = min(b_only, a_only)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return b_only, a_only, float(min(1.0, 2.0 * tail))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("splits", nargs="+")
    ap.add_argument("--modes", nargs="+", default=["none", "anscombe", "gat"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="hypotheses to keep (shipped: 3)")
    ap.add_argument("--scale-tol", type=float, default=POLISH_SCALE_BAND,
                    help="relative magnification window polish_pose can close")
    ap.add_argument("--rot-tol", type=float, default=POLISH_ROT_BAND,
                    help="absolute rotation window (deg) polish_pose can close")
    ap.add_argument("--band", action="store_true",
                    help="band-pass the coarse probe. OFF by default to mirror "
                         "the shipped decode (config.SHIPPED_BAND is False)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import cv2
    cv2.setNumThreads(a.threads)

    rows: list[dict] = []
    for split in a.splits:
        for mode in a.modes:
            t0 = time.perf_counter()
            got = probe_split(split, mode, a.limit, a.k, a.scale_tol,
                              a.rot_tol, a.band)
            rows += got
            name = os.path.basename(os.path.normpath(split))
            print(f"  {name} / {mode}: {len(got)} pairs "
                  f"in {time.perf_counter() - t0:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        df.to_csv(a.out, index=False)
        print(f"\nwrote {a.out}")

    print(f"\ncandidate generation within {100*a.scale_tol:.0f}% scale / "
          f"{a.rot_tol:.1f} deg rotation, k={a.k}\n")
    hdr = f"{'split':<8}{'n':>5}" + "".join(f"{m:>12}" for m in a.modes)
    print(hdr)
    print("-" * len(hdr))
    for split in sorted(df.split.unique()):
        g = df[df.split == split]
        n = len(g[g["mode"] == a.modes[0]])
        line = f"{split:<8}{n:>5}"
        for m in a.modes:
            line += f"{100 * g[g['mode'] == m].generated.mean():>11.1f}%"
        print(line)
    line = f"{'ALL':<8}{len(df[df['mode'] == a.modes[0]]):>5}"
    for m in a.modes:
        line += f"{100 * df[df['mode'] == m].generated.mean():>11.1f}%"
    print(line)

    print(f"\ntop-1 sufficiency (no multi-hypothesis rescue needed)\n")
    print(hdr)
    print("-" * len(hdr))
    for split in sorted(df.split.unique()):
        g = df[df.split == split]
        line = f"{split:<8}{len(g[g['mode'] == a.modes[0]]):>5}"
        for m in a.modes:
            line += f"{100 * g[g['mode'] == m].top1.mean():>11.1f}%"
        print(line)

    # Paired tests against the first mode, which is the incumbent.
    base = a.modes[0]
    for m in a.modes[1:]:
        A = df[df["mode"] == base].sort_values(["split", "id"]).generated.to_numpy()
        B = df[df["mode"] == m].sort_values(["split", "id"]).generated.to_numpy()
        fixed, broken, p = mcnemar(A, B)
        print(f"\n{m} vs {base}: rescued {fixed}, broke {broken}, "
              f"net {fixed - broken:+d} of {len(A)} pairs (McNemar exact p={p:.4f})")

    print("\nmedian coarse-stage seconds/pair")
    for m in a.modes:
        print(f"  {m:<10}{df[df['mode'] == m].secs.median():.3f}")


if __name__ == "__main__":
    main()

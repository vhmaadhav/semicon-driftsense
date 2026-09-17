#!/usr/bin/env python3
"""The issue #19 figure: the lattice ambiguity bound.

Reads the CSV `scripts/lattice_ambiguity.py` writes.

LEFT   the control. Every present pair's best lattice-translated decoy against
       the best random displacement at the SAME radius. Points above the
       diagonal are pairs where periodicity -- not ordinary image smoothness --
       is what makes the wrong position look right. This panel is what makes
       the claim falsifiable.

RIGHT  the ZNCC separation between the truth and its best lattice neighbour.
       Mass at or below zero is the population no matcher without global
       context can be expected to resolve.

A third panel is added when the CSV carries `--decode` output: each decode
error expressed in lattice coordinates, where a "one full repeat over" failure
lands on an integer node and ordinary estimation error does not.
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GROSS_PX = 5.0


def panel_control(ax, pres):
    lo = float(min(pres.zncc_rand.min(), pres.zncc_decoy.min())) - 0.05
    hi = float(max(pres.zncc_rand.max(), pres.zncc_decoy.max())) + 0.05
    ax.plot([lo, hi], [lo, hi], color="0.55", lw=1.1, ls="--", zorder=1,
            label="equal â€” no lattice effect")
    beaten = pres.margin <= 0
    ax.scatter(pres.loc[~beaten, "zncc_rand"], pres.loc[~beaten, "zncc_decoy"],
               s=15, alpha=0.5, color="#3b7dd8", edgecolors="none", zorder=2,
               label=f"truth still wins (n={int((~beaten).sum())})")
    ax.scatter(pres.loc[beaten, "zncc_rand"], pres.loc[beaten, "zncc_decoy"],
               s=34, alpha=0.9, color="#d1495b", edgecolors="black",
               linewidths=0.4, zorder=3,
               label=f"lattice repeat beats truth (n={int(beaten.sum())})")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("best ZNCC at a RANDOM displacement, same radius")
    ax.set_ylabel("best ZNCC at a LATTICE translation")
    ax.set_title("Control: is it the lattice, or just a smooth image?",
                 fontsize=11)
    ax.legend(fontsize=8, loc="lower right", framealpha=0.93)
    exc = pres.lattice_excess.median()
    ax.text(0.03, 0.97,
            f"lattice excess median +{exc:.3f} ZNCC\n"
            f"positive in {(pres.lattice_excess > 0).mean():.1%} of pairs",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#d1a800",
                      lw=0.8))


def panel_margin(ax, pres):
    ax.hist(pres.margin, bins=44, color="#3b7dd8", alpha=0.82,
            edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="#d1495b", lw=1.8,
               label="lattice neighbour ties the truth")
    med = pres.margin.median()
    ax.axvline(med, color="black", ls="--", lw=1.3, label=f"median {med:.3f}")
    ax.set_xlabel("ZNCC(truth) $-$ ZNCC(best lattice neighbour)")
    ax.set_ylabel("pairs")
    ax.set_title("How much correlation separates the truth\n"
                 "from its nearest repeat", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.93)
    ax.text(0.97, 0.55,
            f"{(pres.margin <= 0).mean():.1%} of present pairs\nare LOST outright:\n"
            f"a lattice repeat scores\nhigher than the truth",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f8d7da", ec="#d1495b",
                      lw=0.8))


def panel_lattice_coords(ax, dec):
    """Gross failures inside the fundamental cell.

    Plotting (a, b) directly is useless: gross failures sit ~44 periods out, so
    the integer grid becomes a solid fill. What the claim is actually about is
    the FRACTIONAL part -- an error that is an exact repeat has integer (a, b),
    so it lands on the origin of the unit cell no matter how many periods away
    it is. Non-gross errors are excluded rather than drawn: their error is
    smaller than one cell, so they sit at the origin by construction and would
    reproduce the degenerate control the report already discards.
    """
    bad = dec[dec.err_px > GROSS_PX].copy()
    if not len(bad):
        ax.set_axis_off()
        return
    wrap = lambda v: ((v + 0.5) % 1.0) - 0.5
    bad["fa"], bad["fb"] = wrap(bad.lat_a), wrap(bad.lat_b)
    on = bad.lat_resid < 2.0

    for k in (-0.5, 0.5):
        ax.axhline(k, color="0.75", lw=1.0)
        ax.axvline(k, color="0.75", lw=1.0)
    ax.axhline(0, color="0.88", lw=0.8, zorder=0)
    ax.axvline(0, color="0.88", lw=0.8, zorder=0)
    ax.scatter([0], [0], s=150, marker="+", color="0.35", zorder=2,
               linewidths=1.6, label="exact repeat (integer $a,b$)")
    ax.scatter(bad.loc[on, "fa"], bad.loc[on, "fb"], s=70, color="#d1495b",
               edgecolors="black", linewidths=0.5, zorder=4, alpha=0.9,
               label=f"on the lattice, <2 px (n={int(on.sum())})")
    ax.scatter(bad.loc[~on, "fa"], bad.loc[~on, "fb"], s=54, color="#8d99ae",
               edgecolors="black", linewidths=0.4, zorder=3, alpha=0.85,
               label=f"off the lattice (n={int((~on).sum())})")

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("fractional $a$  (error mod $v_1$)")
    ax.set_ylabel("fractional $b$  (error mod $v_2$)")
    ax.set_title("Gross failures inside the fundamental cell\n"
                 "clustering at the centre = \"one full repeat over\"",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.93)
    ax.text(0.03, 0.03,
            f"{on.mean():.0%} land on a lattice node\n"
            f"against an {chance(bad):.0%} chance rate\n"
            f"median |a| = {bad.lat_a.abs().median():.0f} periods",
            transform=ax.transAxes, fontsize=9, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3cd", ec="#d1a800",
                      lw=0.8))


def chance(bad, r=2.0):
    return float(np.clip(np.pi * r * r / bad.cell_area, 0, 1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=".agents/lattice_ambiguity.csv")
    ap.add_argument("--out", default=".agents/lattice_ambiguity.png")
    a = ap.parse_args()

    d = pd.read_csv(a.csv)
    pres = d[d.margin.notna() & d.zncc_rand.notna()].copy()
    dec = d[d.lat_a.notna()].copy() if "lat_a" in d.columns else pd.DataFrame()

    n = 3 if len(dec) else 2
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.4))
    panel_control(axes[0], pres)
    panel_margin(axes[1], pres)
    if len(dec):
        panel_lattice_coords(axes[2], dec)

    fig.suptitle("Periodicity as an explicit prior â€” the ambiguity bound "
                 f"(issue #19, n={len(pres)} present pairs)",
                 fontsize=12.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=150)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()

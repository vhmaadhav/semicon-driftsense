#!/usr/bin/env python3
"""Rank every overnight candidate on the objective that actually decides:
total points + 4 * P(rejection F1 >= 0.90), the +4 bonus being binary.

Threshold is fitted per checkpoint by 4-fold CV (fit and sweep on the training
folds, score the held-out fold), because whatever ships gets its own threshold.
Gate probability uses the mandated stratified 200-pair draw (A=70/B=70/C=40,
F1 over the 180 grayscale pairs) on fully out-of-fold decisions.
"""
import os, sys, glob
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from optimize_threshold import points, prep

CANDS = [("shipped driftsense.pt", ".agents/feat_base_nb.csv"),
         ("setc_last (epoch 9)",   ".agents/feat_setc_nb.csv"),
         ("Arm A setc complete",   ".agents/feat_setcfull.csv"),
         ("Arm B jitter-weighted", ".agents/feat_jw.csv")]

def oof(csv):
    d = prep(pd.read_csv(csv))
    d = d[d["set"].isin(["A","B","C"])].sort_values("pair_id").reset_index(drop=True)
    s = np.minimum(d.score.values, np.nan_to_num(d.zncc.values, nan=0.0))
    rng = np.random.default_rng(0); fold = rng.permutation(len(d)) % 4
    f = np.zeros(len(d), bool); tot = []; thr = []
    for k in range(4):
        tr, te = fold != k, fold == k
        dtr = d[tr].reset_index(drop=True); best = (-1, 0.0)
        for t in np.unique(s[tr]):
            v = float(points(dtr, s[tr], t))
            if v > best[0]: best = (v, float(t))
        thr.append(best[1]); f[te] = s[te] >= best[1]
        tot.append(float(points(d[te].reset_index(drop=True), s[te], best[1])))
    return d, f, float(np.mean(tot)), float(np.median(thr))

def f1_of(f, gt, idx):
    pf, g = f[idx], gt[idx]
    tp = ((~pf)&(~g)).sum(); fp = ((~pf)&g).sum(); fn = (pf&(~g)).sum()
    den = 2*tp+fp+fn
    return 2*tp/den if den else 0.0

rows = []
for name, csv in CANDS:
    if not os.path.exists(csv):
        print(f"  (skip {name}: {csv} absent)"); continue
    try:
        d, f, tot, thr = oof(csv)
    except Exception as e:
        print(f"  (skip {name}: {e})"); continue
    gt = (d.gt_found.values == 1); st = d["set"].values
    I = {s: np.where(st == s)[0] for s in "ABC"}
    rng = np.random.default_rng(11); v = []
    for _ in range(20000):
        dr = np.concatenate([rng.choice(I["A"], 70, False),
                             rng.choice(I["B"], 70, False),
                             rng.choice(I["C"], 40, False)])
        v.append(f1_of(f, gt, dr))
    v = np.array(v); p = float((v >= 0.90).mean())
    err = np.hypot(d.x-d.gt_x, d.y-d.gt_y)
    locB = d[(d["set"]=="B") & gt]
    b_err = np.hypot(locB.x-locB.gt_x, locB.y-locB.gt_y)
    rows.append(dict(name=name, total=tot, thr=thr, f1=v.mean(), gate=p,
                     obj=tot + 4*p,
                     locB1=float((b_err<=1).mean()), locB5=float((b_err<=5).mean())))

if not rows:
    print("no candidates to rank"); sys.exit(0)
rows.sort(key=lambda r: -r["obj"])
print(f"\n{'candidate':>24} | {'CV total':>8} | {'thr':>5} | {'F1':>6} | {'P(gate)':>7} | {'total+4P':>8} | {'B<=1px':>7} | {'B<=5px':>7}")
print("-"*98)
for r in rows:
    print(f"{r['name']:>24} | {r['total']:8.2f} | {r['thr']:5.3f} | {r['f1']:6.4f} | {r['gate']:6.1%} | {r['obj']:8.2f} | {r['locB1']:6.1%} | {r['locB5']:6.1%}")
print("-"*98)
b, base = rows[0], next((x for x in rows if "shipped" in x["name"]), None)
print(f"\nBEST BY total+4*P(bonus): {b['name']}")
if base and b["name"] != base["name"]:
    print(f"  vs shipped: total {b['total']-base['total']:+.2f}, "
          f"P(gate) {b['gate']-base['gate']:+.1%}, objective {b['obj']-base['obj']:+.2f} pts")
    print(f"  ship with threshold {b['thr']:.3f} (fitted to THIS checkpoint, not inherited)")
    if b["total"] - base["total"] < 0.35:
        print(f"  NOTE: direct delta {b['total']-base['total']:+.2f} is under the +0.35 gate (issue #19);")
        print(f"        the bonus term is what carries it. Human call.")
elif base:
    print("  shipped weights still win -- promote nothing.")

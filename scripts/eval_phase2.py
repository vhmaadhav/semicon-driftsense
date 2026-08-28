#!/usr/bin/env python3
"""Score a split against the Phase 2 rubric.

Implements the published credit tiers directly so that a local number means
the same thing as a blind-set number: tiered localisation credit, pose credit
conditional on localisation, rejection F1 on the `found` flag, and the AUC of
the confidence column against per-pair correctness.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer as I
from driftsense.matching import choose_pose_wide, locate_phase2


def loc_credit(e):        # euclidean px -> credit
    return 1.0 if e <= 1 else 0.8 if e <= 2 else 0.6 if e <= 3 else 0.4 if e <= 5 else 0.0
def scale_credit(r):      # relative error
    return 1.0 if r <= .01 else 0.6 if r <= .02 else 0.3 if r <= .05 else 0.0
def rot_credit(d):        # degrees
    return 1.0 if d <= .25 else 0.6 if d <= .5 else 0.3 if d <= 1.0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("split")
    ap.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4, help="reference machine has 4 cores")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    d = pd.read_csv(os.path.join(a.split, "manifest.csv"))
    if a.limit:
        d = d.head(a.limit)
    model, device = I.load_model(a.weights)

    rows = []
    for _, r in d.iterrows():
        ref = I.read_gray(os.path.join(a.split, r.reference_path))
        sea = I.read_gray(os.path.join(a.split, r.search_path))
        t0 = time.perf_counter()
        res = locate_phase2(model, ref, sea, device, refine=True)
        dt = time.perf_counter() - t0
        err = (float(np.hypot(res["x"] - r.gt_x_corr, res["y"] - r.gt_y_corr))
               if r.found == 1 else np.nan)
        rows.append(dict(found=int(r.found), err=err, secs=dt,
                         score=float(res.get("score", np.nan)),
                         s_err=abs(res["scale"] - r.magnification) / r.magnification,
                         r_err=abs(res["theta"] - r.rotation_deg)))
    o = pd.DataFrame(rows)
    o.to_csv(os.path.join(a.split, "phase2_eval.csv"), index=False)
    p = o[o.found == 1]

    lc = p.err.map(loc_credit)
    ok = lc > 0
    print(f"pairs {len(o)}  present {len(p)}  absent {int((o.found==0).sum())}")
    print(f"LOCALISATION  credit {lc.mean():.3f}   "
          f"<=1px {100*(p.err<=1).mean():.0f}%  <=2px {100*(p.err<=2).mean():.0f}%  "
          f"<=3px {100*(p.err<=3).mean():.0f}%  <=5px {100*(p.err<=5).mean():.0f}%   "
          f"median {p.err.median():.2f}px")
    if ok.any():
        print(f"POSE          scale {p[ok].s_err.map(scale_credit).mean():.3f} "
              f"(med {100*p[ok].s_err.median():.2f}%)   "
              f"rotation {p[ok].r_err.map(rot_credit).mean():.3f} "
              f"(med {p[ok].r_err.median():.2f} deg)   [scored on located pairs]")

    # Rejection: sweep the score threshold, report the best achievable F1.
    x, y = o.score.fillna(-9).values, o.found.values
    best = (0.0, None, 0, 0, 0)
    for t in np.unique(x):
        tp = int(((x >= t) & (y == 1)).sum()); fp = int(((x >= t) & (y == 0)).sum())
        fn = int(y.sum() - tp)
        f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0.0
        if f1 > best[0]: best = (f1, t, tp, fp, fn)
    print(f"REJECTION     best F1 {best[0]:.3f} @ score>={best[1]:.4f} "
          f"(tp {best[2]} fp {best[3]} fn {best[4]})")

    # Calibration: does the score rank correct predictions above incorrect ones?
    correct = np.where(o.found == 1, (o.err <= 5).fillna(False), False)
    aa, bb = o.score.values[correct], o.score.values[~correct]
    if len(aa) and len(bb):
        auc = float(np.mean([[(u > v) + .5*(u == v) for v in bb] for u in aa]))
        print(f"CALIBRATION   AUC(score vs correctness) {auc:.3f}")
    print(f"RUNTIME       median {o.secs.median():.2f}s  p90 {o.secs.quantile(.9):.2f}s  "
          f"max {o.secs.max():.2f}s   [{a.threads} threads]")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Score a split against the Phase 2 rubric.

Implements the published credit tiers directly so that a local number means
the same thing as a blind-set number: tiered localisation credit, pose credit
conditional on localisation, rejection F1 on the `found` flag, and the AUC of
the confidence column against per-pair correctness.

Three properties keep this honest, all of which it previously got wrong:

* **It decodes with the shipped config**, imported from `driftsense.config`
  rather than inherited from `locate_phase2`'s function defaults.
* **It scores the shipped statistic** -- `confidence` (legacy_min), the value
  `register.py` actually thresholds -- not the raw network `score`.
* **It reports rejection F1 reject-positive at the shipped threshold.** The
  lenient reading and the swept oracle optimum are printed alongside, labelled,
  because they are strictly more flattering and are not the planning number.
"""
from __future__ import annotations
import argparse, os, sys, time
import numpy as np, pandas as pd, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import infer as I
from driftsense.matching import locate_phase2
# The shipped decode config -- read from driftsense.config, never re-stated as a
# local literal, so this evaluator cannot drift away from what register.py runs.
# Previously this script relied on locate_phase2's *function defaults*, which
# happened to equal the SHIPPED_* values; that agreement was a coincidence, not
# a guarantee, and a change to config.py would have silently left this script
# measuring the old decode.
from driftsense.config import (SHIPPED_BAND, SHIPPED_SUBPIXEL_ROWS,
                               SHIPPED_THRESHOLD, SHIPPED_VERIFICATION)


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
        res = locate_phase2(model, ref, sea, device, refine=True,
                            band=SHIPPED_BAND, verification=SHIPPED_VERIFICATION,
                            subpixel_rows=SHIPPED_SUBPIXEL_ROWS)
        dt = time.perf_counter() - t0
        err = (float(np.hypot(res["x"] - r.gt_x_corr, res["y"] - r.gt_y_corr))
               if r.found == 1 else np.nan)
        # The SHIPPED statistic, identical to register.py:523 and eval_ext.py:110:
        # `confidence` (legacy_min = min(network score, native ZNCC)), NOT the raw
        # network `score`. They are different unit systems -- SHIPPED_THRESHOLD is
        # calibrated against confidence, so thresholding the network score here
        # measured something the graded system never computes.
        rows.append(dict(found=int(r.found), err=err, secs=dt,
                         score=float(res.get("confidence", res.get("score", np.nan))),
                         net_score=float(res.get("score", np.nan)),
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

    # ---- Rejection F1 -----------------------------------------------------
    # REJECT-POSITIVE is the primary number: a correctly declined absent pair is
    # the true positive. This is the convention scripts/eval_ext.py:252-262 and
    # scripts/grade_emulation.py:148 both implement, and the only one under which
    # the organizer statement "a system that never rejects scores zero" holds --
    # under present-as-positive such a system scores ~0.90 on an 80%-present set,
    # i.e. it is rewarded for doing nothing.
    #
    # The lenient (present-positive) reading is printed alongside because the
    # organizer materials are genuinely ambiguous (see eval_ext.py:236-251) and
    # the earlier self-reported figures were measuring it. Do NOT quote them as
    # if they were the same metric.
    #
    # Reported AT THE SHIPPED THRESHOLD, not at a swept optimum: the swept value
    # is an oracle that picks the best threshold in hindsight and is an upper
    # bound the graded run never achieves. It is still printed, labelled as such.
    x, y = o.score.fillna(-9).values, o.found.values

    def f1_at(t, positive="reject"):
        pf = (x >= t).astype(int)                                # 1 = we say "found"
        if positive == "reject":
            tp = int(((pf == 0) & (y == 0)).sum())               # correct reject
            fp = int(((pf == 0) & (y == 1)).sum())               # rejected a real one
            fn = int(((pf == 1) & (y == 0)).sum())               # missed an absent
        else:
            tp = int(((pf == 1) & (y == 1)).sum())
            fp = int(((pf == 1) & (y == 0)).sum())
            fn = int(((pf == 0) & (y == 1)).sum())
        return (2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) else 0.0), tp, fp, fn

    f1, tp, fp, fn = f1_at(SHIPPED_THRESHOLD)
    f1_lenient = f1_at(SHIPPED_THRESHOLD, "present")[0]
    swept, swept_t = max(((f1_at(t)[0], float(t)) for t in np.unique(x)),
                         default=(0.0, 0.0))
    print(f"REJECTION     F1 {f1:.3f} @ shipped threshold {SHIPPED_THRESHOLD} "
          f"(correct-rej {tp}  lost-real {fp}  missed-abs {fn})   [reject-positive]")
    print(f"              lenient F1 {f1_lenient:.3f} [present-positive, not the "
          f"planning number]   swept upper bound {swept:.3f} @ {swept_t:.4f}")

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

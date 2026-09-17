#!/usr/bin/env python3
"""Rubric, breakdown and bootstrap CI for v2-format predictions (issue #85 tooling).

One prediction file: the A/B/C rubric with a stratified bootstrap CI, plus
where the points go by set, severity and architecture. Two files: the paired
difference (same resampled pairs for both), which is what a promotion decision
needs -- two independent CIs overlap long before a real difference disappears.

    python scripts/compare_phase2_v2.py --data data/phase2_v2_val/dev --pred A.csv
    python scripts/compare_phase2_v2.py --data data/phase2_v2_val/dev --pred A.csv --pred-b B.csv

The rubric is scripts/grade_emulation.py (loc 0.45 A + 0.55 B, pose scored only
where localised, reject-positive F1, AUC of correct-within-5-px vs the rest);
Set D is reported but kept out of the total, as in scripts/score_phase2_v2.py.
Resampling is within Set A, B and C separately, so every draw keeps the
split's composition.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.grade_emulation import LOC_TIERS, ROT_TIERS, SCALE_TIERS, W_A, W_B, tier  # noqa: E402
from driftsense.config import SHIPPED_THRESHOLD  # noqa: E402


def load(data: str, pred_path: str) -> pd.DataFrame:
    gt = pd.read_csv(os.path.join(data, "ground_truth.csv"))
    jury = pd.read_csv(os.path.join(data, "manifest_jury.csv"))[["pair_id", "set", "architecture", "severity"]]
    pred = pd.read_csv(pred_path, comment="#")
    df = (gt.rename(columns={"present": "gt_found", "x": "gt_x", "y": "gt_y", "theta": "gt_rot",
                             "scale": "gt_scale"})
          .merge(jury, on="pair_id").merge(pred, on="pair_id", how="left"))
    missing = df.loc[df.score.isna(), "pair_id"].tolist()
    if missing:
        raise SystemExit(f"{pred_path}: missing predictions for {len(missing)} pairs, e.g. {missing[:5]}")
    df = df.sort_values("pair_id").reset_index(drop=True)
    present = df.gt_found.to_numpy() == 1
    err = np.where(present, np.hypot(df.x - df.gt_x, df.y - df.gt_y), np.nan)
    s_err = np.where(present, np.abs(df.scale - df.gt_scale) / df.gt_scale.where(df.gt_scale != 0, 1.0), np.nan)
    r_err = np.where(present, np.abs(df.theta - df.gt_rot), np.nan)
    return df.assign(err=err, scale_err=s_err, rot_err=r_err,
                     loc_raw=[tier(e, LOC_TIERS) if p else 0.0 for e, p in zip(err, present)],
                     sc_raw=[tier(e, SCALE_TIERS) if p else 0.0 for e, p in zip(s_err, present)],
                     rc_raw=[tier(e, ROT_TIERS) if p else 0.0 for e, p in zip(r_err, present)])


def components(df: pd.DataFrame, idx: np.ndarray, t: float) -> dict:
    """Rubric on rows idx (with repetition) of a loaded A/B/C frame."""
    s = df.set.to_numpy()[idx]
    gt = df.gt_found.to_numpy()[idx] == 1
    score = df.score.to_numpy()[idx]
    said = score >= t
    loc = np.where(gt & said, df.loc_raw.to_numpy()[idx], 0.0)
    la = loc[gt & (s == "A")].mean() if (gt & (s == "A")).any() else np.nan
    lb = loc[gt & (s == "B")].mean() if (gt & (s == "B")).any() else np.nan
    ok = gt & (loc > 0)
    sc = df.sc_raw.to_numpy()[idx][ok].mean() if ok.any() else np.nan
    rc = df.rc_raw.to_numpy()[idx][ok].mean() if ok.any() else np.nan
    tn, fn, fp = int((~said & ~gt).sum()), int((~said & gt).sum()), int((said & ~gt).sum())
    f1 = 2 * tn / (2 * tn + fn + fp) if (2 * tn + fn + fp) else np.nan
    correct = gt & (df.err.to_numpy()[idx] <= 5)
    a, b = score[correct], score[~correct]
    if len(a) and len(b):
        order = np.argsort(np.concatenate([a, b]), kind="mergesort")
        ranks = np.empty(len(order))
        allv = np.concatenate([a, b])[order]
        # average ranks for ties (Mann-Whitney AUC)
        _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
        cum = np.cumsum(counts)
        avg = cum - (counts - 1) / 2.0
        ranks[order] = avg[inv]
        auc = (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))
    else:
        auc = np.nan
    loc_pts = 40 * (W_A * la + W_B * lb)
    total = loc_pts + 10 * sc + 10 * rc + 15 * f1 + 10 * auc
    return {"locA": la, "locB": lb, "loc": loc_pts, "scale": 10 * sc, "rot": 10 * rc, "f1": 15 * f1,
            "auc": 10 * auc, "total": total, "FP": fp, "FN": fn, "TN": tn}


def strata(df):
    return [np.flatnonzero(df.set.to_numpy() == s) for s in ("A", "B", "C")]


def bootstrap(dfs, t_list, draws, seed=0):
    rng = np.random.default_rng(seed)
    parts = strata(dfs[0])
    out = []
    for _ in range(draws):
        idx = np.concatenate([rng.choice(p, size=len(p), replace=True) for p in parts])
        out.append([components(d, idx, t)["total"] for d, t in zip(dfs, t_list)])
    return np.array(out)


def breakdown(df: pd.DataFrame, t: float) -> None:
    said = df.score >= t
    p = df[(df.gt_found == 1) & df.set.isin(["A", "B"])]
    rows = []
    for key in ("set", "severity", "architecture"):
        for val, g in p.groupby(key):
            s = said[g.index]
            loc_cr = np.where(s, g.loc_raw, 0.0)
            ok = s & (g.loc_raw > 0)
            rows.append({"by": key, "value": val, "n": len(g), "declined": int((~s).sum()),
                         "loc_credit": loc_cr.mean(), "le1px": (g.err <= 1).mean(), "gt5px": (g.err > 5).mean(),
                         "scale_cr": g.sc_raw[ok].mean() if ok.any() else np.nan,
                         "rot_cr": g.rc_raw[ok].mean() if ok.any() else np.nan,
                         "med_err": g.err.median(), "med_rot": g.rot_err.median(),
                         "med_scale%": 100 * g.scale_err.median()})
    c = df[df.gt_found == 0]
    for val, g in c.groupby("severity"):
        rows.append({"by": "absent sev", "value": val, "n": len(g), "declined": int((g.score < t).sum()),
                     "loc_credit": np.nan, "le1px": np.nan, "gt5px": np.nan, "scale_cr": np.nan,
                     "rot_cr": np.nan, "med_err": np.nan, "med_rot": np.nan, "med_scale%": np.nan})
    print(pd.DataFrame(rows).round(3).to_string(index=False))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-b", default=None, help="second prediction file for a paired comparison (B - A)")
    ap.add_argument("--threshold", type=float, default=SHIPPED_THRESHOLD)
    ap.add_argument("--threshold-b", type=float, default=None, help="threshold for --pred-b (default: --threshold)")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--no-breakdown", action="store_true")
    a = ap.parse_args(argv)

    pd.set_option("display.width", 220)
    tb = a.threshold if a.threshold_b is None else a.threshold_b
    frames = [load(a.data, a.pred)] + ([load(a.data, a.pred_b)] if a.pred_b else [])
    thresholds = [a.threshold] + ([tb] if a.pred_b else [])
    grays = []
    for name, df, t in zip(("A", "B"), frames, thresholds):
        g = df[df.set.isin(["A", "B", "C"])].reset_index(drop=True)
        grays.append(g)
        c = components(g, np.arange(len(g)), t)
        d = df[df.set == "D"]
        d_line = (f"  | Set D (bonus): n={len(d)} <=1px {(d.err <= 1).sum()}/{len(d)}" if len(d) else "")
        print(f"[{name}] {a.pred if name == 'A' else a.pred_b}  threshold {t:g}  "
              f"n={len(g)} present={int((g.gt_found == 1).sum())} absent={int((g.gt_found == 0).sum())}")
        print(f"    loc {c['loc']:.2f}/40 (A {c['locA']:.3f}, B {c['locB']:.3f})  scale {c['scale']:.2f}  "
              f"rot {c['rot']:.2f}  F1 {c['f1']:.2f} (FP {c['FP']}, FN {c['FN']})  AUC {c['auc']:.2f}  "
              f"TOTAL {c['total']:.2f}/85{d_line}")
        if not a.no_breakdown:
            breakdown(df, t)
    if len(grays) == 2:
        assert (grays[0].pair_id.values == grays[1].pair_id.values).all(), "different pair sets"
        bs = bootstrap(grays, thresholds, a.draws)
        delta = bs[:, 1] - bs[:, 0]
        point = (components(grays[1], np.arange(len(grays[1])), thresholds[1])["total"]
                 - components(grays[0], np.arange(len(grays[0])), thresholds[0])["total"])
        print(f"paired B - A: {point:+.2f}  95% CI [{np.nanpercentile(delta, 2.5):+.2f}, "
              f"{np.nanpercentile(delta, 97.5):+.2f}]  P(delta > 0) {(delta > 0).mean():.3f}  "
              f"P(delta >= +0.35) {(delta >= 0.35).mean():.3f}  ({a.draws} draws)")
    else:
        bs = bootstrap(grays, thresholds, a.draws)[:, 0]
        print(f"total 95% CI [{np.nanpercentile(bs, 2.5):.2f}, {np.nanpercentile(bs, 97.5):.2f}] ({a.draws} draws)")


if __name__ == "__main__":
    main()

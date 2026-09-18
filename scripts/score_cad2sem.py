#!/usr/bin/env python3
"""Score a CAD2SEM set against the Phase 3 rubric, end to end.

Decodes through ``phase3.predict_pair`` rather than calling
``cad_anchor.register`` directly, so a pair the CAD path declines is scored on
what the shipped pipeline actually falls back to -- which is the whole point
when comparing a change that alters how often the CAD path declines.

Tier tables match scripts/phase3_eval.py. Efficiency (5) and the written
analysis (10) are not measurable here, so the total is out of 85.

Usage:
    python scripts/score_cad2sem.py eval20 [eval20_harsh ...]
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import phase3  # noqa: E402

LOC_TIERS = ((1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40))
SCALE_TIERS = ((0.01, 1.00), (0.02, 0.60), (0.05, 0.30))
ROT_TIERS = ((0.25, 1.00), (0.50, 0.60), (1.00, 0.30))


def tier(v: float, tiers) -> float:
    for bound, credit in tiers:
        if v <= bound:
            return credit
    return 0.0


def auc(pos, neg) -> float:
    """Rank AUC with tie averaging -- ties matter here, because the confidence
    bands deliberately put whole classes of pair on the same value."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    r = a.argsort().argsort().astype(float) + 1
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def score(root: str) -> dict:
    gt = {r["pair_id"]: r for r in csv.DictReader(open(os.path.join(root, "ground_truth.csv")))}
    err, pres, fnd, sc_e, rt_e, scr, meth = [], [], [], [], [], [], []
    for p in csv.DictReader(open(os.path.join(root, "pairs.csv"))):
        g = gt[p["pair_id"]]
        r = phase3.predict_pair(None, None,
                                os.path.join(root, p["reference_gds_path"]),
                                os.path.join(root, p["search_path"]),
                                os.path.join(root, p["search_gds_path"]))
        present = g["present"] == "1"
        pres.append(present)
        fnd.append(bool(r["found"]))
        scr.append(float(r["score"]))
        meth.append(r.get("method", "?"))
        err.append(math.hypot(r["x"] - float(g["x"]), r["y"] - float(g["y"])) if present else np.nan)
        sc_e.append(abs(r["scale"] - float(g["scale"])) / max(float(g["scale"]), 1e-9) if present else np.nan)
        rt_e.append(abs(r["theta"] - float(g["theta"])) if present else np.nan)

    err, pres, fnd = np.array(err), np.array(pres), np.array(fnd)
    sc_e, rt_e, scr = np.array(sc_e), np.array(rt_e), np.array(scr)
    loc = float(np.mean([tier(e, LOC_TIERS) if p and f else 0.0 for e, p, f in zip(err, pres, fnd)]))
    ok = pres & fnd & (err <= 5)
    scale = float(np.mean([tier(v, SCALE_TIERS) for v in sc_e[ok]])) if ok.any() else 0.0
    rot = float(np.mean([tier(v, ROT_TIERS) for v in rt_e[ok]])) if ok.any() else 0.0
    tp, fp, fn = int((pres & fnd).sum()), int((~pres & fnd).sum()), int((pres & ~fnd).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    good = pres & (err <= 5)
    a = auc(scr[good], scr[~good])
    total = 40 * loc + 10 * scale + 10 * rot + 15 * f1 + 10 * a

    print(f"\n=== {root} ===")
    print(f"  localisation  {loc:.4f} -> {40 * loc:6.2f}/40   "
          f"(within 1 px {int((pres & fnd & (err <= 1)).sum())}/{int(pres.sum())}, "
          f"median {np.nanmedian(err[pres]):.3f} px)")
    print(f"  scale         {scale:.4f} -> {10 * scale:6.2f}/10")
    print(f"  rotation      {rot:.4f} -> {10 * rot:6.2f}/10")
    print(f"  rejection F1  {f1:.4f} -> {15 * f1:6.2f}/15   (tp={tp} fp={fp} fn={fn})")
    print(f"  calibration   {a:.4f} -> {10 * a:6.2f}/10")
    print(f"  TOTAL                    {total:6.2f}/85")
    print(f"  method mix: cad={meth.count('cad')} image-fallback={meth.count('image')}")
    return {"total": total, "loc": loc, "f1": f1, "auc": a}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        score(d)

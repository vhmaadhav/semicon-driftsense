#!/usr/bin/env python3
"""Final integrated bench: run register.py end-to-end on the 20 official
pairs (the graded entry point, as a subprocess), then score localisation,
pose, rejection F1 (both conventions) and calibration AUC against the
published ground truth.

Usage:
    venv313/bin/python .agents/final_bench_tmp.py [--tag LABEL]

Writes .agents/final_bench_<tag>.json with per-pair timings and scores.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(HERE, ".agents", "ref_material")
PY = os.path.join(HERE, "venv313", "bin", "python")

GT = {}
SET_OF = {}
with open(os.path.join(REF_DIR, "ground_truth.csv"), newline="") as f:
    for r in csv.DictReader(f):
        GT[r["pair_id"]] = dict(present=int(r["present"]), x=float(r["x"]),
                                y=float(r["y"]), theta=float(r["theta"]),
                                scale=float(r["scale"]))
with open(os.path.join(REF_DIR, "manifest_jury.csv"), newline="") as f:
    for r in csv.DictReader(f):
        SET_OF[r["pair_id"]] = r["set"]

TIERS = [(1.0, 1.00), (2.0, 0.80), (3.0, 0.60), (5.0, 0.40)]
SCALE_TIERS = [(0.01, 1.00), (0.02, 0.60), (0.05, 0.30)]
ROT_TIERS = [(0.25, 1.00), (0.5, 0.60), (1.0, 0.30)]


def tier(err, tiers):
    for t, c in tiers:
        if err <= t:
            return c
    return 0.0


def auc(pairs):
    """Mann-Whitney AUC of (score, label) pairs; label 1 = correct."""
    pos = sorted(s for s, l in pairs if l == 1)
    neg = sorted(s for s, l in pairs if l == 0)
    if not pos or not neg:
        return None
    wins = ties = 0
    for s in pos:
        import bisect
        lo = bisect.bisect_left(neg, s)
        hi = bisect.bisect_right(neg, s)
        wins += lo
        ties += hi - lo
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=time.strftime("%H%M%S"))
    ap.add_argument("--threshold", type=float, default=None,
                    help="override the found threshold (default: shipped)")
    a = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        out_csv = os.path.join(td, "preds.csv")
        err_f = os.path.join(td, "stderr.txt")
        cmd = [PY, "register.py", "--input", os.path.join(REF_DIR, "pairs.csv"),
               "--output", out_csv, "--quiet"]
        if a.threshold is not None:
            cmd += ["--threshold", str(a.threshold)]
        t0 = time.perf_counter()
        with open(err_f, "w") as ef:
            rc = subprocess.run(cmd, cwd=HERE, stdout=subprocess.PIPE,
                                stderr=ef, text=True).returncode
        wall = time.perf_counter() - t0
        if rc != 0:
            print(open(err_f).read())
            raise SystemExit(f"register.py exited {rc}")

        with open(out_csv, newline="") as f:
            preds = list(csv.DictReader(f))
        stderr = open(err_f).read()
        timings = {}
        for m in re.finditer(r"# t,([^,]+),([0-9.]+)", stderr):
            timings[m.group(1)] = float(m.group(2))
        rt = re.search(r"# runtime: median ([0-9.]+) p90 ([0-9.]+) max ([0-9.]+) n=(\d+)", stderr)

    if len(preds) != len(GT):
        raise SystemExit(f"expected {len(GT)} rows, got {len(preds)}")

    loc_a, loc_b, loc_d, pose, rows, per_pair = [], [], [], [], [], []
    f1_pairs = []
    auc_pairs = []
    for r in preds:
        pid = r["pair_id"]
        g = GT[pid]
        s = SET_OF.get(pid, "?")
        found = int(r["found"])
        score = float(r["score"])
        correct = 1 if (g["present"] == 1 and found == 1) else 0
        # Rejection F1: over the grayscale pairs = sets A+B+C (D excluded)
        if s in ("A", "B", "C"):
            f1_pairs.append((pid, g["present"], found))
        # calibration AUC: score vs per-pair correctness (present & found)
        auc_pairs.append((score, correct))
        row = {"pid": pid, "set": s, "present": g["present"], "found": found,
               "score": score, "secs": timings.get(pid)}
        if g["present"] and found:
            err = float(np.hypot(float(r["x"]) - g["x"], float(r["y"]) - g["y"]))
            sc_err = abs(float(r["scale"]) - g["scale"]) / g["scale"]
            r_err = abs(float(r["theta"]) - g["theta"])
            c = tier(err, TIERS)
            pc = tier(sc_err, SCALE_TIERS) + tier(r_err, ROT_TIERS)
            if s == "A":
                loc_a.append(c)
            elif s == "B":
                loc_b.append(c)
            elif s == "D":
                loc_d.append(c)
            if s in ("A", "B"):
                pose.append(pc)
            row.update(err=err, loc_credit=c, sc_err=sc_err, r_err=r_err,
                       pose_credit=pc)
        elif g["present"]:
            # declined present pair: forfeits everything (masking semantics)
            if s == "A":
                loc_a.append(0.0)
            elif s == "B":
                loc_b.append(0.0)
            elif s == "D":
                loc_d.append(0.0)
            if s in ("A", "B"):
                pose.append(0.0)
            row.update(err=None, loc_credit=0.0, pose_credit=0.0)
        else:
            row.update(err=None, loc_credit=None, pose_credit=None)
        rows.append(row)
        per_pair.append(row)

    la = np.mean(loc_a) if loc_a else 0.0
    lb = np.mean(loc_b) if loc_b else 0.0
    loc40 = (0.45 * la + 0.55 * lb) * 40.0
    # Per-pair pose credit = scale_credit + rot_credit, each in [0,1] (max 2.0);
    # the 20-pt component scales that mean to /2*20.
    pose20 = ((np.mean(pose) / 2.0) if pose else 0.0) * 20.0

    # Rejection F1, both conventions, over present/absent decision
    tp = sum(1 for _, p, f in f1_pairs if p == 0 and f == 0)
    fp = sum(1 for _, p, f in f1_pairs if p == 1 and f == 0)
    fn = sum(1 for _, p, f in f1_pairs if p == 0 and f == 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1_rejpos = 2 * prec * rec / max(prec + rec, 1e-9) if (tp + fp + fn) else 0.0
    tp2 = sum(1 for _, p, f in f1_pairs if p == 1 and f == 1)
    fp2 = sum(1 for _, p, f in f1_pairs if p == 0 and f == 1)
    fn2 = sum(1 for _, p, f in f1_pairs if p == 1 and f == 0)
    prec2 = tp2 / max(tp2 + fp2, 1)
    rec2 = tp2 / max(tp2 + fn2, 1)
    f1_prespos = 2 * prec2 * rec2 / max(prec2 + rec2, 1e-9) if (tp2 + fp2 + fn2) else 0.0

    # Calibration AUC, eval_ext semantics: PRESENT-ONLY correctness
    # (correct = present pair localised within 5 px; absent pairs excluded).
    # Also the submitted-output variant for reference.
    auc_present_only = [(r["score"], 1 if (r["found"] == 1 and r.get("err") is not None
                                           and r["err"] <= 5.0) else 0)
                        for r in per_pair if r["present"] == 1]
    A = auc(auc_present_only)
    auc_submitted_pairs = [(r["score"], 1 if ((r["present"] == 1 and r["found"] == 1
                                               and r.get("err") is not None and r["err"] <= 5.0)
                                              or (r["present"] == 0 and r["found"] == 0)) else 0)
                           for r in per_pair]
    A_sub = auc(auc_submitted_pairs)
    secs = sorted(t for t in timings.values() if t is not None)
    summary = {
        "tag": a.tag,
        "threshold": a.threshold if a.threshold is not None else "shipped",
        "n": len(preds),
        "wall_s": round(wall, 2),
        "per_pair_median_s": round(float(np.median(secs)), 3) if secs else None,
        "per_pair_p90_s": round(float(np.percentile(secs, 90)), 3) if secs else None,
        "per_pair_max_s": round(max(secs), 3) if secs else None,
        "loc_credit_A": round(la, 4), "loc_credit_B": round(lb, 4),
        "localisation_of_40": round(loc40, 2),
        "pose_of_20": round(pose20, 2),
        "f1_reject_positive": round(f1_rejpos, 4),
        "f1_present_positive": round(f1_prespos, 4),
        "auc_present_only": round(A, 4) if A else None,
        "auc_submitted": round(A_sub, 4) if A_sub else None,
        "known_subtotal_of_85": round(loc40 + pose20 + 15 * f1_rejpos + 10 * (A or 0), 2),
    }
    print(json.dumps(summary, indent=2))
    print("\nper-pair:")
    for row in per_pair:
        print("  " + json.dumps(row))
    out_path = os.path.join(HERE, ".agents", f"final_bench_{a.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"summary": summary, "rows": per_pair}, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

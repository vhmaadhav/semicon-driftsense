#!/usr/bin/env python3
"""Score a predictions.csv against a gen_200 ground_truth.csv on the Phase 2 rubric.

Ports `scripts/eval_ext.py::score` (the shipped scorer -- imported, not
reimplemented, so the two cannot drift) and adds the three things the rubric
needs that eval_ext does not compute as returnable values: the Set D credit,
the two bonus gates, and the efficiency/timeout view of the per-pair timings.

    python judging/score_rubric.py --pred out/predictions.csv \
        --truth judging/S1/ground_truth.csv --timing out/predictions.csv.timing

Component weights and gates come from `.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md`
sections 2 and 3, which is authoritative over every other doc in the repo:

    Localisation 40 | Pose 20 (scale 10 + rot 10) | Rejection 15 |
    Calibration 10 | Efficiency 5 (relative quartile) | Generator/report 10
    BONUS +6 if Set D credit >= 0.40 and Sets A-C >= 0.50; +4 if F1(reject) >= 0.90
    Runtime: MEDIAN <= 5 s/pair; hard timeout 20 s (that pair scores zero)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from driftsense.config import SHIPPED_THRESHOLD  # noqa: E402
from eval_ext import score as eval_ext_score  # noqa: E402

# Slide 5 runtime contract.
MEDIAN_BUDGET_S = 5.0
HARD_TIMEOUT_S = 20.0
# What this campaign is actually testing for.
TARGET_MEDIAN_S = 2.0
# Slide 6 bonus gates.
BONUS_D_CREDIT = 0.40
BONUS_AC_CREDIT = 0.50
BONUS_F1 = 0.90


def load(pred_path: str, truth_path: str) -> pd.DataFrame:
    pred = pd.read_csv(pred_path)
    truth = pd.read_csv(truth_path)

    missing = set(truth.pair_id) - set(pred.pair_id)
    if missing:
        raise SystemExit(
            f"{len(missing)} pair_id(s) missing from predictions -- a missing row "
            f"scores zero under the contract: {sorted(missing)[:5]}")
    dupes = pred.pair_id[pred.pair_id.duplicated()].tolist()
    if dupes:
        raise SystemExit(f"duplicate pair_id(s) in predictions: {dupes[:5]}")

    df = truth.merge(pred, on="pair_id", suffixes=("_gt", ""))
    df = df.rename(columns={"present": "gt_found", "x_gt": "gt_x", "y_gt": "gt_y",
                            "theta_gt": "gt_rot", "scale_gt": "gt_scale"})
    df["set"] = df.pair_id.str[0]
    return df


def read_timing(path: str, n_pairs: int) -> dict | None:
    """Per-pair seconds from register.py's `# t,<pair>,<secs>` records."""
    if not path or not os.path.exists(path):
        return None
    secs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith("# t,"):
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                try:
                    secs.append(float(parts[2]))
                except ValueError:
                    pass
    if not secs:
        return None
    a = np.array(secs, dtype=float)
    return {
        "pairs_timed": int(a.size),
        "pairs_expected": int(n_pairs),
        "median_s": float(np.median(a)),
        "mean_s": float(a.mean()),
        "p90_s": float(np.percentile(a, 90)),
        "p99_s": float(np.percentile(a, 99)),
        "max_s": float(a.max()),
        "min_s": float(a.min()),
        "over_hard_timeout": int((a > HARD_TIMEOUT_S).sum()),
        "over_median_budget": bool(np.median(a) > MEDIAN_BUDGET_S),
        "meets_target_median": bool(np.median(a) <= TARGET_MEDIAN_S),
    }


def rubric(df: pd.DataFrame, threshold: float, timing: dict | None,
           label: str) -> dict:
    res, scored = eval_ext_score(df, threshold, quiet=True)

    loc_credit, loc_pts = res["localisation"]
    sc_credit, sc_pts = res["scale"]
    rot_credit, rot_pts = res["rotation"]
    f1, rej_pts = res["rejection"]
    auc, cal_pts = res["calibration"]
    auc_sub, cal_sub_pts = res["calibration_submitted"]

    gray = scored[scored["set"].isin(["A", "B", "C"])]
    present = gray[gray.gt_found == 1]
    per_set = {}
    for s in ("A", "B"):
        p = present[present["set"] == s]
        per_set[s] = {
            "pairs": int(len(p)),
            "credit": float(p.loc_credit.mean()) if len(p) else float("nan"),
            "median_err_px": float(p.err.median()) if len(p) else float("nan"),
            "within_1px": float((p.err <= 1).mean()) if len(p) else float("nan"),
            "within_5px": float((p.err <= 5).mean()) if len(p) else float("nan"),
        }

    # Set C: no localisation target -- its contribution is the correct-reject
    # rate, which is what "Set C credit" can only mean.
    c = gray[gray["set"] == "C"]
    c_correct_reject = float((c.pred_found == 0).mean()) if len(c) else float("nan")
    per_set["C"] = {"pairs": int(len(c)), "credit": c_correct_reject}

    # Set D: reads the SAME submission-masked loc_credit as A/B/C -- eval_ext
    # applies the pred_found mask to the full frame before any subset is taken,
    # so a declined present-D pair earns zero here exactly as on the CSV.
    d = scored[scored["set"] == "D"]
    dp = d[d.gt_found == 1]
    d_credit = float(dp.loc_credit.mean()) if len(dp) else float("nan")
    per_set["D"] = {
        "pairs": int(len(d)),
        "credit": d_credit,
        "within_5px": float((dp.err <= 5).mean()) if len(dp) else float("nan"),
    }

    subtotal = loc_pts + sc_pts + rot_pts + rej_pts + cal_pts

    # +6 gate, per-set mean credits (ORGANIZER_PHASE2_GROUND_TRUTH.md sec 3).
    # The harsher "every A-C per-set credit clears 0.50" reading is reported
    # alongside, because it is the one that would hurt if we guessed wrong.
    ac_natural = float(loc_credit)
    ac_strict = float(min(per_set["A"]["credit"], per_set["B"]["credit"], f1))
    bonus6_natural = bool(d_credit >= BONUS_D_CREDIT and ac_natural >= BONUS_AC_CREDIT)
    bonus6_strict = bool(d_credit >= BONUS_D_CREDIT and ac_strict >= BONUS_AC_CREDIT)
    bonus4 = bool(f1 >= BONUS_F1)
    bonus_pts = (6 if bonus6_natural else 0) + (4 if bonus4 else 0)

    return {
        "label": label,
        "threshold": float(threshold),
        "pairs": int(len(scored)),
        "components": {
            "localisation_40": {"credit": float(loc_credit), "points": float(loc_pts)},
            "pose_scale_10": {"credit": float(sc_credit), "points": float(sc_pts)},
            "pose_rotation_10": {"credit": float(rot_credit), "points": float(rot_pts)},
            "rejection_15": {"f1_reject": float(f1), "points": float(rej_pts)},
            "calibration_10": {"auc": float(auc), "points": float(cal_pts),
                               "auc_submitted": float(auc_sub),
                               "points_submitted": float(cal_sub_pts)},
        },
        "subtotal_85": float(subtotal),
        "per_set": per_set,
        "bonus": {
            "set_d_credit": d_credit,
            "sets_ac_credit_natural": ac_natural,
            "sets_ac_credit_strict": ac_strict,
            "plus6_natural_reading": bonus6_natural,
            "plus6_strict_reading": bonus6_strict,
            "plus4_f1_gate": bonus4,
            "points": int(bonus_pts),
        },
        "timing": timing,
        "not_self_assessable": {
            "efficiency_5": "relative quartile ranking across teams",
            "generator_docs_10": "judge-assessed",
        },
    }


def render(r: dict) -> str:
    W = 78
    c = r["components"]
    b = r["bonus"]
    ps = r["per_set"]
    loc = c["localisation_40"]
    sca = c["pose_scale_10"]
    rot = c["pose_rotation_10"]
    rej = c["rejection_15"]
    cal = c["calibration_10"]
    o = []
    o.append("=" * W)
    o.append("PHASE 2 RUBRIC - {}  ({} pairs, threshold {:.3f})".format(
        r["label"], r["pairs"], r["threshold"]))
    o.append("=" * W)
    o.append("{:<30}{:>34}{:>14}".format("component", "metric", "points"))
    o.append("-" * W)
    o.append("{:<30}{:>34}{:>14.2f}".format(
        "Localisation (40)", "credit {:.4f}".format(loc["credit"]), loc["points"]))
    for s in ("A", "B"):
        p = ps[s]
        o.append("{:<30}{:>34}".format(
            "   set {} ({} pairs)".format(s, p["pairs"]),
            "{:.4f}  med {:.2f}px  <=1px {:.1f}%  <=5px {:.1f}%".format(
                p["credit"], p["median_err_px"],
                100 * p["within_1px"], 100 * p["within_5px"])))
    o.append("{:<30}{:>34}{:>14.2f}".format(
        "Pose - scale (10)", "credit {:.4f}".format(sca["credit"]), sca["points"]))
    o.append("{:<30}{:>34}{:>14.2f}".format(
        "Pose - rotation (10)", "credit {:.4f}".format(rot["credit"]), rot["points"]))
    o.append("{:<30}{:>34}{:>14.2f}".format(
        "Rejection (15)", "F1(reject) {:.4f}".format(rej["f1_reject"]), rej["points"]))
    o.append("{:<30}{:>34}".format(
        "   set C correct-reject", "{:.4f}".format(ps["C"]["credit"])))
    o.append("{:<30}{:>34}{:>14.2f}".format(
        "Calibration (10)", "AUC {:.4f}".format(cal["auc"]), cal["points"]))
    o.append("{:<30}{:>34}{:>14.2f}".format(
        "   [submitted AUC]", "{:.4f}".format(cal["auc_submitted"]),
        cal["points_submitted"]))
    o.append("-" * W)
    o.append("{:<30}{:>34}{:>14.2f}".format(
        "SUBTOTAL (85 measurable)", "", r["subtotal_85"]))
    o.append("{:<30}{:>34}".format("  + efficiency (5)", "relative quartile - judge"))
    o.append("{:<30}{:>34}".format("  + generator/report (10)", "judge-assessed"))
    o.append("")
    o.append("{:<30}{:>34}{:>14d}".format("BONUS (+10 available)", "", b["points"]))
    o.append("{:<30}{:>34}".format(
        "  set D credit", "{:.4f}  (gate >= {:.2f})".format(
            b["set_d_credit"], BONUS_D_CREDIT)))
    o.append("{:<30}{:>34}".format(
        "  sets A-C credit", "{:.4f}  (gate >= {:.2f})".format(
            b["sets_ac_credit_natural"], BONUS_AC_CREDIT)))
    o.append("{:<30}{:>34}".format(
        "  +6 (D>=0.40, A-C>=0.50)",
        "MET" if b["plus6_natural_reading"] else "not met"))
    o.append("{:<30}{:>34}".format(
        "     [strict reading]",
        "MET" if b["plus6_strict_reading"] else "not met"))
    o.append("{:<30}{:>34}".format(
        "  +4 (F1(reject) >= 0.90)", "MET" if b["plus4_f1_gate"] else "not met"))
    t = r["timing"]
    if t:
        o.append("")
        o.append("{:<30}{:>34}".format(
            "RUNTIME (per pair)", "{} pairs timed".format(t["pairs_timed"])))
        o.append("{:<30}{:>34}".format(
            "  median", "{:.3f} s".format(t["median_s"])))
        o.append("{:<30}{:>34}".format("  mean", "{:.3f} s".format(t["mean_s"])))
        o.append("{:<30}{:>34}".format(
            "  p90 / p99 / max", "{:.3f} / {:.3f} / {:.3f} s".format(
                t["p90_s"], t["p99_s"], t["max_s"])))
        o.append("{:<30}{:>34}".format(
            "  median budget (<= 5 s)",
            "PASS" if not t["over_median_budget"] else "FAIL"))
        o.append("{:<30}{:>34}".format(
            "  campaign target (<= 2 s)",
            "PASS" if t["meets_target_median"] else "FAIL"))
        o.append("{:<30}{:>34d}".format(
            "  pairs over 20 s timeout", t["over_hard_timeout"]))
    o.append("=" * W)
    return "\n".join(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--timing", default="")
    ap.add_argument("--threshold", type=float, default=SHIPPED_THRESHOLD)
    ap.add_argument("--label", default="S1")
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    df = load(a.pred, a.truth)
    timing = read_timing(a.timing, len(df))
    r = rubric(df, a.threshold, timing, a.label)
    print(render(r))
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump(r, fh, indent=2, sort_keys=True)
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

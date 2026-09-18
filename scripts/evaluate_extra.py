#!/usr/bin/env python3
"""Run model inference and evaluate metrics on extra Phase 2 and Phase 3 datasets.

Datasets evaluated in C:\\Users\\nisha\\semicon-driftsense\\data_extra:
- Phase 2: p2_A_nominal, p2_B_edgepose, p2_C_edgepose
- Phase 3: p3_A_train, p3_B_blind, p3_C_dram
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import infer as I
import register as R
from driftsense import gds, pairs3
from driftsense.config import (
    SHIPPED_BAND,
    SHIPPED_CONFIDENCE,
    SHIPPED_STRIP_ROTATION,
    SHIPPED_SUBPIXEL_ROWS,
    SHIPPED_THRESHOLD,
    SHIPPED_VERIFICATION,
)
from driftsense.matching import locate_phase2
from scripts.grade_emulation import LOC_TIERS, ROT_TIERS, SCALE_TIERS, tier

DATA_EXTRA_DIR = r"C:\Users\nisha\semicon-driftsense\data_extra"
RESULTS_DIR = os.path.join(REPO_ROOT, "results", "extra_eval")


def calc_f1(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    tp = int(((y_pred == pos_label) & (y_true == pos_label)).sum())
    fp = int(((y_pred == pos_label) & (y_true != pos_label)).sum())
    fn = int(((y_pred != pos_label) & (y_true == pos_label)).sum())
    denom = 2 * tp + fp + fn
    return (2.0 * tp / denom) if denom > 0 else 0.0


def calc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Calculate ROC AUC handling ties."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Mann-Whitney U statistic
    comp = pos[:, None] - neg[None, :]
    return float(np.mean((comp > 0) + 0.5 * (comp == 0)))


def evaluate_phase2(model, device, split_name: str, split_dir: str) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Running Phase 2 Inference on: {split_name}")
    print(f"=======================================================")
    manifest_path = os.path.join(split_dir, "manifest.csv")
    df = pd.read_csv(manifest_path)

    eval_records = []
    predictions_rows = []

    for idx, row in df.iterrows():
        pid = str(row["id"])
        ref_path = os.path.join(split_dir, row["reference_path"])
        sea_path = os.path.join(split_dir, row["search_path"])

        ref = I.read_gray(ref_path)
        sea = I.read_gray(sea_path)

        t0 = time.perf_counter()
        res = locate_phase2(
            model,
            ref,
            sea,
            device,
            refine=True,
            verification=SHIPPED_VERIFICATION,
            band=SHIPPED_BAND,
            subpixel_rows=SHIPPED_SUBPIXEL_ROWS,
            strip_rot=SHIPPED_STRIP_ROTATION,
        )
        dt = time.perf_counter() - t0

        score = float(res.get("confidence", res.get("score", 0.0)))
        pred_found = int(score >= SHIPPED_THRESHOLD)
        pred_x = float(res["x"]) if pred_found else 0.0
        pred_y = float(res["y"]) if pred_found else 0.0
        pred_theta = float(res.get("theta", 0.0)) if pred_found else 0.0
        pred_scale = float(res.get("scale", 10.0)) if pred_found else 0.0

        predictions_rows.append({
            "pair_id": pid,
            "x": f"{pred_x:.4f}" if pred_found else 0,
            "y": f"{pred_y:.4f}" if pred_found else 0,
            "theta": f"{pred_theta:.4f}" if pred_found else 0,
            "scale": f"{pred_scale:.4f}" if pred_found else 0,
            "found": pred_found,
            "score": f"{score:.6f}",
        })

        gt_found = int(row["found"])
        gt_x_corr = float(row.get("gt_x_corr", row.get("gt_x", 0.0)))
        gt_y_corr = float(row.get("gt_y_corr", row.get("gt_y", 0.0)))
        gt_x = float(row.get("gt_x", 0.0))
        gt_y = float(row.get("gt_y", 0.0))
        gt_rot = float(row.get("rotation_deg", 0.0))
        gt_scale = float(row.get("magnification", 10.0))
        arch = str(row.get("architecture", ""))
        noise_prof = str(row.get("noise_profile", ""))

        if gt_found == 1:
            raw_x = float(res["x"])
            raw_y = float(res["y"])
            raw_theta = float(res.get("theta", 0.0))
            raw_scale = float(res.get("scale", 10.0))

            err_corr = float(np.hypot(raw_x - gt_x_corr, raw_y - gt_y_corr))
            err_raw = float(np.hypot(raw_x - gt_x, raw_y - gt_y))
            scale_rel_err = abs(raw_scale - gt_scale) / gt_scale
            rot_abs_err = abs(raw_theta - gt_rot)

            lc = tier(err_corr, LOC_TIERS) if pred_found else 0.0
            sc = tier(scale_rel_err, SCALE_TIERS) if (pred_found and lc > 0) else 0.0
            rc = tier(rot_abs_err, ROT_TIERS) if (pred_found and lc > 0) else 0.0
        else:
            err_corr = np.nan
            err_raw = np.nan
            scale_rel_err = np.nan
            rot_abs_err = np.nan
            lc = np.nan
            sc = np.nan
            rc = np.nan

        eval_records.append({
            "split": split_name,
            "phase": "phase2",
            "pair_id": pid,
            "gt_found": gt_found,
            "pred_found": pred_found,
            "score": score,
            "time_sec": dt,
            "raw_x": float(res["x"]),
            "raw_y": float(res["y"]),
            "raw_theta": float(res.get("theta", 0.0)),
            "raw_scale": float(res.get("scale", 10.0)),
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_theta": pred_theta,
            "pred_scale": pred_scale,
            "gt_x_corr": gt_x_corr if gt_found else np.nan,
            "gt_y_corr": gt_y_corr if gt_found else np.nan,
            "gt_x": gt_x if gt_found else np.nan,
            "gt_y": gt_y if gt_found else np.nan,
            "gt_rot": gt_rot if gt_found else np.nan,
            "gt_scale": gt_scale if gt_found else np.nan,
            "err_corr_px": err_corr,
            "err_raw_px": err_raw,
            "scale_err_rel": scale_rel_err,
            "rot_err_deg": rot_abs_err,
            "loc_credit": lc,
            "scale_credit": sc,
            "rot_credit": rc,
            "architecture": arch,
            "noise_profile": noise_prof,
        })
        print(f"[{split_name}] pair {pid:>5} | gt={gt_found} pred={pred_found} score={score:.4f} "
              f"| err={err_corr:.2f}px rot_err={rot_abs_err:.2f}° sc_err={scale_rel_err*100:.1f}% ({dt:.2f}s)")

    # Write predictions to split dir and results dir
    for target_dir in [split_dir, RESULTS_DIR]:
        os.makedirs(target_dir, exist_ok=True)
        pred_csv = os.path.join(target_dir, f"{split_name}_predictions.csv" if target_dir == RESULTS_DIR else "predictions.csv")
        with open(pred_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=R.OUT_FIELDS)
            writer.writeheader()
            writer.writerows(predictions_rows)

    return {"split": split_name, "records": eval_records}


def evaluate_phase3(model, device, split_name: str, split_dir: str) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"Running Phase 3 Inference on: {split_name}")
    print(f"=======================================================")
    pairs_path = os.path.join(split_dir, "pairs.csv")
    gt_path = os.path.join(split_dir, "ground_truth.csv")

    rows = pairs3.read_pairs(pairs_path, absolute_paths=True)
    gt_df = pd.read_csv(gt_path)
    gt_dict = {str(r["pair_id"]): r for _, r in gt_df.iterrows()}

    eval_records = []
    predictions_rows = []

    for r in rows:
        pid = str(r.pair_id)
        gt_row = gt_dict.get(pid, {})
        gt_found = int(gt_row.get("present", 1))
        gt_x = float(gt_row.get("x", 0.0))
        gt_y = float(gt_row.get("y", 0.0))
        gt_rot = float(gt_row.get("theta", 0.0))
        gt_scale = float(gt_row.get("scale", 10.0))

        t0 = time.perf_counter()
        try:
            ref = gds.render_reference(r.reference_gds_path, size=gds.REF_SIZE)
            sea = I.read_gray(r.search_path)
            res = locate_phase2(
                model,
                ref,
                sea,
                device,
                refine=True,
                verification=SHIPPED_VERIFICATION,
                band=SHIPPED_BAND,
                subpixel_rows=SHIPPED_SUBPIXEL_ROWS,
                strip_rot=SHIPPED_STRIP_ROTATION,
            )
            score = float(res.get("confidence", res.get("score", 0.0)))
        except Exception as exc:
            print(f"[ERROR] pair {pid} failed in locate_phase2: {exc}")
            res = {"x": 0.0, "y": 0.0, "theta": 0.0, "scale": 10.0, "score": 0.0}
            score = 0.0

        dt = time.perf_counter() - t0
        pred_found = int(score >= SHIPPED_THRESHOLD)
        pred_x = float(res["x"]) if pred_found else 0.0
        pred_y = float(res["y"]) if pred_found else 0.0
        pred_theta = float(res.get("theta", 0.0)) if pred_found else 0.0
        pred_scale = float(res.get("scale", 10.0)) if pred_found else 0.0

        predictions_rows.append({
            "pair_id": pid,
            "x": f"{pred_x:.4f}" if pred_found else 0,
            "y": f"{pred_y:.4f}" if pred_found else 0,
            "theta": f"{pred_theta:.4f}" if pred_found else 0,
            "scale": f"{pred_scale:.4f}" if pred_found else 0,
            "found": pred_found,
            "score": f"{score:.6f}",
        })

        if gt_found == 1:
            raw_x = float(res["x"])
            raw_y = float(res["y"])
            raw_theta = float(res.get("theta", 0.0))
            raw_scale = float(res.get("scale", 10.0))

            err_corr = float(np.hypot(raw_x - gt_x, raw_y - gt_y))
            scale_rel_err = abs(raw_scale - gt_scale) / gt_scale
            rot_abs_err = abs(raw_theta - gt_rot)

            lc = tier(err_corr, LOC_TIERS) if pred_found else 0.0
            sc = tier(scale_rel_err, SCALE_TIERS) if (pred_found and lc > 0) else 0.0
            rc = tier(rot_abs_err, ROT_TIERS) if (pred_found and lc > 0) else 0.0
        else:
            err_corr = np.nan
            scale_rel_err = np.nan
            rot_abs_err = np.nan
            lc = np.nan
            sc = np.nan
            rc = np.nan

        eval_records.append({
            "split": split_name,
            "phase": "phase3",
            "pair_id": pid,
            "gt_found": gt_found,
            "pred_found": pred_found,
            "score": score,
            "time_sec": dt,
            "raw_x": float(res["x"]),
            "raw_y": float(res["y"]),
            "raw_theta": float(res.get("theta", 0.0)),
            "raw_scale": float(res.get("scale", 10.0)),
            "pred_x": pred_x,
            "pred_y": pred_y,
            "pred_theta": pred_theta,
            "pred_scale": pred_scale,
            "gt_x_corr": gt_x if gt_found else np.nan,
            "gt_y_corr": gt_y if gt_found else np.nan,
            "gt_x": gt_x if gt_found else np.nan,
            "gt_y": gt_y if gt_found else np.nan,
            "gt_rot": gt_rot if gt_found else np.nan,
            "gt_scale": gt_scale if gt_found else np.nan,
            "err_corr_px": err_corr,
            "err_raw_px": err_corr,
            "scale_err_rel": scale_rel_err,
            "rot_err_deg": rot_abs_err,
            "loc_credit": lc,
            "scale_credit": sc,
            "rot_credit": rc,
            "architecture": "GDS_design",
            "noise_profile": "phase3_rendered",
        })
        print(f"[{split_name}] pair {pid:>5} | gt={gt_found} pred={pred_found} score={score:.4f} "
              f"| err={err_corr:.2f}px rot_err={rot_abs_err:.2f}° sc_err={scale_rel_err*100:.1f}% ({dt:.2f}s)")

    # Write predictions to split dir and results dir
    for target_dir in [split_dir, RESULTS_DIR]:
        os.makedirs(target_dir, exist_ok=True)
        pred_csv = os.path.join(target_dir, f"{split_name}_predictions.csv" if target_dir == RESULTS_DIR else "predictions.csv")
        with open(pred_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=R.OUT_FIELDS)
            writer.writeheader()
            writer.writerows(predictions_rows)

    return {"split": split_name, "records": eval_records}


def compute_split_metrics(df_split: pd.DataFrame) -> Dict[str, Any]:
    n_total = len(df_split)
    n_present = int((df_split["gt_found"] == 1).sum())
    n_absent = int((df_split["gt_found"] == 0).sum())

    # Rejection metrics
    y_true = df_split["gt_found"].to_numpy().astype(int)
    y_pred = df_split["pred_found"].to_numpy().astype(int)
    scores = df_split["score"].to_numpy().astype(float)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec_found = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec_found = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_found = calc_f1(y_true, y_pred, pos_label=1)
    f1_reject = calc_f1(y_true, y_pred, pos_label=0)

    # Localisation metrics on present pairs
    df_pres = df_split[df_split["gt_found"] == 1]
    errs = df_pres["err_corr_px"].dropna().to_numpy()

    if len(errs) > 0:
        med_err = float(np.median(errs))
        mean_err = float(np.mean(errs))
        p90_err = float(np.percentile(errs, 90))
        max_err = float(np.max(errs))
        acc_1px = float((errs <= 1.0).mean())
        acc_2px = float((errs <= 2.0).mean())
        acc_3px = float((errs <= 3.0).mean())
        acc_5px = float((errs <= 5.0).mean())
        acc_10px = float((errs <= 10.0).mean())
        loc_credit_mean = float(df_pres["loc_credit"].fillna(0.0).mean())
    else:
        med_err = mean_err = p90_err = max_err = acc_1px = acc_2px = acc_3px = acc_5px = acc_10px = loc_credit_mean = 0.0

    # Pose metrics on correctly localised pairs
    located = df_pres[df_pres["loc_credit"] > 0]
    if len(located) > 0:
        scale_err_pct = float(located["scale_err_rel"].mean() * 100)
        scale_err_med = float(located["scale_err_rel"].median() * 100)
        scale_credit_mean = float(located["scale_credit"].mean())
        rot_err_deg = float(located["rot_err_deg"].mean())
        rot_err_med = float(located["rot_err_deg"].median())
        rot_credit_mean = float(located["rot_credit"].mean())
    else:
        scale_err_pct = scale_err_med = scale_credit_mean = 0.0
        rot_err_deg = rot_err_med = rot_credit_mean = 0.0

    # Calibration AUC (correct = present & err <= 5)
    correct_labels = np.where((df_split["gt_found"] == 1) & (df_split["err_corr_px"] <= 5.0), 1, 0)
    calib_auc = calc_auc(scores, correct_labels)

    # 85-point rubric total: loc(40) + scale(10) + rot(10) + f1_reject(15) + auc(10)
    auc_val = calib_auc if np.isfinite(calib_auc) else 0.5
    rubric_total = (40.0 * loc_credit_mean +
                    10.0 * scale_credit_mean +
                    10.0 * rot_credit_mean +
                    15.0 * f1_reject +
                    10.0 * auc_val)

    return {
        "n_total": n_total,
        "n_present": n_present,
        "n_absent": n_absent,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "prec_found": prec_found,
        "rec_found": rec_found,
        "f1_found": f1_found,
        "f1_reject": f1_reject,
        "med_err_px": med_err,
        "mean_err_px": mean_err,
        "p90_err_px": p90_err,
        "max_err_px": max_err,
        "acc@1px": acc_1px,
        "acc@2px": acc_2px,
        "acc@3px": acc_3px,
        "acc@5px": acc_5px,
        "acc@10px": acc_10px,
        "loc_credit": loc_credit_mean,
        "scale_err_pct": scale_err_pct,
        "scale_err_med_pct": scale_err_med,
        "scale_credit": scale_credit_mean,
        "rot_err_deg": rot_err_deg,
        "rot_err_med_deg": rot_err_med,
        "rot_credit": rot_credit_mean,
        "calib_auc": calib_auc,
        "rubric_total": rubric_total,
        "med_time_sec": float(df_split["time_sec"].median()),
    }


def main():
    R.cap_threads(4)
    print("Loading model weights from:", I.DEFAULT_WEIGHTS)
    model, device = I.load_model(I.DEFAULT_WEIGHTS)
    if model is None:
        raise SystemExit("FATAL: model could not be loaded!")

    all_records = []

    # 1. Phase 2 Splits
    p2_splits = ["p2_A_nominal", "p2_B_edgepose", "p2_C_edgepose"]
    for s in p2_splits:
        s_dir = os.path.join(DATA_EXTRA_DIR, s)
        res = evaluate_phase2(model, device, s, s_dir)
        all_records.extend(res["records"])

    # 2. Phase 3 Splits
    p3_splits = ["p3_A_train", "p3_B_blind", "p3_C_dram"]
    for s in p3_splits:
        s_dir = os.path.join(DATA_EXTRA_DIR, s)
        res = evaluate_phase3(model, device, s, s_dir)
        all_records.extend(res["records"])

    # Combine into DataFrame
    df_all = pd.DataFrame(all_records)
    all_eval_csv = os.path.join(RESULTS_DIR, "all_extra_evaluation_details.csv")
    df_all.to_csv(all_eval_csv, index=False)
    print(f"\nWrote full evaluation details to: {all_eval_csv}")

    # Compute metrics per split and overall per phase
    summary = {}
    for s in p2_splits + p3_splits:
        df_s = df_all[df_all["split"] == s]
        summary[s] = compute_split_metrics(df_s)

    summary["phase2_pooled"] = compute_split_metrics(df_all[df_all["phase"] == "phase2"])
    summary["phase3_pooled"] = compute_split_metrics(df_all[df_all["phase"] == "phase3"])
    summary["all_pooled"] = compute_split_metrics(df_all)

    summary_json = os.path.join(RESULTS_DIR, "summary_metrics.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary metrics to: {summary_json}")

    # Print summary tables
    print("\n" + "=" * 85)
    print(f"{'Split':<16} {'N':<4} {'Pres':<5} {'F1_rej':<7} {'<=1px':<6} {'<=2px':<6} {'<=5px':<6} {'MedErr':<7} {'Scale%':<7} {'RotDeg':<7} {'Rubric':<7}")
    print("=" * 85)
    for s in p2_splits + ["phase2_pooled"] + p3_splits + ["phase3_pooled"]:
        m = summary[s]
        print(f"{s:<16} {m['n_total']:<4} {m['n_present']:<5} {m['f1_reject']:<7.3f} "
              f"{m['acc@1px']*100:<6.1f} {m['acc@2px']*100:<6.1f} {m['acc@5px']*100:<6.1f} "
              f"{m['med_err_px']:<7.2f} {m['scale_err_med_pct']:<7.2f} {m['rot_err_med_deg']:<7.2f} "
              f"{m['rubric_total']:<7.2f}")
    print("=" * 85)

    # Identify and save failure cases
    failures = []
    for idx, r in df_all.iterrows():
        reasons = []
        if r["gt_found"] == 0 and r["pred_found"] == 1:
            reasons.append("FALSE_ACCEPT (FP: absent accepted)")
        elif r["gt_found"] == 1 and r["pred_found"] == 0:
            reasons.append("FALSE_REJECT (FN: present declined)")
        elif r["gt_found"] == 1:
            if r["err_corr_px"] > 5.0:
                reasons.append(f"GROSS_LOC_FAIL ({r['err_corr_px']:.2f}px > 5px)")
            elif r["err_corr_px"] > 1.0:
                reasons.append(f"SUBPX_LOC_DEGRADE ({r['err_corr_px']:.2f}px > 1px)")

            if r["scale_err_rel"] > 0.02:
                reasons.append(f"SCALE_ERR ({r['scale_err_rel']*100:.1f}% > 2%)")
            if r["rot_err_deg"] > 0.5:
                reasons.append(f"ROT_ERR ({r['rot_err_deg']:.2f}° > 0.5°)")

        if reasons:
            failures.append({
                "split": r["split"],
                "phase": r["phase"],
                "pair_id": r["pair_id"],
                "architecture": r["architecture"],
                "noise_profile": r["noise_profile"],
                "gt_found": r["gt_found"],
                "pred_found": r["pred_found"],
                "score": r["score"],
                "err_corr_px": r["err_corr_px"],
                "scale_err_rel": r["scale_err_rel"],
                "rot_err_deg": r["rot_err_deg"],
                "reasons": "; ".join(reasons),
            })

    fail_df = pd.DataFrame(failures)
    fail_csv = os.path.join(RESULTS_DIR, "failures_analysis.csv")
    fail_df.to_csv(fail_csv, index=False)
    print(f"\nWrote {len(fail_df)} failure cases to: {fail_csv}")


if __name__ == "__main__":
    main()

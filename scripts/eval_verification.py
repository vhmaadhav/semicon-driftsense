#!/usr/bin/env python3
"""Evaluate robust verification over the existing Phase-2 K=3 hypotheses.

Inference is performed once per pair.  The output CSV contains one row per
hypothesis and can be re-analysed without re-running the Siamese network.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import infer as I  # noqa: E402
from driftsense.matching import locate_phase2  # noqa: E402


SCORES = ("zncc", "rank_score", "band_score", "dog_score")
SELECTORS = ("zncc", "rank", "band", "dog", "verify_majority",
             "verify_consensus_override")


def loc_credit(error: float) -> float:
    if not np.isfinite(error):
        return 0.0
    return 1.0 if error <= 1 else 0.8 if error <= 2 else 0.6 if error <= 3 else 0.4 if error <= 5 else 0.0


def scale_credit(relative_error: float) -> float:
    return 1.0 if relative_error <= .01 else 0.6 if relative_error <= .02 else 0.3 if relative_error <= .05 else 0.0


def rotation_credit(error_deg: float) -> float:
    return 1.0 if error_deg <= .25 else 0.6 if error_deg <= .5 else 0.3 if error_deg <= 1 else 0.0


def _value(row: pd.Series, names: tuple[str, ...], default):
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def _metadata(row: pd.Series, index: int, proxy_set_from_noise: bool = False) -> dict:
    noise = str(_value(row, ("noise_profile", "noise"), "unknown"))
    severity = _value(row, ("severity", "noise_severity"), np.nan)
    if pd.isna(severity):
        severity = {"low": 1, "medium": 2, "high": 3, "severe": 4}.get(noise, np.nan)
    dataset_set = str(_value(row, ("set", "dataset_set"), "unknown"))
    if proxy_set_from_noise and dataset_set == "unknown":
        dataset_set = "A" if noise == "default" else "B"
    return {
        "pair_id": _value(row, ("pair_id", "id"), index),
        "set": dataset_set,
        "severity": severity,
        "architecture": str(_value(row, ("architecture",), "unknown")),
        "gt_found": int(_value(row, ("gt_found", "found", "present"), 1)),
        "gt_x": float(_value(row, ("gt_x_corr", "gt_x"), np.nan)),
        "gt_y": float(_value(row, ("gt_y_corr", "gt_y"), np.nan)),
        "gt_scale": float(_value(row, ("gt_scale", "magnification", "scale"), np.nan)),
        "gt_theta": float(_value(row, ("gt_theta", "rotation_deg", "theta"), np.nan)),
    }


def _resolve(split: Path, value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else split / path


def run_inference(splits: list[Path], output: Path, weights: str, limit: int,
                  threads: int, proxy_set_from_noise: bool) -> pd.DataFrame:
    torch.set_num_threads(threads)
    model, device = I.load_model(weights)
    rows: list[dict] = []
    total_pairs = 0
    for split in splits:
        manifest = pd.read_csv(split / "manifest.csv")
        if limit:
            manifest = manifest.head(limit)
        for index, source in manifest.iterrows():
            ref = I.read_gray(str(_resolve(split, str(source.reference_path))))
            sea = I.read_gray(str(_resolve(split, str(source.search_path))))
            t0 = time.perf_counter()
            result = locate_phase2(model, ref, sea, device, refine=True, polish=False,
                                   hypotheses=3, return_hypotheses=True,
                                   verification="zncc")
            secs_total = time.perf_counter() - t0
            meta = _metadata(source, total_pairs,
                             proxy_set_from_noise=proxy_set_from_noise)
            # Pair ids are only guaranteed unique inside one manifest.
            meta["pair_id"] = f"{split.name}:{meta['pair_id']}"
            for hypothesis_index, hyp in enumerate(result["hypotheses"]):
                present = meta["gt_found"] == 1
                error = (float(np.hypot(hyp["x"] - meta["gt_x"], hyp["y"] - meta["gt_y"]))
                         if present else np.nan)
                rows.append({
                    **meta,
                    "hypothesis_index": hypothesis_index,
                    "x": hyp["x"], "y": hyp["y"],
                    "scale": hyp["scale"], "theta": hyp["theta"],
                    "location_error": error,
                    "network_score": hyp["network_score"],
                    "peak_ratio": hyp["peak_ratio"], "pose_peak": hyp["pose_peak"],
                    "zncc": hyp["zncc"], "rank_score": hyp["rank"],
                    "band_score": hyp["band"], "dog_score": hyp["dog"],
                    "is_correct_1px": bool(present and error <= 1),
                    "is_correct_2px": bool(present and error <= 2),
                    "is_correct_3px": bool(present and error <= 3),
                    "is_correct_5px": bool(present and error <= 5),
                    "secs_total": secs_total,
                    "secs_verification": result["secs_verification"],
                })
            total_pairs += 1
            print(f"{len(rows):5d} hypothesis rows from {total_pairs} pairs",
                  flush=True)
    frame = pd.DataFrame(rows)
    for score in SCORES:
        rank_name = score.replace("_score", "") + "_rank"
        frame[rank_name] = frame.groupby("pair_id")[score].rank(
            method="min", ascending=False).astype(int)
    ordered = [
        "pair_id", "set", "severity", "architecture", "gt_found",
        "gt_x", "gt_y", "gt_scale", "gt_theta", "hypothesis_index",
        "x", "y", "scale", "theta", "location_error", "network_score",
        "peak_ratio", "pose_peak", "zncc", "rank_score", "band_score",
        "dog_score", "zncc_rank", "rank_rank", "band_rank", "dog_rank",
        "is_correct_1px", "is_correct_2px", "is_correct_3px", "is_correct_5px",
        "secs_total", "secs_verification",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[ordered].to_csv(output, index=False)
    return frame[ordered]


def _winner(group: pd.DataFrame, selector: str) -> pd.Series:
    winners = {
        "zncc": int(group.zncc.idxmax()),
        "rank": int(group.rank_score.idxmax()),
        "band": int(group.band_score.idxmax()),
        "dog": int(group.dog_score.idxmax()),
    }
    if selector in winners:
        return group.loc[winners[selector]]
    if selector == "verify_consensus_override":
        chosen = winners["rank"] if winners["rank"] == winners["band"] else winners["zncc"]
        return group.loc[chosen]
    votes = [winners["zncc"], winners["rank"], winners["band"]]
    chosen = max(votes, key=votes.count)
    if votes.count(chosen) < 2:
        chosen = winners["zncc"]
    return group.loc[chosen]


def selected_rows(frame: pd.DataFrame, selector: str) -> pd.DataFrame:
    return pd.DataFrame([_winner(group, selector)
                         for _, group in frame.groupby("pair_id", sort=False)])


def _metrics(selected: pd.DataFrame) -> dict:
    present = selected[selected.gt_found == 1]
    errors = present.location_error.astype(float)
    located = errors <= 5
    scale_rel = (present.scale - present.gt_scale).abs() / present.gt_scale.abs()
    rotation_err = (present.theta - present.gt_theta).abs()
    return {
        "n_present": int(len(present)),
        "localisation_credit": float(errors.map(loc_credit).mean()) if len(present) else np.nan,
        "le_1px": float((errors <= 1).mean()) if len(present) else np.nan,
        "le_2px": float((errors <= 2).mean()) if len(present) else np.nan,
        "le_3px": float((errors <= 3).mean()) if len(present) else np.nan,
        "le_5px": float(located.mean()) if len(present) else np.nan,
        "scale_credit_located": float(scale_rel[located].map(scale_credit).mean()) if located.any() else np.nan,
        "rotation_credit_located": float(rotation_err[located].map(rotation_credit).mean()) if located.any() else np.nan,
    }


def analyse(frame: pd.DataFrame) -> dict:
    present = frame[frame.gt_found == 1]
    baseline = selected_rows(frame, "zncc")
    baseline_correct = baseline.set_index("pair_id").location_error <= 5
    summary: dict = {"selectors": {}, "oracle": {}, "robust_oracle": {},
                     "runtime": {}, "warnings": []}

    for selector in SELECTORS:
        chosen = selected_rows(frame, selector)
        correct = chosen.set_index("pair_id").location_error <= 5
        common = correct.index.intersection(baseline_correct.index)
        rescued = int((~baseline_correct.loc[common] & correct.loc[common]).sum())
        broken = int((baseline_correct.loc[common] & ~correct.loc[common]).sum())
        entry = {**_metrics(chosen), "rescued": rescued, "broken": broken,
                 "net_changed_correct": rescued - broken, "by_set": {},
                 "by_group": {}}
        for set_name, subset in chosen[chosen.gt_found == 1].groupby("set"):
            entry["by_set"][str(set_name)] = _metrics(subset)
        if {"A", "B"} <= set(entry["by_set"]):
            entry["weighted_ab_localisation_credit"] = float(
                .45 * entry["by_set"]["A"]["localisation_credit"]
                + .55 * entry["by_set"]["B"]["localisation_credit"])
        else:
            entry["weighted_ab_localisation_credit"] = np.nan
        for name, subset in chosen[chosen.gt_found == 1].groupby(["set", "severity"], dropna=False):
            entry["by_group"][f"{name[0]}|{name[1]}"] = _metrics(subset)
        summary["selectors"][selector] = entry

    oracle_pair = present.groupby("pair_id").location_error.min()
    base_present = baseline[baseline.gt_found == 1].set_index("pair_id")
    baseline_fail = base_present.location_error > 5
    oracle_correct = oracle_pair <= 5
    recoverable = baseline_fail & oracle_correct.reindex(baseline_fail.index, fill_value=False)
    summary["oracle"] = {
        "n_present": int(len(oracle_pair)),
        "le_1px": float((oracle_pair <= 1).mean()),
        "le_5px": float(oracle_correct.mean()),
        "baseline_le_5px": float((base_present.location_error <= 5).mean()),
        "recoverable_gap": float(oracle_correct.mean() - (base_present.location_error <= 5).mean()),
        "baseline_failures_with_correct_alternative": int(recoverable.sum()),
        "by_group": {},
    }
    pair_meta = present.drop_duplicates("pair_id").set_index("pair_id")
    for name, ids in pair_meta.groupby(["set", "severity"], dropna=False).groups.items():
        values = oracle_pair.reindex(ids)
        base_values = base_present.location_error.reindex(ids)
        summary["oracle"]["by_group"][f"{name[0]}|{name[1]}"] = {
            "n_present": int(len(values)),
            "oracle_le_1px": float((values <= 1).mean()),
            "oracle_le_5px": float((values <= 5).mean()),
            "baseline_le_5px": float((base_values <= 5).mean()),
        }

    recoverable_ids = recoverable[recoverable].index
    robust = {}
    union: set = set()
    for selector in ("rank", "band", "dog"):
        chosen = selected_rows(frame, selector).set_index("pair_id")
        rescued_ids = set(recoverable_ids[(chosen.reindex(recoverable_ids).location_error <= 5).fillna(False)])
        robust[f"{selector}_can_rescue"] = len(rescued_ids)
        union |= rescued_ids
    summary["robust_oracle"] = {
        "recoverable_failures": int(len(recoverable_ids)), **robust,
        "any_robust_verifier_can_rescue": int(len(union)),
    }
    per_pair = frame.drop_duplicates("pair_id")
    summary["runtime"] = {
        "median_total_s": float(per_pair.secs_total.median()),
        "median_verification_s": float(per_pair.secs_verification.median()),
        "p90_verification_s": float(per_pair.secs_verification.quantile(.9)),
    }
    sets = {str(v).upper() for v in frame["set"].dropna().unique()}
    if not {"A", "B"}.issubset(sets):
        summary["warnings"].append(
            "Manifest does not identify both Set A and Set B; weighted A+B and official Set-B claims are unavailable.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split", nargs="*", help="directories containing manifest.csv")
    parser.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--from-csv", type=Path, help="analyse an existing hypothesis CSV")
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--proxy-set-from-noise", action="store_true",
                        help="for generated research data only: default noise -> A; low..severe -> B")
    args = parser.parse_args()
    if args.from_csv:
        frame = pd.read_csv(args.from_csv)
        output = args.from_csv
    else:
        if not args.split:
            parser.error("split is required unless --from-csv is used")
        splits = [Path(value).resolve() for value in args.split]
        output = args.output or (splits[0] / "verification_hypotheses.csv"
                                 if len(splits) == 1 else ROOT / "results" / "verification_hypotheses.csv")
        frame = run_inference(splits, output, args.weights, args.limit, args.threads,
                              args.proxy_set_from_noise)
    summary_path = args.summary or output.with_name("verification_summary.json")
    summary = analyse(frame)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"wrote {output}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

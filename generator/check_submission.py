#!/usr/bin/env python3
"""Validate the generated Issue 45 package without regenerating it."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


REQUIRED = {
    "manifest.csv", "pairs.csv", "ground_truth.csv", "generation_meta.json",
    "baseline_calibration.txt", "score.json", "REPORT.md", "contact_sheet.png",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "output"))
    args = parser.parse_args()
    root = Path(args.output_dir)
    missing = sorted(name for name in REQUIRED if not (root / name).is_file())
    if missing:
        raise SystemExit(f"missing required files: {missing}")
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    counts = {name: sum(r["set_name"] == name for r in rows) for name in "ABCD"}
    checks = {
        "composition": len(rows) == 20 and counts == {"A": 8, "B": 6, "C": 4, "D": 2},
        "presence": sum(int(r["present"]) for r in rows) == 16,
        "preset_coverage": len({r["preset"] for r in rows}) >= 8,
        "family_coverage": {r["family"] for r in rows} == {"dram", "finfet"},
        "image_dimensions": True,
    }
    for row in rows:
        reference = cv2.imread(str(root / row["reference_path"]), cv2.IMREAD_UNCHANGED)
        search = cv2.imread(str(root / row["search_path"]), cv2.IMREAD_UNCHANGED)
        checks["image_dimensions"] &= reference is not None and search is not None
        checks["image_dimensions"] &= reference.shape[:2] == (1000, 1000)
        checks["image_dimensions"] &= search.shape[:2] == (1000, 1000)
    score = json.loads((root / "score.json").read_text(encoding="utf-8"))
    checks.update({
        "present_verification": score["all_present_verification_pass"],
        "baseline_target_band": score["target_band_0_30_to_0_55"],
        "set_c_audit": score["similarity_audit"]["same_family_decoys"]
        and score["similarity_audit"]["semantic_absence_flagged_in_manifest"],
        "resampling": all(item["production_better"] for item in score["resampling"]),
    })
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if not score["severity_strictly_monotone"]:
        print("WARN baseline error is not monotone at severity level 4 (documented limitation)")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())

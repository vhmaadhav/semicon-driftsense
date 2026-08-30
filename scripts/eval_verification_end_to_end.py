#!/usr/bin/env python3
"""Apply existing pose polishing once to frozen candidate-selector winners."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import infer as I  # noqa: E402
from driftsense.matching import polish_pose  # noqa: E402
from scripts.eval_verification import (  # noqa: E402
    _metrics, selected_rows,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("hypotheses", type=Path)
    parser.add_argument("splits", nargs="+", type=Path)
    parser.add_argument("--selector", default="verify_consensus_override")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.hypotheses)
    chosen = selected_rows(frame, args.selector).copy()
    manifests = {}
    for split in args.splits:
        manifest = pd.read_csv(split / "manifest.csv")
        id_col = "pair_id" if "pair_id" in manifest else "id"
        manifests[split.name] = (split, manifest.set_index(id_col, drop=False))

    polished_theta = []
    polish_seconds = []
    for _, candidate in chosen.iterrows():
        split_name, raw_id = str(candidate.pair_id).split(":", 1)
        split, manifest = manifests[split_name]
        try:
            key = int(raw_id)
        except ValueError:
            key = raw_id
        source = manifest.loc[key]
        reference = I.read_gray(str(split / str(source.reference_path)))
        search = I.read_gray(str(split / str(source.search_path)))
        start = time.perf_counter()
        _, theta, _ = polish_pose(reference, search, candidate.x, candidate.y,
                                  candidate.scale, candidate.theta)
        polish_seconds.append(time.perf_counter() - start)
        polished_theta.append(float(theta))
    chosen["theta"] = polished_theta
    chosen["secs_polish"] = polish_seconds

    summary = {
        "selector": args.selector,
        "overall": _metrics(chosen),
        "by_set": {str(name): _metrics(group)
                   for name, group in chosen.groupby("set")},
        "by_group": {f"{name[0]}|{name[1]}": _metrics(group)
                     for name, group in chosen.groupby(["set", "severity"], dropna=False)},
        "median_polish_s": float(chosen.secs_polish.median()),
        "polished_hypotheses_per_pair": 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=True))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

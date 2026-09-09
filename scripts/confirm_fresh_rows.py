#!/usr/bin/env python3
"""Run six prespecified organizer shards against a frozen row checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_shards(output, filename):
    frames = []
    for i in range(6):
        frame = pd.read_csv(output / f"shard{i}" / filename)
        frame["pair_id"] = f"shard{i}:" + frame.pair_id.astype(str)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run(a):
    freeze = json.loads((a.output / "freeze.json").read_text())
    amendment = a.output / "aggregation_fix.json"
    if amendment.exists():
        fix = json.loads(amendment.read_text())
        freeze["hashes"]["scripts/confirm_fresh_rows.py"] = fix["corrected_sha256"]
    for name, expected in freeze["hashes"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    for i, seed in enumerate(freeze["seeds"]):
        data = a.data / f"shard{i}"
        out = a.output / f"shard{i}"
        if not (out / "provenance.json").exists():
            subprocess.run(
                [
                    sys.executable,
                    str(a.generator),
                    "--set-id",
                    f"freshrow{i}",
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(data),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/eval_row_refinement.py"),
                    "--data",
                    str(data),
                    "--weights",
                    str(ROOT / "weights/driftsense.pt"),
                    "--context-refiner",
                    str(
                        ROOT
                        / freeze.get(
                            "checkpoint", "experiments/setb85/fresh_rows/candidate.pt"
                        )
                    ),
                    "--output",
                    str(out),
                ],
                check=True,
            )
    b = load_shards(a.output, "baseline.csv")
    c = load_shards(a.output, "candidate.csv")
    assert b.pair_id.equals(c.pair_id) and b.pair_id.is_unique

    def success(d):
        return (
            (np.hypot(d.x - d.gt_x, d.y - d.gt_y) < 1) & (d.score >= 0.18)
        ).to_numpy()

    bs, cs = success(b), success(c)
    rng = np.random.default_rng(2026090916)
    result = {
        "rows": len(b),
        "metric": "strict radial <1px, all present pairs including rejects",
        "sets": {},
        "B_severity": {},
    }

    def measure(mask):
        delta = cs[mask].astype(float) - bs[mask].astype(float)
        boot = np.array(
            [rng.choice(delta, len(delta), replace=True).mean() for _ in range(10000)]
        )
        return dict(
            n=int(mask.sum()),
            baseline_correct=int(bs[mask].sum()),
            candidate_correct=int(cs[mask].sum()),
            baseline=float(bs[mask].mean()),
            candidate=float(cs[mask].mean()),
            delta=float(delta.mean()),
            paired_delta95=np.quantile(boot, [0.025, 0.975]).tolist(),
        )

    for name in ["A", "B"]:
        result["sets"][name] = measure(
            ((b["set"] == name) & (b.gt_found == 1)).to_numpy()
        )
    for severity in sorted(b.loc[b["set"] == "B", "severity"].unique()):
        result["B_severity"][str(severity)] = measure(
            (
                (b["set"] == "B") & (b.gt_found == 1) & (b.severity == severity)
            ).to_numpy()
        )
    absent = b.gt_found.eq(0).to_numpy()
    result["rejection_control"] = {
        "n": int(absent.sum()),
        "baseline_rejected": int((b.score.to_numpy()[absent] < 0.18).sum()),
        "candidate_rejected": int((c.score.to_numpy()[absent] < 0.18).sum()),
    }
    assert np.array_equal(b.score, c.score)
    b.to_csv(a.output / "baseline.csv", index=False)
    c.to_csv(a.output / "candidate.csv", index=False)
    (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--generator", type=Path, required=True)
    a = p.parse_args()
    try:
        run(a)
    except Exception:
        (a.output / "FAILED.txt").write_text(traceback.format_exc())
        raise

#!/usr/bin/env python3
"""Score every paired prediction; unavailable refinements remain no-ops."""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scripts.grade_emulation import rubric, tier, LOC_TIERS


def assess(d):
    gray = d[d["set"].isin(["A", "B", "C"])]
    out = rubric(gray, 0.18)
    for s in ["A", "B", "D"]:
        g = d[(d["set"] == s) & (d.gt_found == 1)]
        e = np.hypot(g.x - g.gt_x, g.y - g.gt_y)
        out[s + "_within1"] = int(((e <= 1) & (g.score >= 0.18)).sum())
        out[s + "_n"] = len(g)
        out[s + "_credit"] = (
            float(
                np.mean(
                    [
                        tier(v, LOC_TIERS) if sc >= 0.18 else 0
                        for v, sc in zip(e, g.score)
                    ]
                )
            )
            if len(g)
            else None
        )
    accepted = gray.score >= 0.18
    present = gray.gt_found == 1
    err = np.hypot(gray.x - gray.gt_x, gray.y - gray.gt_y)
    correct = (present & accepted & (err <= 5)) | (~present & ~accepted)
    pos = gray.loc[correct, "score"].to_numpy()
    neg = gray.loc[~correct, "score"].to_numpy()
    out["submitted_correctness_auc"] = (
        float(
            ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum())
            / (len(pos) * len(neg))
        )
        if len(pos) * len(neg)
        else None
    )
    return out


def compare(b, c, draws=2000):
    if not b.pair_id.is_unique or not c.pair_id.is_unique:
        raise ValueError("duplicate pair IDs")
    if set(b.pair_id) != set(c.pair_id):
        raise ValueError("baseline/candidate pair IDs differ")
    c = c.set_index("pair_id").loc[b.pair_id].reset_index()
    for col in [
        "set",
        "gt_found",
        "gt_x",
        "gt_y",
        "gt_scale",
        "gt_rot",
        "y",
        "scale",
        "theta",
        "score",
    ]:
        pd.testing.assert_series_equal(
            b[col].reset_index(drop=True), c[col], check_names=False
        )
    out = {"baseline": assess(b), "candidate": assess(c)}
    out["delta85"] = out["candidate"]["total"] - out["baseline"]["total"]
    gray = b["set"].isin(["A", "B", "C"])
    b = b[gray].reset_index(drop=True)
    c = c[gray].reset_index(drop=True)
    # One independent source canvas per confirmation row is required here.
    if not b.source_group.is_unique:
        raise ValueError("use a clustered bootstrap for repeated source canvases")
    groups = [np.flatnonzero(b["set"].to_numpy() == s) for s in ["A", "B", "C"]]
    if any(not len(g) for g in groups):
        raise ValueError("missing grayscale stratum")
    rng = np.random.default_rng(20260909)
    delta = []
    for _ in range(draws):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        delta.append(
            rubric(c.iloc[ix], 0.18)["total"] - rubric(b.iloc[ix], 0.18)["total"]
        )
    if not np.isfinite(delta).all():
        raise ValueError("undefined bootstrap metric")
    out["paired95"] = np.quantile(delta, [0.025, 0.975]).tolist()
    out["draws"] = draws
    out["passes_gate"] = out["delta85"] >= 0.35 and out["paired95"][0] >= 0
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("directory", type=Path)
    a = ap.parse_args()
    b = pd.read_csv(a.directory / "baseline.csv")
    c = pd.read_csv(a.directory / "candidate.csv")
    result = compare(b, c)
    (a.directory / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))

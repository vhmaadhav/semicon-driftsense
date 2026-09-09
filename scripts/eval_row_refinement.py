#!/usr/bin/env python3
"""Cache frozen coarse predictions for paired row-refinement experiments."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd
import torch
import infer as I
import driftsense.matching as M
from driftsense.model import net_from_checkpoint
from driftsense.config import SHIPPED_BAND, SHIPPED_VERIFICATION


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--row-refiner", type=Path)
    ap.add_argument("--context-refiner", type=Path)
    a = ap.parse_args()
    if a.row_refiner and a.context_refiner:
        ap.error("choose one refiner")
    context_model = None
    if a.context_refiner:
        from driftsense.context_row import ContextRow

        context_model = ContextRow().eval()
        context_model.load_state_dict(
            torch.load(a.context_refiner, map_location="cpu", weights_only=True)
        )
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    checkpoint = torch.load(a.weights, map_location="cpu", weights_only=True)
    model = net_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint.get("model", checkpoint))
    model.eval()
    model = I._fuse_conv_bn(model).to(memory_format=torch.channels_last)
    if (a.data / "manifest_jury.csv").exists():
        table = pd.read_csv(a.data / "pairs.csv").merge(
            pd.read_csv(a.data / "ground_truth.csv"),
            on="pair_id",
            validate="one_to_one",
        )
        table = table.merge(
            pd.read_csv(a.data / "manifest_jury.csv")[
                ["pair_id", "set", "architecture", "severity", "seed"]
            ],
            on="pair_id",
            validate="one_to_one",
        )
        table = table.rename(
            columns={
                "present": "gt_found",
                "x": "gt_x",
                "y": "gt_y",
                "scale": "gt_scale",
                "theta": "gt_rot",
                "seed": "source_group",
            }
        )
        table["base"] = str(a.data.resolve())
    else:
        parts = []
        for file in sorted(a.data.glob("*/manifest.csv")):
            p = pd.read_csv(file).rename(
                columns={
                    "phase2_set": "set",
                    "severity_level": "severity",
                    "found": "gt_found",
                    "gt_x_corr": "target_x",
                    "gt_y_corr": "target_y",
                    "magnification": "gt_scale",
                    "rotation_deg": "gt_rot",
                }
            )
            p["gt_x"], p["gt_y"] = p.target_x, p.target_y
            p["source_group"] = str(file.parent.name) + ":" + p.canvas_id.astype(str)
            p["base"] = str(file.parent.resolve())
            parts.append(p)
        table = pd.concat(parts, ignore_index=True)
        if a.sample:
            table = (
                table.groupby(["set", "severity"], group_keys=False)
                .sample(n=a.sample, random_state=20260909)
                .sort_values("pair_id")
            )
    if not table.pair_id.is_unique:
        raise ValueError("pair IDs must be unique")
    table.to_csv(a.output / "inputs.csv", index=False)
    rows = []
    candidates = []
    fingerprints = []
    original = M.drift_row_refine
    captured = {}

    def capture(search, template, x, y, **kwargs):
        captured.update(pre_x=x, pre_y=y)
        return original(search, template, x, y, **kwargs)

    M.drift_row_refine = capture
    try:
        for i, (_, r) in enumerate(table.iterrows()):
            captured.clear()
            ref = I.read_gray(str(Path(r.base) / r.reference_path))
            sea = I.read_gray(str(Path(r.base) / r.search_path))
            t = time.perf_counter()
            o = M.locate_phase2(
                model,
                ref,
                sea,
                torch.device("cpu"),
                refine=True,
                band=SHIPPED_BAND,
                verification=SHIPPED_VERIFICATION,
                subpixel_rows=True,
            )
            row = {
                k: r[k]
                for k in (
                    "pair_id",
                    "set",
                    "severity",
                    "architecture",
                    "source_group",
                    "gt_found",
                    "gt_x",
                    "gt_y",
                    "gt_scale",
                    "gt_rot",
                )
            }
            row.update({k: o[k] for k in ("x", "y", "scale", "theta")})
            row.update(score=o["confidence"], secs=time.perf_counter() - t, **captured)
            rows.append(row)
            if a.row_refiner or a.context_refiner:
                if context_model is not None:
                    from driftsense.context_row import refine as refine_row

                    refiner_arg = context_model
                else:
                    from driftsense.row_refiner import refine as refine_row

                    refiner_arg = a.row_refiner

                candidate = row.copy()
                start = time.perf_counter()
                if row["score"] >= 0.18:
                    candidate["x"] = refine_row(
                        sea,
                        M.make_template(ref, row["scale"], row["theta"]),
                        row["x"],
                        row["y"],
                        refiner_arg,
                    )
                candidate["refiner_secs"] = time.perf_counter() - start
                candidates.append(candidate)
                fingerprints.append(
                    dict(
                        pair_id=r.pair_id,
                        reference_sha256=hashlib.sha256(
                            (Path(r.base) / r.reference_path).read_bytes()
                        ).hexdigest(),
                        search_sha256=hashlib.sha256(
                            (Path(r.base) / r.search_path).read_bytes()
                        ).hexdigest(),
                    )
                )
            if (i + 1) % 10 == 0:
                pd.DataFrame(rows).to_csv(a.output / "baseline.csv", index=False)
                if candidates:
                    pd.DataFrame(candidates).to_csv(
                        a.output / "candidate.csv", index=False
                    )
                print(f"decoded {i+1}/{len(table)}", flush=True)
    finally:
        M.drift_row_refine = original
    pd.DataFrame(rows).to_csv(a.output / "baseline.csv", index=False)
    if candidates:
        pd.DataFrame(candidates).to_csv(a.output / "candidate.csv", index=False)
        pd.DataFrame(fingerprints).to_csv(a.output / "image_hashes.csv", index=False)
    (a.output / "provenance.json").write_text(
        json.dumps(
            {
                "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
                "inputs_sha256": hashlib.sha256(
                    (a.output / "inputs.csv").read_bytes()
                ).hexdigest(),
                "rows": len(rows),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

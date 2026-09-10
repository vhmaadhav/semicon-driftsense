#!/usr/bin/env python3
"""Fresh-scene fine-feature training with train-only empirical pose errors.

No confirmation data is read. All checkpoint selection uses the historically
reused development validation split and must be confirmed independently.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import pandas as pd
import torch
from driftsense.generate import make_pairs, PoseSpec, PRESETS
from driftsense.matching import make_template
from driftsense.context_row import ContextRow, patches
from driftsense.fine_row import decode
from scripts.train_setb85 import accuracy
from scripts.grade_emulation import rubric


def split_groups(d):
    return np.array(
        [
            int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16) % 10
            for g in d.source_group
        ]
    )


def run(a):
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(20260912)
    d = pd.read_csv(a.development / "baseline.csv")
    old = np.load(a.development / "patches.npz")
    bucket = split_groups(d)
    eligible = (
        (d.gt_found == 1)
        & (d.score >= 0.18)
        & d.available
        & (np.hypot(d.x - d.gt_x, d.y - d.gt_y) < 4)
        & (abs(d.y - d.gt_y) < 1)
    ).to_numpy()
    errors = d[(bucket < 7) & eligible].copy()
    if len(errors) < 50:
        raise ValueError("insufficient train-only error examples")
    shards = a.output / "shards"
    shards.mkdir(exist_ok=True)
    # Each scene seed is independent. Multiple crops remain one source group.
    for scene in range(a.scenes):
        path = shards / f"{scene:05d}.npz"
        if path.exists():
            continue
        entropy = 120920260000 + scene * 104729
        rng = np.random.default_rng(entropy)
        low, high = (scene % 4) / 4, (scene % 4 + 1) / 4
        spec = PoseSpec(
            rotation_deg=(-5, 5),
            magnification=(8, 12),
            severity=(low, high),
            polygon_scale=(-0.05, 0.05),
        )
        generated = make_pairs(
            entropy,
            list(PRESETS),
            "randomized",
            crops=32,
            pose=spec,
            pixel_center_labels=a.pixel_center_labels,
            box_prefilter=a.box_prefilter,
        )
        pool = errors[errors.severity == scene % 4 + 1]
        if pool.empty:
            pool = errors
        ts = []
        ss = []
        targets = []
        dys = []
        for row in generated:
            e = pool.iloc[int(rng.integers(len(pool)))]
            px = row["gt_x"] + float(e.x - e.gt_x)
            py = row["gt_y"] + float(e.y - e.gt_y)
            scale = float(np.clip(row["magnification"] * e.scale / e.gt_scale, 8, 12))
            theta = float(np.clip(row["rotation_deg"] + e.theta - e.gt_rot, -5, 5))
            pair = patches(
                row["search"],
                make_template(row["reference"], scale, theta),
                px,
                py,
                y_center_offset=a.patch_y_offset,
            )
            if pair is None:
                continue
            ts.append(pair[0])
            ss.append(pair[1])
            targets.append(row["gt_x"] - round(px))
            dys.append(row["gt_y"] - py)
        if not ts:
            raise ValueError(f"scene {scene} produced no usable crops")
        np.savez(
            path,
            template=np.array(ts),
            search=np.array(ss),
            target=np.array(targets, dtype=np.float32),
            dy=np.array(dys, dtype=np.float32),
            source_seed=entropy,
        )
    chunks = []
    for i in range(a.scenes):
        with np.load(shards / f"{i:05d}.npz") as z:
            chunks.append(
                {k: z[k].copy() for k in ["template", "search", "target", "dy"]}
            )
    fresh = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    del chunks
    # Retain original training crops as an anchor; no validation rows enter fits.
    ids = np.flatnonzero((bucket < 7) & eligible)
    ft = torch.from_numpy(np.concatenate([fresh["template"], old["template"][ids]]))[
        :, None
    ]
    fs = torch.from_numpy(np.concatenate([fresh["search"], old["search"][ids]]))[
        :, None
    ]
    target = torch.from_numpy(
        np.concatenate(
            [fresh["target"], (d.gt_x - np.round(d.x)).to_numpy(dtype=np.float32)[ids]]
        )
    )
    dy = torch.from_numpy(
        np.concatenate([fresh["dy"], (d.gt_y - d.y).to_numpy(dtype=np.float32)[ids]])
    )
    et = torch.from_numpy(old["template"])[:, None]
    es = torch.from_numpy(old["search"])[:, None]
    val = (bucket >= 7) & (bucket < 9)
    base_a = accuracy(d, val, "A")
    base_b = accuracy(d, val, "B")
    if a.reliability_base:
        from driftsense.reliability_row import ReliabilityRow

        model = ReliabilityRow()
        missing, unexpected = model.load_state_dict(
            torch.load(a.reliability_base, weights_only=True), strict=False
        )
        assert not unexpected and all(
            k.startswith(("reliability.", "residual.")) for k in missing
        )
        for name, param in model.named_parameters():
            param.requires_grad_(name.startswith(("reliability.", "residual.")))
    else:
        model = ContextRow()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best = (base_b, -float("inf"))
    selected = None
    best_frame = d.copy()
    history = []
    for epoch in range(a.epochs):
        model.train()
        order = torch.randperm(len(ft))
        for ix in order.split(48):
            shift = torch.randint(-4, 5, (len(ix),))
            cols = (torch.arange(96)[None] + 8 + shift[:, None])[:, None, None].expand(
                -1, 1, 17, -1
            )
            logits = model(ft[ix], fs[ix].gather(-1, cols))
            y = (target[ix] - shift).clamp(-8, 8)
            position = y + 8
            lo = position.floor().long()
            hi = (lo + 1).clamp(max=16)
            frac = position - lo
            lp = logits.log_softmax(-1)
            ce = -(
                lp.gather(1, lo[:, None])[:, 0] * (1 - frac)
                + lp.gather(1, hi[:, None])[:, 0] * frac
            ).mean()
            radius = (
                (torch.arange(-8, 9)[None] - y[:, None]).square()
                + dy[ix, None].square()
            ).sqrt()
            success = torch.sigmoid((1 - radius) / 0.1)
            loss = ce + 0.5 * (1 - (logits.softmax(-1) * success).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 5 != 0 and epoch != a.epochs - 1:
            continue
        model.eval()
        offset = []
        with torch.inference_mode():
            for start in range(0, len(d), 48):
                offset.extend(
                    decode(
                        model(
                            et[start : start + 48], es[start : start + 48, :, :, 8:104]
                        )
                    ).tolist()
                )
        c = d.copy()
        x = np.round(d.x) + np.array(offset)
        use = d.available & (d.score >= 0.18) & (abs(x - d.x) <= 4)
        c.loc[use, "x"] = x[use]
        aa = accuracy(c, val, "A")
        bb = accuracy(c, val, "B")
        with np.errstate(divide="ignore", invalid="ignore"):
            total = rubric(c[val], 0.18)["total"]
        record = dict(epoch=epoch, A_within1=aa, B_within1=bb, total85=total)
        history.append(record)
        if aa >= base_a - 0.01 and (bb, total) > best:
            best = (bb, total)
            selected = record
            best_frame = c.copy()
            torch.save(model.state_dict(), a.output / "candidate.pt")
    result = {
        "pixel_center_labels": a.pixel_center_labels,
        "box_prefilter": a.box_prefilter,
        "reliability_residual": bool(a.reliability_base),
        "patch_y_offset": a.patch_y_offset,
        "fresh_scenes": a.scenes,
        "fresh_crops": len(fresh["target"]),
        "training_crops": len(ft),
        "selected": selected,
        "history": history,
        "splits": {},
        "limitations": "Synthetic pose/crop errors sampled from training partition; real end-to-end candidate errors may differ. No fresh confirmation consumed.",
    }
    for name, mask in [
        ("validation", val),
        ("assessment", bucket == 9),
        ("all", np.ones(len(d), bool)),
    ]:
        result["splits"][name] = {
            "baseline_B": accuracy(d, mask, "B"),
            "candidate_B": accuracy(best_frame, mask, "B"),
            "baseline_A": accuracy(d, mask, "A"),
            "candidate_A": accuracy(best_frame, mask, "A"),
        }
    best_frame.to_csv(a.output / "candidate.csv", index=False)
    (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--development", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--scenes", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pixel-center-labels", action="store_true")
    ap.add_argument("--box-prefilter", action="store_true")
    ap.add_argument("--reliability-base", type=Path)
    ap.add_argument("--patch-y-offset", type=float, default=0.0)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        run(a)
    except Exception:
        (a.output / "FAILED.txt").write_text(traceback.format_exc())
        raise

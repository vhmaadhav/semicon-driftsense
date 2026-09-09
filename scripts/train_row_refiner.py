#!/usr/bin/env python3
"""Train a bounded local refiner on source-disjoint development partitions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
import infer as I
from driftsense.matching import make_template
from driftsense.row_refiner import features, correction, GRID
from scripts.grade_emulation import rubric


def build(data, predictions, output):
    d = pd.read_csv(predictions)
    parts = []
    for f in sorted(data.glob("*/manifest.csv")):
        t = pd.read_csv(f)
        t["base"] = str(f.parent)
        t["group"] = f.parent.name + ":" + t.canvas_id.astype(str)
        parts.append(t)
    t = pd.concat(parts).set_index("pair_id")
    d = d[d.pair_id.isin(t.index)].reset_index(drop=True)
    rows = []
    feat = []
    for i, r in d.iterrows():
        v = t.loc[r.pair_id]
        ref = I.read_gray(str(Path(v.base) / v.reference_path))
        sea = I.read_gray(str(Path(v.base) / v.search_path))
        a = features(sea, make_template(ref, r.scale, r.theta), r.x, r.y)
        if a is not None:
            row = r.to_dict()
            row["source_group"] = v["group"]
            rows.append(row)
            feat.append(a)
        if (i + 1) % 250 == 0:
            print(f"features {i+1}/{len(d)}", flush=True)
    pd.DataFrame(rows).to_csv(output / "features.csv", index=False)
    np.save(output / "features.npy", np.array(feat))


def train(output, augment=False):
    torch.set_num_threads(2)
    torch.manual_seed(20260909)
    d = pd.read_csv(output / "features.csv")
    f = np.load(output / "features.npy")
    # Stable, source-group split; not an ID/order or per-rendering split.
    bucket = np.array(
        [
            int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16) % 10
            for g in d.source_group
        ]
    )
    train_idx = bucket < 7
    val_idx = (bucket >= 7) & (bucket < 9)
    test_idx = bucket == 9
    target = (d.gt_x - d.x).to_numpy(np.float32)
    observable = (
        (d.gt_found == 1)
        & (np.hypot(d.x - d.gt_x, d.y - d.gt_y) <= 4)
        & (d.score >= 0.18)
    ).to_numpy()
    target = np.where(observable, target, 0)
    x = torch.from_numpy(f)
    y = torch.from_numpy(target)
    grid = torch.from_numpy(GRID)
    best = None
    history = []
    for tier_weight in (0.0, 1.0):
        torch.manual_seed(20260909)
        model = nn.Sequential(
            nn.Linear(138 if augment else 202, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(GRID)),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        for epoch in range(301):
            model.train()
            order = torch.where(
                torch.from_numpy(train_idx & observable if augment else train_idx)
            )[0]
            order = order[torch.randperm(len(order))]
            for idx in order.split(128):
                if augment:
                    shift = torch.randint(-3, 4, (len(idx),))
                    windows = x[idx, :200].reshape(-1, 8, 25)
                    take = (torch.arange(17)[None] + 4 + shift[:, None])[
                        :, None
                    ].expand(-1, 8, -1)
                    local = windows.gather(2, take).reshape(-1, 136)
                    logits = model(torch.cat([local, x[idx, 200:]], dim=1))
                    shifted = (y[idx] - shift).clamp(-4, 4)
                else:
                    logits = model(x[idx])
                    shifted = y[idx]
                position = (shifted + 4) * 4
                lo = position.floor().long().clamp(0, 32)
                hi = (lo + 1).clamp(max=32)
                frac = position - lo
                logp = logits.log_softmax(-1)
                loss = -(
                    logp.gather(1, lo[:, None])[:, 0] * (1 - frac)
                    + logp.gather(1, hi[:, None])[:, 0] * frac
                ).mean()
                if tier_weight:
                    err = (grid[None] - shifted[:, None]).abs()
                    utility = sum(
                        w * torch.sigmoid((t - err) / 0.15)
                        for t, w in [(1, 0.2), (2, 0.2), (3, 0.2), (5, 0.4)]
                    )
                    loss = (
                        loss
                        + tier_weight
                        * (1 - (logits.softmax(-1) * utility).sum(-1)).mean()
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % 25:
                continue
            model.eval()
            weights = {}
            for n, layer in enumerate((model[0], model[2], model[4]), 1):
                weights["w" + str(n)] = layer.weight.detach().numpy().copy()
                weights["b" + str(n)] = layer.bias.detach().numpy().copy()
            dx = correction(f, weights)
            candidate = d.copy()
            candidate.x += np.where(d.score >= 0.18, dx, 0)
            mask = val_idx & d["set"].isin(["A", "B", "C"]).to_numpy()
            gain = (
                rubric(candidate[mask], 0.18)["total"] - rubric(d[mask], 0.18)["total"]
            )
            history.append(
                dict(tier_weight=tier_weight, epoch=epoch, validation_delta=gain)
            )
            if best is None or gain > best[0]:
                best = (gain, weights, tier_weight, epoch)
    np.savez(output / "row_refiner.npz", **best[1])
    dx = correction(f, best[1])
    candidate = d.copy()
    candidate.x += np.where(d.score >= 0.18, dx, 0)
    results = {
        "translation_augmentation": augment,
        "selected_tier_weight": best[2],
        "selected_epoch": best[3],
        "history": history,
        "splits": {},
    }
    for name, mask in [
        ("train", train_idx),
        ("validation", val_idx),
        ("assessment", test_idx),
        ("all", np.ones(len(d), bool)),
    ]:
        mask = mask & d["set"].isin(["A", "B", "C"]).to_numpy()
        b = rubric(d[mask], 0.18)
        c = rubric(candidate[mask], 0.18)
        results["splits"][name] = dict(
            baseline=b, candidate=c, delta=c["total"] - b["total"]
        )
    candidate.to_csv(output / "candidate.csv", index=False)
    (output / "training.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({k: v["delta"] for k, v in results["splits"].items()}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--translation-augmentation", action="store_true")
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if not (a.output / "features.npy").exists():
        build(a.data, a.predictions, a.output)
    train(a.output, a.translation_augmentation)


if __name__ == "__main__":
    main()

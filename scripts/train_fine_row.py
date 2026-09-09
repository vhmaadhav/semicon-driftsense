#!/usr/bin/env python3
"""Fine-feature experiment; select on development validation only."""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
import infer as I
from driftsense.matching import make_template
from driftsense.fine_row import FineRow, patches, decode
from scripts.grade_emulation import rubric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path)
    ap.add_argument("--predictions", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(20260909)
    if not (a.output / "patches.npz").exists():
        d = pd.read_csv(a.predictions)
        parts = []
        for p in sorted(a.data.glob("*/manifest.csv")):
            q = pd.read_csv(p)
            q["base"] = str(p.parent)
            q["source_group"] = p.parent.name + ":" + q.canvas_id.astype(str)
            parts.append(q)
        inputs = pd.concat(parts).set_index("pair_id")
        d = d[d.pair_id.isin(inputs.index)].reset_index(drop=True)
        ts = []
        ss = []
        valid = []
        groups = []
        for i, r in d.iterrows():
            v = inputs.loc[r.pair_id]
            groups.append(v.source_group)
            ref = I.read_gray(str(Path(v.base) / v.reference_path))
            sea = I.read_gray(str(Path(v.base) / v.search_path))
            pair = patches(sea, make_template(ref, r.scale, r.theta), r.x, r.y)
            valid.append(pair is not None)
            if pair is None:
                pair = (np.zeros((9, 64), np.float32), np.zeros((9, 96), np.float32))
            ts.append(pair[0])
            ss.append(pair[1])
            if (i + 1) % 250 == 0:
                print(f"patches {i+1}/{len(d)}", flush=True)
        d["source_group"] = groups
        d["available"] = valid
        d.to_csv(a.output / "baseline.csv", index=False)
        np.savez(a.output / "patches.npz", template=np.array(ts), search=np.array(ss))
    d = pd.read_csv(a.output / "baseline.csv")
    z = np.load(a.output / "patches.npz")
    t = torch.from_numpy(z["template"])[:, None]
    s = torch.from_numpy(z["search"])[:, None]
    target = torch.tensor(
        (d.gt_x - np.round(d.x)).fillna(0).to_numpy(), dtype=torch.float32
    )
    bucket = np.array(
        [
            int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16) % 10
            for g in d.source_group
        ]
    )
    eligible = (
        (d.gt_found == 1)
        & (d.score >= 0.18)
        & d.available
        & (np.hypot(d.x - d.gt_x, d.y - d.gt_y) <= 4)
    ).to_numpy()
    train = np.flatnonzero((bucket < 7) & eligible)
    val = (bucket >= 7) & (bucket < 9) & d["set"].isin(["A", "B", "C"]).to_numpy()
    model = FineRow()
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    history = []
    best = (-float("inf"), None, None)
    for epoch in range(101):
        model.train()
        order = train[torch.randperm(len(train)).numpy()]
        for ids in np.array_split(order, max(1, len(order) // 64)):
            shift = torch.randint(-4, 5, (len(ids),))
            ind = (torch.arange(80)[None] + 8 + shift[:, None])[:, None, None].expand(
                -1, 1, 9, -1
            )
            sea = s[ids].gather(-1, ind)
            logits = model(t[ids], sea)
            y = (target[ids] - shift).clamp(-8, 8)
            p = y + 8
            lo = p.floor().long()
            hi = (lo + 1).clamp(max=16)
            frac = p - lo
            lp = logits.log_softmax(-1)
            loss = -(
                lp.gather(1, lo[:, None])[:, 0] * (1 - frac)
                + lp.gather(1, hi[:, None])[:, 0] * frac
            ).mean()
            err = (torch.arange(-8, 9)[None] - y[:, None]).abs()
            utility = sum(
                w * torch.sigmoid((cut - err) / 0.15)
                for cut, w in [(1, 0.2), (2, 0.2), (3, 0.2), (5, 0.4)]
            )
            loss = loss + (1 - (logits.softmax(-1) * utility).sum(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10:
            continue
        model.eval()
        offset = []
        with torch.inference_mode():
            for start in range(0, len(d), 64):
                offset.extend(
                    decode(
                        model(t[start : start + 64], s[start : start + 64, :, :, 8:88])
                    ).tolist()
                )
        c = d.copy()
        new = np.round(d.x) + np.array(offset)
        use = d.available & (d.score >= 0.18) & (np.abs(new - d.x) <= 4)
        c.loc[use, "x"] = new[use]
        gain = rubric(c[val], 0.18)["total"] - rubric(d[val], 0.18)["total"]
        history.append(dict(epoch=epoch, validation_delta=gain))
        print(history[-1], flush=True)
        if gain > best[0]:
            best = (gain, epoch, c.copy())
            torch.save(model.state_dict(), a.output / "fine_row.pt")
    results = {"selected_epoch": best[1], "history": history, "splits": {}}
    for name, mask in [
        ("train", bucket < 7),
        ("validation", (bucket >= 7) & (bucket < 9)),
        ("assessment", bucket == 9),
        ("all", np.ones(len(d), bool)),
    ]:
        mask = mask & d["set"].isin(["A", "B", "C"]).to_numpy()
        b = rubric(d[mask], 0.18)
        c = rubric(best[2][mask], 0.18)
        results["splits"][name] = dict(
            baseline=b, candidate=c, delta=c["total"] - b["total"]
        )
    best[2].to_csv(a.output / "candidate.csv", index=False)
    (a.output / "training.json").write_text(json.dumps(results, indent=2) + "\n")
    print({k: v["delta"] for k, v in results["splits"].items()}, flush=True)


if __name__ == "__main__":
    main()

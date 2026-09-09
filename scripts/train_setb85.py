#!/usr/bin/env python3
"""Select on strict B <1px accuracy, with an A regression guard. Development only."""
import argparse, hashlib, json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import infer as I
from driftsense.matching import make_template
from driftsense.context_row import ContextRow, patches
from driftsense.fine_row import decode
from scripts.grade_emulation import rubric


def accuracy(d, mask, set_name):
    take = mask & d["set"].eq(set_name).to_numpy() & d.gt_found.eq(1).to_numpy()
    if not take.any():
        raise ValueError("empty accuracy stratum")
    return float(
        ((np.hypot(d.x - d.gt_x, d.y - d.gt_y) < 1) & (d.score >= 0.18))
        .to_numpy()[take]
        .mean()
    )


def run(a):
    torch.set_num_threads(2)
    torch.manual_seed(20260910)
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
        available = []
        groups = []
        for _, r in d.iterrows():
            v = inputs.loc[r.pair_id]
            groups.append(v.source_group)
            ref = I.read_gray(str(Path(v.base) / v.reference_path))
            sea = I.read_gray(str(Path(v.base) / v.search_path))
            pair = patches(sea, make_template(ref, r.scale, r.theta), r.x, r.y)
            available.append(pair is not None)
            if pair is None:
                pair = (np.zeros((17, 80), np.float32), np.zeros((17, 112), np.float32))
            ts.append(pair[0])
            ss.append(pair[1])
        d["source_group"] = groups
        d["available"] = available
        d.to_csv(a.output / "baseline.csv", index=False)
        np.savez(a.output / "patches.npz", template=np.array(ts), search=np.array(ss))
    d = pd.read_csv(a.output / "baseline.csv")
    z = np.load(a.output / "patches.npz")
    t = torch.from_numpy(z["template"])[:, None]
    s = torch.from_numpy(z["search"])[:, None]
    dx = torch.tensor(
        (d.gt_x - np.round(d.x)).fillna(0).to_numpy(), dtype=torch.float32
    )
    dy = torch.tensor((d.gt_y - d.y).fillna(0).to_numpy(), dtype=torch.float32)
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
        & (abs(d.y - d.gt_y) < 1)
    ).to_numpy()
    train = np.flatnonzero((bucket < 7) & eligible)
    val = (bucket >= 7) & (bucket < 9)
    baseline_a = accuracy(d, val, "A")
    baseline_b = accuracy(d, val, "B")
    best_key = (baseline_b, -float("inf"))
    best_frame = d.copy()
    best_choice = None
    history = []
    # Two prespecified objectives; weights are selected only on development.
    for objective in ["distribution", "within1"]:
        torch.manual_seed(20260910)
        model = ContextRow()
        opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
        for epoch in range(a.epochs + 1):
            model.train()
            order = train[torch.randperm(len(train)).numpy()]
            for ids in np.array_split(order, max(1, len(order) // 48)):
                shift = torch.randint(-4, 5, (len(ids),))
                ix = (torch.arange(96)[None] + 8 + shift[:, None])[
                    :, None, None
                ].expand(-1, 1, 17, -1)
                logits = model(t[ids], s[ids].gather(-1, ix))
                y = (dx[ids] - shift).clamp(-8, 8)
                pos = y + 8
                lo = pos.floor().long()
                hi = (lo + 1).clamp(max=16)
                frac = pos - lo
                lp = logits.log_softmax(-1)
                ce = -(
                    lp.gather(1, lo[:, None])[:, 0] * (1 - frac)
                    + lp.gather(1, hi[:, None])[:, 0] * frac
                ).mean()
                if objective == "within1":
                    # Smooth radial success probability, evaluated with actual dy.
                    err = (
                        (torch.arange(-8, 9)[None] - y[:, None]).square()
                        + dy[ids, None].square()
                    ).sqrt()
                    success = torch.sigmoid((1.0 - err) / 0.1)
                    loss = (
                        0.2 * ce
                        - (logits.softmax(-1) * success)
                        .sum(-1)
                        .clamp_min(1e-7)
                        .log()
                        .mean()
                    )
                else:
                    loss = ce
                opt.zero_grad()
                loss.backward()
                opt.step()
            if epoch % 10:
                continue
            model.eval()
            values = []
            with torch.inference_mode():
                for start in range(0, len(d), 48):
                    values.extend(
                        decode(
                            model(
                                t[start : start + 48],
                                s[start : start + 48, :, :, 8:104],
                            )
                        ).tolist()
                    )
            c = d.copy()
            x = np.round(d.x) + np.array(values)
            use = d.available & (d.score >= 0.18) & (abs(x - d.x) <= 4)
            c.loc[use, "x"] = x[use]
            aa = accuracy(c, val, "A")
            bb = accuracy(c, val, "B")
            with np.errstate(divide="ignore", invalid="ignore"):
                total = rubric(c[val], 0.18)["total"]
            record = dict(
                objective=objective,
                epoch=epoch,
                A_within1=aa,
                B_within1=bb,
                total85=total,
            )
            history.append(record)
            print(json.dumps(record), flush=True)
            if aa >= baseline_a - 0.01 and (bb, total) > best_key:
                best_key = (bb, total)
                best_frame = c.copy()
                best_choice = record
                torch.save(model.state_dict(), a.output / "candidate.pt")
    result = {
        "target": 0.85,
        "criterion": "strict radial error <1px; denominator all present B, including rejected/unsupported",
        "baseline_validation_B": baseline_b,
        "baseline_validation_A": baseline_a,
        "selected": best_choice,
        "history": history,
        "splits": {},
    }
    for name, mask in [
        ("train", bucket < 7),
        ("validation", val),
        ("development_assessment", bucket == 9),
        ("full_development", np.ones(len(d), bool)),
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
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=100)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        run(a)
    except Exception:
        (a.output / "FAILED.txt").write_text(traceback.format_exc())
        raise

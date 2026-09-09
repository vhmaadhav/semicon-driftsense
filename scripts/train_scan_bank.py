#!/usr/bin/env python3
"""Generate image-only candidates, then fit a development-only selector."""
import argparse, hashlib, json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from torch import nn
import infer as I
from driftsense.scan_bank import candidates
from scripts.train_setb85 import accuracy
from scripts.grade_emulation import rubric


def run(a):
    torch.set_num_threads(2)
    torch.manual_seed(20260911)
    if not (a.output / "bank.npz").exists():
        d = pd.read_csv(a.predictions)
        parts = []
        for p in sorted(a.data.glob("*/manifest.csv")):
            q = pd.read_csv(p)
            q["base"] = str(p.parent)
            q["source_group"] = p.parent.name + ":" + q.canvas_id.astype(str)
            parts.append(q)
        inputs = pd.concat(parts).set_index("pair_id")
        d = d[d.pair_id.isin(inputs.index)].reset_index(drop=True)
        xs = []
        fs = []
        vs = []
        groups = []
        for _, r in d.iterrows():
            v = inputs.loc[r.pair_id]
            groups.append(v.source_group)
            ref = I.read_gray(str(Path(v.base) / v.reference_path))
            sea = I.read_gray(str(Path(v.base) / v.search_path))
            # Candidate generation never receives labels or generator parameters.
            x, f, valid = candidates(ref, sea, r.x, r.y, r.scale, r.theta)
            xs.append(x)
            fs.append(f)
            vs.append(valid)
        d["source_group"] = groups
        d.to_csv(a.output / "baseline.csv", index=False)
        np.savez(
            a.output / "bank.npz",
            x=np.array(xs),
            features=np.array(fs),
            valid=np.array(vs),
        )
    d = pd.read_csv(a.output / "baseline.csv")
    z = np.load(a.output / "bank.npz")
    x = z["x"]
    valid = z["valid"]
    f = z["features"]
    # Append shared image-based identity-candidate descriptors to every option.
    shared = np.concatenate(
        [
            f[:, 1:, 3].max(1)[:, None],
            valid[:, 1:].mean(1)[:, None],
            d.score.to_numpy()[:, None],
        ],
        1,
    )
    f = np.concatenate(
        [f, np.repeat(shared[:, None, :], f.shape[1], axis=1)], 2
    ).astype(np.float32)
    bucket = np.array(
        [
            int(hashlib.sha256(str(g).encode()).hexdigest()[:8], 16) % 10
            for g in d.source_group
        ]
    )
    val = (bucket >= 7) & (bucket < 9)
    accepted = ((d.gt_found == 1) & (d.score >= 0.18)).to_numpy()
    success = (
        np.hypot(
            x - d.gt_x.to_numpy()[:, None],
            d.y.to_numpy()[:, None] - d.gt_y.to_numpy()[:, None],
        )
        < 1
    ) & valid
    train = np.flatnonzero((bucket < 7) & accepted)
    bmask = d["set"].eq("B").to_numpy()
    oracle = float((success.any(1) & accepted)[bmask].mean())
    base_a = accuracy(d, val, "A")
    base_b = accuracy(d, val, "B")
    best = (base_b, -float("inf"))
    selected = None
    c_best = d.copy()
    history = []
    model = nn.Sequential(
        nn.Linear(f.shape[2], 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.03)
    tf = torch.from_numpy(f)
    target = torch.from_numpy(success.astype(np.float32))
    tv = torch.from_numpy(valid)
    for epoch in range(201):
        model.train()
        order = train[torch.randperm(len(train)).numpy()]
        for ids in np.array_split(order, max(1, len(order) // 64)):
            logits = model(tf[ids]).squeeze(-1)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[tv[ids]], target[ids][tv[ids]]
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % 10:
            continue
        model.eval()
        with torch.no_grad():
            score = model(tf).squeeze(-1).masked_fill(~tv, -float("inf"))
            index = score.argmax(1).numpy()
        c = d.copy()
        chosen = x[np.arange(len(d)), index]
        c.loc[d.score >= 0.18, "x"] = chosen[d.score >= 0.18]
        aa = accuracy(c, val, "A")
        bb = accuracy(c, val, "B")
        with np.errstate(divide="ignore", invalid="ignore"):
            total = rubric(c[val], 0.18)["total"]
        record = dict(epoch=epoch, A_within1=aa, B_within1=bb, total85=total)
        history.append(record)
        if aa >= base_a - 0.01 and (bb, total) > best:
            best = (bb, total)
            selected = record
            c_best = c.copy()
            torch.save(model.state_dict(), a.output / "selector.pt")
    result = {
        "oracle_all_B": oracle,
        "oracle_note": "Label-assisted choice among image-only candidates, counting rejects; not deployable accuracy",
        "selected": selected,
        "history": history,
        "splits": {},
    }
    for name, mask in [
        ("train", bucket < 7),
        ("validation", val),
        ("assessment", bucket == 9),
        ("all", np.ones(len(d), bool)),
    ]:
        result["splits"][name] = {
            "baseline_B": accuracy(d, mask, "B"),
            "candidate_B": accuracy(c_best, mask, "B"),
            "baseline_A": accuracy(d, mask, "A"),
            "candidate_A": accuracy(c_best, mask, "A"),
        }
    c_best.to_csv(a.output / "candidate.csv", index=False)
    (a.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    try:
        run(a)
    except Exception:
        (a.output / "FAILED.txt").write_text(traceback.format_exc())
        raise

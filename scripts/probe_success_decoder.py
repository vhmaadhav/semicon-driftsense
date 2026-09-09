#!/usr/bin/env python3
"""Compare fixed maximum-success decoder with mode mean on seen development only."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from driftsense.context_row import ContextRow
from driftsense.fine_row import decode
from scripts.train_fresh_rows import split_groups
from scripts.train_setb85 import accuracy


def success_decode(logits):
    # Treat class mass as uniform over each one-pixel cell. Maximize probability
    # inside a two-pixel interval. Resolve plateaus nearest the original decoder.
    p = logits.softmax(-1)
    grid = torch.linspace(-8, 8, 321, device=p.device)
    bins = torch.arange(-8, 9, device=p.device)
    overlap = (
        torch.minimum(grid[:, None] + 1, bins[None] + 0.5)
        - torch.maximum(grid[:, None] - 1, bins[None] - 0.5)
    ).clamp(0, 1)
    mass = p @ overlap.T
    best = mass.max(-1, keepdim=True).values
    distance = (grid[None] - decode(logits)[:, None]).abs()
    idx = distance.masked_fill(mass < best - 1e-6, float("inf")).argmin(-1)
    return grid[idx]


if __name__ == "__main__":
    torch.set_num_threads(2)
    root = Path("experiments/setb85")
    d = pd.read_csv(root / "seen_context/baseline.csv")
    z = np.load(root / "seen_context/patches.npz")
    model = ContextRow().eval()
    model.load_state_dict(
        torch.load(root / "box_aligned/candidate.pt", weights_only=True)
    )
    values = []
    with torch.inference_mode():
        for i in range(0, len(d), 48):
            logits = model(
                torch.from_numpy(z["template"][i : i + 48])[:, None],
                torch.from_numpy(z["search"][i : i + 48, :, 8:104])[:, None],
            )
            values.extend(success_decode(logits).tolist())
    c = d.copy()
    x = np.round(d.x) + values
    use = d.available & (d.score >= 0.18) & (abs(x - d.x) <= 4)
    c.loc[use, "x"] = x[use]
    old = pd.read_csv(root / "box_aligned/candidate.csv")
    groups = split_groups(d)
    out = {}
    for name, mask in [
        ("validation", (groups >= 7) & (groups < 9)),
        ("assessment", groups == 9),
        ("all", np.ones(len(d), bool)),
    ]:
        out[name] = {
            k: {
                "mode_mean": accuracy(old, mask, k),
                "success_decoder": accuracy(c, mask, k),
            }
            for k in ["A", "B"]
        }
    p = root / "success_decoder"
    p.mkdir(exist_ok=True)
    (p / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out))

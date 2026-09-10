"""Label-assisted row-evidence diagnostics on explicitly seen development."""

from pathlib import Path
import json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np, pandas as pd, torch
from driftsense.context_row import ContextRow

p = Path("experiments/setb85")
d = pd.read_csv(p / "seen_context/baseline.csv")
c = pd.read_csv(p / "box_aligned/candidate.csv")
z = np.load(p / "seen_context/patches.npz")
torch.set_num_threads(2)
m = ContextRow().eval()
m.load_state_dict(torch.load(p / "box_aligned/candidate.pt", weights_only=True))
curves = []
h = m.head.register_forward_pre_hook(
    lambda module, args: curves.append(args[0].detach().numpy().copy())
)
with torch.inference_mode():
    for i in range(0, len(d), 48):
        m(
            torch.from_numpy(z["template"][i : i + 48])[:, None],
            torch.from_numpy(z["search"][i : i + 48, :, 8:104])[:, None],
        )
h.remove()
corr = np.concatenate(curves)
peak = corr.argmax(-1)
off = peak.astype(float) - 8
for i in range(len(d)):
    for row in range(17):
        k = peak[i, row]
        if 0 < k < 16:
            a, b, v = corr[i, row, k - 1 : k + 2]
            den = a - 2 * b + v
            if abs(den) > 1e-8:
                off[i, row] += np.clip(0.5 * (a - v) / den, -0.5, 0.5)
err = np.hypot(
    np.round(d.x.to_numpy())[:, None] + off - d.gt_x.to_numpy()[:, None],
    (d.y - d.gt_y).to_numpy()[:, None],
)
supported = d.available.to_numpy() & (d.score.to_numpy() >= 0.18)
ok = (np.hypot(c.x - c.gt_x, c.y - c.gt_y) < 1).to_numpy() & supported
mask = (d["set"] == "B").to_numpy() & (d.gt_found == 1).to_numpy()
texture = np.diff(z["template"][:, 8], axis=-1).std(-1)
q = np.quantile(texture[mask], [0.25, 0.5, 0.75])
out = {
    "note": "Seen data; oracle row choice uses labels and is NOT deployable accuracy.",
    "B_n": int(mask.sum()),
    "model_correct": int(ok[mask].sum()),
    "centre_peak_correct": int(((err[:, 8] < 1) & supported & mask).sum()),
    "any_17row_peak_oracle_correct": int(((err.min(1) < 1) & supported & mask).sum()),
    "texture_quartiles": [],
}
for low, high in zip([-np.inf, *q], [*q, np.inf]):
    take = mask & (texture >= low) & (texture < high)
    out["texture_quartiles"].append(
        {"n": int(take.sum()), "correct": int(ok[take].sum())}
    )
p = p / "row_evidence"
p.mkdir(exist_ok=True)
(p / "results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))

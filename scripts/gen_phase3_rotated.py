"""Generate a Phase 3 CAD set WITH search rotation (the organizer generator's
CLI never passes search_rotation_deg, so every locally generated set so far is
0 deg).  Emits the six-column Phase 3 pairs.csv plus a Phase 2-shaped
ground_truth.csv that records the TRUE drawn angle, so rotation credit is
scorable for the first time."""
import argparse, csv, os, sys, time
import numpy as np, cv2, gdstk

sys.path.insert(0, os.environ.get("I4C_ROOT", "../mentor-phase3"))
from src.cad_pipeline import (CadGenerationParams, build_cad_geometry,
                              render_cad_sample)

PAIRS_FIELDS = ["pair_id", "search_path", "reference_gds_path",
                "search_gds_path", "reference_sem_path", "params_json_path"]

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--rot", type=float, default=10.0)
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=20260918)
ap.add_argument("--absent", type=float, default=0.08)
a = ap.parse_args()

root = a.out
for d in ("reference", "search"):
    os.makedirs(os.path.join(root, d), exist_ok=True)

rng = np.random.default_rng(a.seed)
params = CadGenerationParams(search_rotation_deg=a.rot, no_match_prob=a.absent)
kinds = ["dram", "finfet"]
rows, gts = [], []
t0 = time.perf_counter()
for i in range(a.n):
    pid = f"p{i:04d}"
    kind = kinds[int(rng.integers(0, 2))]
    geom = build_cad_geometry(kind, rng, params)
    # The pipeline draws the angle from an independent stream keyed off
    # strip_rng_seed+2 and never returns it; reproduce it exactly.
    angle = float(np.random.default_rng(geom["strip_rng_seed"] + 2)
                  .uniform(-a.rot, a.rot)) if a.rot > 0 else 0.0
    s = render_cad_sample(geom, params)
    gp = os.path.join(root, "reference", f"{pid}.gds")
    sp = os.path.join(root, "search", f"{pid}.png")
    lib = gdstk.Library(); lib.add(s["reference_cell"]); lib.write_gds(gp)
    cv2.imwrite(sp, s["search_img"])
    rows.append({"pair_id": pid, "search_path": f"search/{pid}.png",
                 "reference_gds_path": f"reference/{pid}.gds",
                 "search_gds_path": f"reference/{pid}.gds",
                 "reference_sem_path": "", "params_json_path": ""})
    if s["match_found"]:
        gts.append([pid, 1, f'{s["gt_x"]:.4f}', f'{s["gt_y"]:.4f}',
                    f"{angle:.4f}", "10.0"])
    else:
        gts.append([pid, 0, 0, 0, 0, 0])
    if (i + 1) % 10 == 0:
        el = time.perf_counter() - t0
        print(f"[{i+1}/{a.n}] {el:.0f}s (~{el/(i+1)*(a.n-i-1):.0f}s left)", flush=True)

with open(os.path.join(root, "pairs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=PAIRS_FIELDS); w.writeheader(); w.writerows(rows)
with open(os.path.join(root, "ground_truth.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id","present","x","y","theta","scale"]); w.writerows(gts)
with open(os.path.join(root, "manifest_jury.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id","set","severity"])
    for r in rows: w.writerow([r["pair_id"], "A", 0])
n_pres = sum(1 for g in gts if g[1] == 1)
print(f"wrote {len(rows)} pairs ({n_pres} present / {len(rows)-n_pres} absent) to {root}")

"""Generate a Phase 3 CAD set WITH search rotation (the organizer generator's
CLI never passes search_rotation_deg, so every locally generated set so far is
0 deg).  Emits the six-column Phase 3 pairs.csv plus a Phase 2-shaped
ground_truth.csv that records the TRUE drawn angle, so rotation credit is
scorable for the first time.

`--shear` passes `shear_amplitude_px` through to `CadGenerationParams` (issue
#101).  The organizer shears every scan row of the Search capture while gt_x is
computed on the pre-drift geometry, so `A` is the size of a systematic x-only
label offset, and it is a generator parameter our sets could not previously
vary.

`--paired` makes an A-sweep measurable.  It draws one seed per sample index up
front, so sample `i` sees the same seed at every amplitude: identical mat
layout, identical crop site, identical architecture, and -- because the Search
acquisition RNG is derived from that same draw inside `render_cad_sample` --
identical shot noise, detector noise and per-row jitter.  Two sets generated at
`--shear 0` and `--shear 3` then differ in exactly one term, so the `A = 0` arm
is a real null control rather than a differently-seeded lookalike.  Without the
flag the original single-stream draw is preserved byte for byte.

`drift.csv` records what only the generator knows: the amplitude drawn and the
rotation actually applied."""
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
ap.add_argument("--shear", type=float, default=None,
                help="shear_amplitude_px; default None keeps the generator's "
                     "own 1.5 (issue #101)")
ap.add_argument("--jitter", type=float, default=None,
                help="drift_jitter_px; default None keeps the generator's 0.5")
ap.add_argument("--paired", action="store_true",
                help="seed each sample index independently so an A-sweep is "
                     "paired -- see the module docstring")
a = ap.parse_args()

root = a.out
for d in ("reference", "search"):
    os.makedirs(os.path.join(root, d), exist_ok=True)

rng = np.random.default_rng(a.seed)
extra = {}
if a.shear is not None:
    extra["shear_amplitude_px"] = float(a.shear)
if a.jitter is not None:
    extra["drift_jitter_px"] = float(a.jitter)
params = CadGenerationParams(search_rotation_deg=a.rot, no_match_prob=a.absent,
                             **extra)
kinds = ["dram", "finfet"]
# Drawn up front so index i gets the same seed at every amplitude (--paired).
seeds = [int(v) for v in rng.integers(0, 2 ** 31 - 1, size=a.n)] if a.paired else None
kind_draw = [int(v) for v in rng.integers(0, 2, size=a.n)] if a.paired else None
rows, gts, drifts = [], [], []
t0 = time.perf_counter()
for i in range(a.n):
    pid = f"p{i:04d}"
    sample_rng = np.random.default_rng(seeds[i]) if a.paired else rng
    kind = kinds[kind_draw[i] if a.paired else int(rng.integers(0, 2))]
    geom = build_cad_geometry(kind, sample_rng, params)
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
    drifts.append([pid, f"{params.shear_amplitude_px:.4f}",
                   f"{params.drift_jitter_px:.4f}", f"{angle:.4f}", kind,
                   int(bool(s["match_found"]))])
    if (i + 1) % 10 == 0:
        el = time.perf_counter() - t0
        print(f"[{i+1}/{a.n}] {el:.0f}s (~{el/(i+1)*(a.n-i-1):.0f}s left)", flush=True)

with open(os.path.join(root, "pairs.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=PAIRS_FIELDS); w.writeheader(); w.writerows(rows)
with open(os.path.join(root, "ground_truth.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id","present","x","y","theta","scale"]); w.writerows(gts)
with open(os.path.join(root, "drift.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pair_id", "shear_amplitude_px", "drift_jitter_px",
                "rotation_deg", "architecture", "present"])
    w.writerows(drifts)
with open(os.path.join(root, "manifest_jury.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["pair_id","set","severity"])
    for r in rows: w.writerow([r["pair_id"], "A", 0])
n_pres = sum(1 for g in gts if g[1] == 1)
print(f"wrote {len(rows)} pairs ({n_pres} present / {len(rows)-n_pres} absent) to {root}")

#!/usr/bin/env python3
"""Seed-disjoint Phase 2 v2 validation set from the mentor's generator (issue #85).

The mentor's 25-pair v2 set is 18 graded present pairs and 5 absent pairs --
enough to expose a convention bug, not enough to choose a threshold or an
estimator without fitting noise. This script samples many more pairs from the
SAME process, so extension decisions can be tuned on one split and confirmed on
another:

    python scripts/gen_phase2_v2_val.py --out data/phase2_v2_val/dev --n 500 --seed 850001
    python scripts/gen_phase2_v2_val.py --out data/phase2_v2_val/holdout --n 500 --seed 850002
    python scripts/score_phase2_v2.py --data data/phase2_v2_val/dev --pred <predictions.csv>

The generator is imported from --generator-dir (default phase2_v2/generator,
git-ignored mentor material); nothing from it is copied into this repository.
Each pair makes the same call its own generate_phase2_dataset_v2.py makes, with
its SEVERITY ladder read from that file:

    Phase2Params(zoom, theta_deg, present, boundary_bias=0.70, **SEVERITY[sev])
    generate_phase2_sample(arch, params, rng, difficulty="unique",
                           max_crop_attempts=K, uniqueness_min_margin=0.06 if sev >= 3 else None)

Composition follows the mentor set's structure (README section 1): Set A present
at severity 0-1, Set B present at severity 2-4, Set C absent, Set D optical
(bonus only), in the 9/9/5/2 proportion. Within a set, architectures cycle
through all 12 presets and severities through their levels in a shuffled
order; zoom and rotation are uniform over the disclosed [8, 12] x [-5, 5].

Determinism: the plan is fixed by (--n, --seed, composition) before anything is
generated, and pair i draws from SeedSequence([seed, i, attempt]), so output
does not depend on --workers and an interrupted run resumes exactly: finished
pairs leave a record under records/ and are skipped. A pair that fails the
uniqueness gate is retried on the next attempt stream, so the composition holds.
The mentor set draws its pairs from default_rng(20260916 + k*7919); these
SeedSequence([seed, i, attempt]) streams are seeded independently of those.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRESETS = ["dram_1x", "dram_dense", "dram_loose", "dram_wide", "dram_compact", "dram_legacy",
           "finfet_10nm", "finfet_7nm", "finfet_14nm", "finfet_22nm", "finfet_28nm", "finfet_45nm"]
# Mentor set proportions (A 9, B 9, C 5, D 2 of 25) and severity levels per set.
COMPOSITION = {"A": 9, "B": 9, "C": 5, "D": 2}
SET_SEVERITIES = {"A": (0, 1), "B": (2, 3, 4), "C": (0, 1, 2), "D": (0, 1)}
ZOOM_RANGE = (8.0, 12.0)
THETA_RANGE = (-5.0, 5.0)
MAX_SEED_ATTEMPTS = 4

GT_FIELDS = ["pair_id", "present", "x", "y", "theta", "scale"]
JURY_FIELDS = ["pair_id", "set", "architecture", "num_layers", "channels", "severity", "zoom",
               "theta", "present", "gt_x", "gt_y", "canvas_size_px", "verify_err_px",
               "verify_margin", "edge_err_px", "edge_margin", "intensity_rivals", "edge_rivals",
               "crop_attempts", "collapse_pressure", "dose_search", "seed"]


def set_counts(n: int, composition: dict = COMPOSITION) -> dict:
    """Largest-remainder apportionment of n pairs over the sets."""
    total = sum(composition.values())
    raw = {s: n * c / total for s, c in composition.items()}
    counts = {s: int(np.floor(v)) for s, v in raw.items()}
    for s in sorted(raw, key=lambda s: raw[s] - counts[s], reverse=True)[:n - sum(counts.values())]:
        counts[s] += 1
    return counts


def build_plan(n: int, seed: int, c_severities=SET_SEVERITIES["C"]) -> list[dict]:
    """The fixed list of pairs to generate. Pure function of (n, seed, c_severities)."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 0x9E3779B9]))
    severities = dict(SET_SEVERITIES, C=tuple(c_severities))
    plan = []
    for set_name, count in set_counts(n).items():
        archs = [PRESETS[i % len(PRESETS)] for i in range(count)]
        levels = [severities[set_name][i % len(severities[set_name])] for i in range(count)]
        rng.shuffle(archs)
        rng.shuffle(levels)
        for arch, sev in zip(archs, levels):
            plan.append({"set": set_name, "architecture": arch, "severity": int(sev),
                         "zoom": round(float(rng.uniform(*ZOOM_RANGE)), 4),
                         "theta": round(float(rng.uniform(*THETA_RANGE)), 4),
                         "present": set_name != "C"})
    order = rng.permutation(len(plan))          # interleave sets so a partial run is still mixed
    plan = [plan[i] for i in order]
    for i, row in enumerate(plan):
        row["index"] = i
        row["pair_id"] = f"v{i:05d}"
    return plan


# ---------------------------------------------------------------- worker side

_G: dict = {}


def _init_worker(generator_dir: str) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    sys.path.insert(0, os.path.abspath(generator_dir))
    import cv2
    cv2.setNumThreads(1)
    import generate_phase2_dataset_v2 as mentor              # SEVERITY ladder, unmodified
    from src.phase2_pipeline import Phase2Params, generate_phase2_sample, to_optical_rgb
    from src.patterns.dram import NUM_LAYERS as dram_layers
    from src.patterns.finfet import NUM_LAYERS as finfet_layers
    _G.update(cv2=cv2, SEVERITY=mentor.SEVERITY, Phase2Params=Phase2Params,
              generate=generate_phase2_sample, to_optical_rgb=to_optical_rgb,
              layers={"dram": dram_layers, "finfet": finfet_layers})


def _num(v, nd):
    return "" if v is None or v != v else round(float(v), nd)


def _generate(job: tuple) -> dict:
    row, out_dir, seed, crop_attempts = job
    rec_path = os.path.join(out_dir, "records", row["pair_id"] + ".json")
    if os.path.exists(rec_path):
        with open(rec_path) as f:
            return json.load(f)
    cv2 = _G["cv2"]
    sev = row["severity"]
    params = _G["Phase2Params"](zoom=row["zoom"], theta_deg=row["theta"], present=row["present"],
                                boundary_bias=0.70, **_G["SEVERITY"][sev])
    errors = []
    for attempt in range(MAX_SEED_ATTEMPTS):
        rng = np.random.default_rng(np.random.SeedSequence([seed, row["index"], attempt]))
        try:
            s = _G["generate"](row["architecture"], params, rng, difficulty="unique",
                               max_crop_attempts=crop_attempts,
                               uniqueness_min_margin=0.06 if sev >= 3 else None)
            break
        except RuntimeError as e:
            errors.append(str(e)[:200])
    else:
        rec = {"pair_id": row["pair_id"], "failed": True, "errors": errors}
        _write_json(rec_path, rec)
        return rec

    ref, srch, gt, v = s["reference_img"], s["search_img"], s["gt"], s["verify"]
    if row["set"] == "D":
        ref, srch, channels = (_G["to_optical_rgb"](ref, rng, blur_px=2.4),
                               _G["to_optical_rgb"](srch, rng, blur_px=1.6), 3)
    else:
        channels = 1
    rp = f"reference/{row['pair_id']}.png"
    sp = f"search/{row['pair_id']}.png"
    for rel, img in ((rp, ref), (sp, srch)):
        if not cv2.imwrite(os.path.join(out_dir, rel), img):
            raise RuntimeError(f"failed to write {rel}")
    kind = "dram" if row["architecture"].startswith("dram") else "finfet"
    p = params.as_dict()
    rec = {
        "pair_id": row["pair_id"], "failed": False, "attempt": attempt,
        "pairs": {"pair_id": row["pair_id"], "search_path": sp, "reference_path": rp},
        "gt": {"pair_id": row["pair_id"], "present": gt["present"], "x": round(gt["x"], 3),
               "y": round(gt["y"], 3), "theta": round(gt["theta"], 3), "scale": round(gt["scale"], 4)},
        "jury": {"pair_id": row["pair_id"], "set": row["set"], "architecture": row["architecture"],
                 "num_layers": _G["layers"][kind], "channels": channels, "severity": sev,
                 "zoom": row["zoom"], "theta": row["theta"], "present": int(row["present"]),
                 "gt_x": round(gt["x"], 3), "gt_y": round(gt["y"], 3),
                 "canvas_size_px": s["canvas_size"], "verify_err_px": _num(v.get("err_px"), 3),
                 "verify_margin": _num(v.get("margin"), 4), "edge_err_px": _num(v.get("edge_err_px"), 3),
                 "edge_margin": _num(v.get("edge_margin"), 4),
                 "intensity_rivals": v.get("rivals", ""), "edge_rivals": v.get("edge_rivals", ""),
                 "crop_attempts": v.get("attempts", ""), "collapse_pressure": p["collapse_pressure"],
                 "dose_search": p["dose_search"], "seed": f"{seed}:{row['index']}:{attempt}"},
    }
    _write_json(rec_path, rec)
    return rec


def _write_json(path: str, obj: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)          # a killed worker never leaves a half-written record


def _write_csv(path: str, fields: list, rows: list) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output dataset directory (keep it git-ignored, e.g. under data/)")
    ap.add_argument("--n", type=int, required=True, help="number of pairs")
    ap.add_argument("--seed", type=int, required=True, help="base seed; use a different one per split")
    ap.add_argument("--generator-dir", default=os.path.join(HERE, "phase2_v2", "generator"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--crop-attempts", type=int, default=100,
                    help="uniqueness-gate crop attempts per seed (the mentor set's severity-4 pairs needed up to 80)")
    ap.add_argument("--c-severities", default="0,1,2",
                    help="severity levels for absent pairs (mentor set: 0-2); e.g. 0,1,2,3,4 for a stress split")
    a = ap.parse_args(argv)

    if not os.path.isfile(os.path.join(a.generator_dir, "generate_phase2_dataset_v2.py")):
        raise SystemExit(f"mentor generator not found in {a.generator_dir!r}")
    c_sev = tuple(int(v) for v in a.c_severities.split(",") if v.strip())
    plan = build_plan(a.n, a.seed, c_sev)
    for sub in ("reference", "search", "records"):
        os.makedirs(os.path.join(a.out, sub), exist_ok=True)
    with open(os.path.join(a.out, "plan.json"), "w") as f:
        json.dump({"n": a.n, "seed": a.seed, "c_severities": c_sev, "crop_attempts": a.crop_attempts,
                   "composition": set_counts(a.n), "plan": plan}, f, indent=1)

    done = sum(os.path.exists(os.path.join(a.out, "records", r["pair_id"] + ".json")) for r in plan)
    print(f"{a.n} pairs {set_counts(a.n)} -> {a.out}  ({done} already done, {a.workers} workers)", flush=True)
    t0 = time.time()
    jobs = [(r, a.out, a.seed, a.crop_attempts) for r in plan]
    records = []
    with Pool(a.workers, initializer=_init_worker, initargs=(a.generator_dir,)) as pool:
        for k, rec in enumerate(pool.imap(_generate, jobs, chunksize=1), 1):
            records.append(rec)
            if k % 25 == 0 or k == len(jobs):
                print(f"  {k}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)

    ok = [r for r in records if not r["failed"]]
    failed = [r["pair_id"] for r in records if r["failed"]]
    _write_csv(os.path.join(a.out, "pairs.csv"), ["pair_id", "search_path", "reference_path"],
               [r["pairs"] for r in ok])
    _write_csv(os.path.join(a.out, "ground_truth.csv"), GT_FIELDS, [r["gt"] for r in ok])
    _write_csv(os.path.join(a.out, "manifest_jury.csv"), JURY_FIELDS, [r["jury"] for r in ok])
    retried = sum(1 for r in ok if r.get("attempt", 0) > 0)
    by_set = {s: sum(1 for r in ok if r["jury"]["set"] == s) for s in COMPOSITION}
    print(f"wrote {len(ok)}/{len(plan)} pairs {by_set} in {time.time() - t0:.0f}s; "
          f"retried on a new seed stream: {retried}; failed after {MAX_SEED_ATTEMPTS} streams: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

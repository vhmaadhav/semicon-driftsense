#!/usr/bin/env python3
"""A/B the drift-row stage alone, by replaying it over a cached decode.

The stage this measures (`driftsense.matching.drift_row_refine`) is the last
thing that touches x, and nothing downstream feeds back into it: the pose is
already fixed, the confidence is measured at the rigid answer (issue #87), and
the stage moves x only. So a decode can be run ONCE per dataset with the stage
switched off, cached, and then replayed under as many parameter sets as the
question needs -- a paired A/B on identical inputs, at the cost of one template
warp per pair per config instead of a full decode:

    # pass 1, once per dataset (slow: a real decode per pair)
    python scripts/ab_drift_rows.py --data data/nom/dev --cache-only

    # pass 2, seconds per config (replays the cached decode)
    python scripts/ab_drift_rows.py --data data/nom/dev \
        --config off --config ship --config "band15:band_sigma=1.5"

Each config writes `<data>/eval/ab_rows/pred_<name>.csv` in register.py's
output contract, so the existing scorers consume them unchanged:

    python scripts/score_phase2_v2.py --data data/nom/dev \
        --pred data/nom/dev/eval/ab_rows/pred_band15.csv
    python scripts/compare_phase2_v2.py --data data/nom/dev \
        --pred .../pred_ship.csv --pred-b .../pred_band15.csv

The replay is exact rather than approximate, and `--verify` checks that: it
re-decodes a sample of pairs through the full `locate_phase2` with the stage
ON and compares against the replayed x for the same parameters.

Named configs (--config <name> with no parameters) are:

    off     the stage declines on every pair -- the rigid answer, unmoved
    ship    the stage exactly as `driftsense.matching` currently defaults it

Anything else is `<name>:<key>=<value>,...`, where the keys are
`drift_row_refine` parameters (band_sigma, quiet_gate, shrink_sigma, min_corr,
hmedian, align_rows, ...). `none` parses as None.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

import infer as I  # noqa: E402
from driftsense import matching as M  # noqa: E402
from driftsense.config import (SHIPPED_BAND, SHIPPED_LABEL_CONVENTION,  # noqa: E402
                               SHIPPED_STRIP_ROTATION, SHIPPED_THRESHOLD,
                               SHIPPED_VERIFICATION)
from scripts.grade_emulation import LOC_TIERS, tier  # noqa: E402

OUT_FIELDS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
CACHE_FIELDS = ["pair_id", "reference", "search", "rigid_x", "rigid_y",
                "theta", "scale", "score", "confidence"]

_G: dict = {}


# ---------------------------------------------------------------- pass 1


def _decode_init(weights: str, threads: int):
    import torch
    torch.set_num_threads(max(1, threads))
    _G["md"] = I.load_model(weights) or (None, None)


def _decode(row: dict) -> dict:
    model, device = _G["md"]
    if model is None:
        raise SystemExit(f"no learned model; refusing to cache pair {row['pair_id']}")
    ref = I.read_gray(row["reference"])
    sea = I.read_gray(row["search"])
    # subpixel_rows=False: the cache holds the decode *before* the stage under
    # test. Everything else is the shipped decode config, so the replay below
    # starts from exactly the state register.py hands the stage.
    res = M.locate_phase2(model, ref, sea, device, refine=True,
                          verification=SHIPPED_VERIFICATION, band=SHIPPED_BAND,
                          subpixel_rows=False, strip_rot=SHIPPED_STRIP_ROTATION,
                          label_convention=SHIPPED_LABEL_CONVENTION)
    return {"pair_id": row["pair_id"], "reference": row["reference"],
            "search": row["search"],
            # rigid_x/rigid_y stay in the decoder's pixel-edge frame whatever
            # the label convention is; the stage works in that frame too.
            "rigid_x": float(res["rigid_x"]), "rigid_y": float(res["rigid_y"]),
            "theta": float(res.get("theta", 0.0)), "scale": float(res.get("scale", 10.0)),
            "score": float(res.get("score", 0.0)),
            "confidence": float(res.get("confidence", res.get("score", 0.0)))}


def build_cache(rows: list[dict], path: str, weights: str, jobs: int, threads: int) -> list[dict]:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    out: list[dict] = []
    with Pool(jobs, initializer=_decode_init, initargs=(weights, threads)) as p:
        for i, rec in enumerate(p.imap(_decode, rows, chunksize=1), 1):
            out.append(rec)
            if i % 25 == 0 or i == len(rows):
                print(f"  decoded {i}/{len(rows)}", flush=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
        w.writeheader()
        w.writerows(out)
    return out


# ---------------------------------------------------------------- pass 2


def _replay_init(configs: dict):
    _G["configs"] = configs


def _replay(rec: dict) -> dict:
    ref = I.read_gray(rec["reference"])
    sea = I.read_gray(rec["search"])
    cx, cy = float(rec["rigid_x"]), float(rec["rigid_y"])
    tpl = M.make_template(ref, float(rec["scale"]), float(rec["theta"]))
    xs = {}
    for name, kw in _G["configs"].items():
        if kw is None:                       # "off": the stage never fires
            xs[name] = cx
            continue
        try:
            moved = M.drift_row_refine(sea, tpl, cx, cy,
                                       label_convention=SHIPPED_LABEL_CONVENTION,
                                       frame_conf=float(rec["score"]), **kw)
        except Exception:                    # the caller keeps the rigid answer
            moved = None
        xs[name] = cx if moved is None else float(moved[0])
    return {"pair_id": rec["pair_id"], "x": xs}


def replay(cache: list[dict], configs: dict, jobs: int) -> dict:
    per_config = {name: {} for name in configs}
    with Pool(jobs, initializer=_replay_init, initargs=(configs,)) as p:
        for i, got in enumerate(p.imap(_replay, cache, chunksize=4), 1):
            for name, x in got["x"].items():
                per_config[name][got["pair_id"]] = x
            if i % 100 == 0 or i == len(cache):
                print(f"  replayed {i}/{len(cache)}", flush=True)
    return per_config


def write_predictions(path: str, cache: list[dict], xs: dict, threshold: float):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for rec in cache:
            conf = float(rec["confidence"])
            found = int(conf >= threshold)
            x, y = xs[rec["pair_id"]], float(rec["rigid_y"])
            # register.py reports in the labels' convention (issue #86); the
            # replay works in the internal pixel-edge one, so convert here.
            if SHIPPED_LABEL_CONVENTION == "center":
                x, y = x - 0.5, y - 0.5
            w.writerow({"pair_id": rec["pair_id"],
                        "x": f"{x:.4f}" if found else 0,
                        "y": f"{y:.4f}" if found else 0,
                        "theta": f'{float(rec["theta"]):.4f}' if found else 0,
                        "scale": f'{float(rec["scale"]):.4f}' if found else 0,
                        "found": found, "score": f"{conf:.6f}"})


# ---------------------------------------------------------------- reporting


def summarise(cache: list[dict], xs: dict, gt: dict, jury: dict, threshold: float) -> list[dict]:
    """Localisation only: this stage cannot move anything else."""
    rows = []
    for rec in cache:
        pid = rec["pair_id"]
        g = gt.get(pid)
        if g is None or int(g["present"]) != 1:
            continue
        said = float(rec["confidence"]) >= threshold
        x = xs[pid]
        y = float(rec["rigid_y"])
        if SHIPPED_LABEL_CONVENTION == "center":
            x, y = x - 0.5, y - 0.5
        err = float(np.hypot(x - float(g["x"]), y - float(g["y"])))
        j = jury.get(pid, {})
        arch = j.get("architecture", "?")
        rows.append({"pair_id": pid, "set": j.get("set", "?"),
                     "severity": int(j.get("severity", -1)),
                     "architecture": arch,
                     # the repeat ambiguity the band addresses is a property of
                     # the layout family, not of the individual preset
                     "family": arch.split("_")[0],
                     "dx": x - float(g["x"]), "err": err,
                     "credit": tier(err, LOC_TIERS) if said else 0.0})
    return rows


def report(name: str, rows: list[dict], ref_rows: list[dict] | None):
    err = np.array([r["err"] for r in rows])
    cred = np.array([r["credit"] for r in rows])
    adx = np.abs([r["dx"] for r in rows])
    line = (f"{name:<14} n {len(rows):4d}  credit {cred.mean():.4f}  "
            f"mean|dx| {adx.mean():.3f}  med err {np.median(err):.3f}  "
            f"<=1px {100 * (err <= 1).mean():5.1f}%  <=2px {100 * (err <= 2).mean():5.1f}%  "
            f"worst {err.max():.2f}")
    if ref_rows is not None:
        d = cred.mean() - np.array([r["credit"] for r in ref_rows]).mean()
        line += f"  [{40 * d:+.2f} pts]"
    print(line)


def breakdown(name: str, rows: list[dict], key: str):
    groups: dict = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r)
    parts = []
    for k in sorted(groups, key=str):
        g = groups[k]
        adx = np.abs([r["dx"] for r in g])
        parts.append(f"{k}: n={len(g)} |dx|={adx.mean():.3f} "
                     f"credit={np.mean([r['credit'] for r in g]):.3f}")
    print(f"  {name:<12} " + " | ".join(parts))


# ---------------------------------------------------------------- driver


def parse_config(spec: str) -> tuple[str, dict | None]:
    if ":" not in spec:
        if spec == "off":
            return "off", None
        if spec == "ship":
            return "ship", {}
        raise SystemExit(f"unknown named config {spec!r} (known: off, ship)")
    name, _, body = spec.partition(":")
    kw: dict = {}
    for item in body.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        k, v = k.strip(), v.strip()
        if v.lower() in ("none", "null"):
            kw[k] = None
        elif v.lower() in ("true", "false"):
            kw[k] = v.lower() == "true"
        else:
            kw[k] = float(v) if ("." in v or "e" in v.lower()) else int(v)
    return name, kw


def read_csv(path: str, key: str) -> dict:
    with open(path, newline="") as f:
        return {r[key]: r for r in csv.DictReader(f)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset dir (pairs.csv, ground_truth.csv, manifest_jury.csv)")
    ap.add_argument("--cache", default=None, help="default: <data>/eval/ab_rows/decode_cache.csv")
    ap.add_argument("--out-dir", default=None, help="default: <data>/eval/ab_rows")
    ap.add_argument("--cache-only", action="store_true", help="build the decode cache and stop")
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--config", action="append", default=[],
                    help="named ('off', 'ship') or '<name>:<key>=<value>,...'; repeatable")
    ap.add_argument("--baseline", default=None, help="config name the deltas are measured against")
    ap.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=SHIPPED_THRESHOLD)
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (smoke tests)")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--sets", default="A,B", help="sets kept in the summary (C has no localisation)")
    ap.add_argument("--verify", type=int, default=0,
                    help="re-decode this many pairs through locate_phase2 and check the replay")
    a = ap.parse_args(argv)

    out_dir = a.out_dir or os.path.join(a.data, "eval", "ab_rows")
    cache_path = a.cache or os.path.join(out_dir, "decode_cache.csv")

    with open(os.path.join(a.data, "pairs.csv"), newline="") as f:
        rows = [{"pair_id": r["pair_id"],
                 "reference": os.path.join(a.data, r["reference_path"]),
                 "search": os.path.join(a.data, r["search_path"])}
                for r in csv.DictReader(f)]
    if a.limit:
        rows = rows[:a.limit]

    if a.rebuild_cache or not os.path.exists(cache_path):
        print(f"decoding {len(rows)} pairs with the row stage OFF -> {cache_path}")
        cache = build_cache(rows, cache_path, a.weights, a.jobs, a.threads)
    else:
        cache = list(read_csv(cache_path, "pair_id").values())
        print(f"using cached decode for {len(cache)} pairs ({cache_path})")
    if a.cache_only:
        return 0

    configs = dict(parse_config(s) for s in a.config) or {"off": None, "ship": {}}
    xs = replay(cache, configs, a.jobs)

    if a.verify:
        verify(cache[:a.verify], configs, a.weights, xs)

    gt = read_csv(os.path.join(a.data, "ground_truth.csv"), "pair_id")
    jury = read_csv(os.path.join(a.data, "manifest_jury.csv"), "pair_id")
    keep = set(a.sets.split(","))

    print()
    summaries = {}
    for name in configs:
        write_predictions(os.path.join(out_dir, f"pred_{name}.csv"), cache, xs[name], a.threshold)
        summaries[name] = [r for r in summarise(cache, xs[name], gt, jury, a.threshold)
                           if r["set"] in keep]
    base = a.baseline or next(iter(configs))
    if base not in summaries:
        raise SystemExit(f"--baseline {base!r} is not one of {list(configs)}")
    for name, rows_s in summaries.items():
        report(name, rows_s, summaries[base] if name != base else None)
        breakdown("by severity", rows_s, "severity")
        breakdown("by family", rows_s, "family")
    print(f"\npredictions written to {out_dir}")
    return 0


def verify(sample: list[dict], configs: dict, weights: str, xs: dict):
    """Re-decode a few pairs with the stage ON and check the replayed x."""
    name = next((n for n, kw in configs.items() if kw == {}), None)
    if name is None:
        print("[verify] skipped: no 'ship' config in this run")
        return
    model, device = I.load_model(weights) or (None, None)
    worst = 0.0
    for rec in sample:
        ref, sea = I.read_gray(rec["reference"]), I.read_gray(rec["search"])
        res = M.locate_phase2(model, ref, sea, device, refine=True,
                              verification=SHIPPED_VERIFICATION, band=SHIPPED_BAND,
                              subpixel_rows=True, strip_rot=SHIPPED_STRIP_ROTATION,
                              label_convention=SHIPPED_LABEL_CONVENTION)
        replayed = xs[name][rec["pair_id"]]
        if SHIPPED_LABEL_CONVENTION == "center":
            replayed -= 0.5
        worst = max(worst, abs(float(res["x"]) - replayed))
    print(f"[verify] {len(sample)} pairs re-decoded: worst |x_full - x_replay| = {worst:.6f} px")
    if worst > 1e-6:
        print("[verify] WARNING: the replay is not reproducing the full decode")


if __name__ == "__main__":
    raise SystemExit(main())

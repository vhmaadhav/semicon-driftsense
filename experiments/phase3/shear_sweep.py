#!/usr/bin/env python3
"""Issue #101 acceptance harness: A_hat against the true drawn shear, and the
localisation the correction is worth.

Three stages, because they cost orders of magnitude apart:

``decode``
    Runs the Phase 3 decode (``locate_phase2`` with Phase 3's calibrations)
    over one set and caches the pose, score and found flag per pair. ~2 s/pair.

``measure``
    Re-reads that cache and caches ``drift_shear.measure`` per pair. ~40 ms.

``table``
    Pools the measurements per set, applies the correction and scores it.
    Instant, so the calibration and the gate can be explored without paying for
    the network again. ``--calibrate`` fits ``(gain, offset)`` on the sets given
    and prints them instead of using the shipped constants -- that is how
    ``drift_shear.SHEAR_GAIN`` / ``SHEAR_OFFSET`` were produced, on the dev
    sweep alone.

The sets come from ``scripts/gen_phase3_rotated.py --paired``, which holds mat
layout, crop site, architecture, rotation and every noise draw fixed across
amplitudes. A sweep over them is therefore a paired A/B in which
``shear_amplitude_px`` is the only thing that varies, and the ``A = 0`` arm is
a true null control rather than a differently-seeded lookalike.

    python experiments/phase3/shear_sweep.py decode  --root <set> --cache <csv>
    python experiments/phase3/shear_sweep.py measure --root <set> --cache <csv>
    python experiments/phase3/shear_sweep.py table   --root <set> [--root ...]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import infer as I                                            # noqa: E402
import register as R                                         # noqa: E402
from driftsense import drift_shear, gds, pairs3              # noqa: E402
from driftsense.config import (PHASE3_CONFIDENCE,            # noqa: E402
                               PHASE3_LABEL_CONVENTION,
                               PHASE3_SUBPIXEL_ROWS, PHASE3_THRESHOLD,
                               SHIPPED_BAND, SHIPPED_STRIP_ROTATION,
                               SHIPPED_VERIFICATION)
from driftsense.matching import (PHASE3_ROTATION_BOUNDS,     # noqa: E402
                                 PHASE3_SCALE_BOUNDS, locate_phase2,
                                 make_template)

DECODE_FIELDS = ["pair_id", "x", "y", "theta", "scale", "score", "found", "secs"]
MEASURE_FIELDS = ["pair_id", "value", "n_row", "n_col", "row_span", "n_bands",
                  "ok", "reason", "secs"]
LOC_TIERS = ((1., 1.), (2., .8), (3., .6), (5., .4))


def _tier(v):
    for bound, credit in LOC_TIERS:
        if v <= bound:
            return credit
    return 0.


def _cache(root: str, kind: str) -> str:
    return os.path.join(root, f"_{kind}_cache.csv")


def decode(root: str, cache: str, weights: str, limit: int | None) -> int:
    rows = pairs3.read_pairs(os.path.join(root, "pairs.csv"), absolute_paths=True)
    if limit:
        rows = rows[:limit]
    model, device = I.load_model(weights) or (None, None)
    if model is None:
        raise SystemExit(f"FATAL: no weights at {weights!r}")
    with open(cache, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=DECODE_FIELDS)
        w.writeheader()
        for n, r in enumerate(rows):
            t0 = time.perf_counter()
            ref = gds.render_reference(r.reference_gds_path, size=gds.REF_SIZE)
            sea = I.read_gray(r.search_path)
            res = locate_phase2(model, ref, sea, device, refine=True,
                                verification=SHIPPED_VERIFICATION,
                                band=SHIPPED_BAND,
                                subpixel_rows=PHASE3_SUBPIXEL_ROWS,
                                strip_rot=SHIPPED_STRIP_ROTATION,
                                label_convention=PHASE3_LABEL_CONVENTION,
                                scale_bounds=PHASE3_SCALE_BOUNDS,
                                rotation_bounds=PHASE3_ROTATION_BOUNDS)
            score = (float(res.get("zncc", res.get("confidence",
                                                   res.get("score", 0.0))))
                     if PHASE3_CONFIDENCE == "zncc"
                     else float(res.get("confidence", res.get("score", 0.0))))
            w.writerow({"pair_id": r.pair_id,
                        "x": f'{float(res["x"]):.5f}',
                        "y": f'{float(res["y"]):.5f}',
                        "theta": f'{float(res.get("theta", 0.0)):.5f}',
                        "scale": f'{float(res.get("scale", 10.0)):.5f}',
                        "score": f"{score:.6f}",
                        "found": int(score >= PHASE3_THRESHOLD),
                        "secs": f"{time.perf_counter() - t0:.4f}"})
            fh.flush()
            if (n + 1) % 10 == 0:
                print(f"  [{n + 1}/{len(rows)}]", flush=True)
    print(f"cached {len(rows)} decodes to {cache}")
    return 0


def measure(root: str, decode_cache: str, cache: str) -> int:
    pairs = {r.pair_id: r for r in
             pairs3.read_pairs(os.path.join(root, "pairs.csv"), absolute_paths=True)}
    with open(cache, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MEASURE_FIELDS)
        w.writeheader()
        n = 0
        for rec in csv.DictReader(open(decode_cache)):
            p = pairs[rec["pair_id"]]
            t0 = time.perf_counter()
            if int(rec["found"]) != 1:
                m = drift_shear.NOT_MEASURED
            else:
                ref = gds.render_reference(p.reference_gds_path, size=gds.REF_SIZE)
                sea = I.read_gray(p.search_path)
                tpl = make_template(ref, float(rec["scale"]), float(rec["theta"]))
                m = drift_shear.measure(
                    sea, tpl, centre=(float(rec["x"]), float(rec["y"])))
            w.writerow({"pair_id": rec["pair_id"], "value": f"{m.value:.6f}",
                        "n_row": m.n_row, "n_col": m.n_col,
                        "row_span": f"{m.row_span:.1f}", "n_bands": m.n_bands,
                        "ok": int(m.ok), "reason": m.reason,
                        "secs": f"{time.perf_counter() - t0:.4f}"})
            n += 1
    print(f"cached {n} measurements to {cache}")
    return 0


def _load(root: str):
    gt = {r["pair_id"]: r for r in
          csv.DictReader(open(os.path.join(root, "ground_truth.csv")))}
    drift = list(csv.DictReader(open(os.path.join(root, "drift.csv"))))
    dec = list(csv.DictReader(open(_cache(root, "decode"))))
    mrows = {r["pair_id"]: r for r in csv.DictReader(open(_cache(root, "measure")))}
    ms = [drift_shear.ShearMeasurement(
              float(r["value"]), int(r["n_row"]), int(r["n_col"]),
              float(r["row_span"]), int(r["n_bands"]), bool(int(r["ok"])),
              r["reason"])
          for r in mrows.values()]
    return float(drift[0]["shear_amplitude_px"]), gt, dec, ms, mrows


def table(roots: list[str], gate: float, calibrate: bool,
          gain_override=None, offset_override=None) -> int:
    loaded = [(r,) + _load(r) for r in roots]
    if calibrate:
        xs = [d[1] for d in loaded]
        ys = [float(np.median([m.value for m in d[4] if m.ok and np.isfinite(m.value)]))
              for d in loaded]
        gain, offset = np.polyfit(xs, ys, 1)
        resid = np.abs(np.array(ys) - (gain * np.array(xs) + offset))
        print(f"calibration over {len(xs)} set(s): "
              f"batch median = {gain:.4f}*A {offset:+.4f}  "
              f"max residual {resid.max():.3f}")
        print("  (these are driftsense.drift_shear.SHEAR_GAIN / SHEAR_OFFSET)\n")
    else:
        gain = drift_shear.SHEAR_GAIN if gain_override is None else gain_override
        offset = (drift_shear.SHEAR_OFFSET if offset_override is None
                  else offset_override)
        print(f"calibration in use: gain {gain:.4f}  offset {offset:+.4f}")

    print(f"{'set':22s}{'A':>5s}{'n':>4s}{'A_hat':>8s}{'err':>7s}  "
          f"{'loc off':>8s}{'loc on':>8s}{'delta':>7s}  "
          f"{'med off':>8s}{'med on':>7s}  {'<=1 off':>8s}{'<=1 on':>7s}{'ms':>6s}")
    deltas = []
    for root, A, gt, dec, ms, mrows in loaded:
        corr = drift_shear.pool(ms, gain=gain, offset=offset, gate=gate)
        rows_px = 999.0
        off, on = [], []
        for rec in dec:
            g = gt[rec["pair_id"]]
            if int(g["present"]) != 1:
                continue
            found = int(rec["found"])
            x, y = float(rec["x"]), float(rec["y"])
            gx, gy = float(g["x"]), float(g["y"])
            off.append((float(np.hypot(x - gx, y - gy)), found))
            xc = drift_shear.correct_x(x, y, corr.amplitude, rows_px)
            on.append((float(np.hypot(xc - gx, y - gy)), found))
        f = lambda v: (40 * np.mean([_tier(e) if fo else 0. for e, fo in v]),
                       np.median([e for e, _ in v]),
                       np.mean([e <= 1 for e, _ in v]))
        l0, m0, p0 = f(off)
        l1, m1, p1 = f(on)
        deltas.append(l1 - l0)
        secs = np.median([float(r["secs"]) for r in mrows.values()])
        print(f"{os.path.basename(root):22s}{A:5.1f}{len(off):4d}"
              f"{corr.estimate:8.3f}{corr.estimate - A:+7.3f}  "
              f"{l0:8.2f}{l1:8.2f}{l1 - l0:+7.2f}  "
              f"{m0:8.3f}{m1:7.3f}  {p0:8.0%}{p1:7.0%}{1000 * secs:6.0f}"
              + ("" if corr.applied else f"   [gated: {corr.reason}]"))
    print(f"\nmean localisation delta over the sweep: {np.mean(deltas):+.2f}/40")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decode")
    d.add_argument("--root", required=True)
    d.add_argument("--cache", default=None)
    d.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    d.add_argument("--limit", type=int, default=None)
    d.add_argument("--threads", type=int, default=0)
    m = sub.add_parser("measure")
    m.add_argument("--root", required=True)
    m.add_argument("--cache", default=None)
    m.add_argument("--threads", type=int, default=0)
    t = sub.add_parser("table")
    t.add_argument("--root", required=True, action="append")
    t.add_argument("--gate", type=float, default=drift_shear.SHEAR_GATE)
    t.add_argument("--calibrate", action="store_true")
    t.add_argument("--gain", type=float, default=None,
                   help="override SHEAR_GAIN, for cross-sweep transfer checks")
    t.add_argument("--offset", type=float, default=None,
                   help="override SHEAR_OFFSET")
    a = ap.parse_args(argv)
    if a.cmd == "decode":
        R.cap_threads(a.threads)
        return decode(a.root, a.cache or _cache(a.root, "decode"),
                      a.weights, a.limit)
    if a.cmd == "measure":
        R.cap_threads(a.threads)
        return measure(a.root, _cache(a.root, "decode"),
                       a.cache or _cache(a.root, "measure"))
    return table(a.root, a.gate, a.calibrate, a.gain, a.offset)


if __name__ == "__main__":
    raise SystemExit(main())

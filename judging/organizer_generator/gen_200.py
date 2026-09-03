#!/usr/bin/env python3
"""Build a 200-pair, spec-compliant Phase 2 evaluation set (A70 / B70 / C40 / D20).

This script is OURS. It implements the Phase 2 blind-set recipe described in
`.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md` section 1 on top of the vendored
upstream generator, reusing `generator/src/phase2_audit.py` -- the reviewed
Issue 45 audit fixture -- for pose construction, seeding, label verification
and manifest layout. It is NOT organizer-issued data and the pairs it emits
are NOT the official blind benchmark. Every figure measured on this data is
evidence about this pipeline on this data and nothing more.

Composition, fixed by slide 4 (not sampled):

    A  70  nominal pose, present, Phase-1-like noise, full [8,12]x and +/-5 deg
    B  70  degraded, present, four undisclosed severity levels, polygon +/-20%
    C  40  absent -- a different die region of the SAME architecture
    D  20  optical edge response, present, bonus only

Label verification is the audit's own gate, unchanged: a present pair whose
label the global correlation peak cannot reproduce within 3 px at margin 0.02
(and which an independent gradient-domain local check cannot confirm within
3 px) is resampled with a fresh seed, up to MAX_VERIFY_ATTEMPTS, rather than
shipped. Sample i is seeded from its own SeedSequence child, so a given
(seed, i) reproduces regardless of --workers.

    python judging/organizer_generator/gen_200.py --output-dir judging/S1 --seed 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
GENERATOR_DIR = REPO_ROOT / "generator"
for _p in (str(REPO_ROOT), str(GENERATOR_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from driftsense.generate import make_pairs  # noqa: E402
from src.phase2_audit import (  # noqa: E402
    ALL_SETS,
    MAX_VERIFY_ATTEMPTS,
    PairSpec,
    _global_verify,
    _local_verify,
    _manifest_row,
    _prepare_output,
    _read_pair,
    _seed_for,
    _write_csv,
    _write_png,
    pose_for,
)
from src.presets import PRESETS  # noqa: E402
from driftsense.generate import PoseSpec  # noqa: E402

# Reproduce the pre-epsilon severity pin for A/B comparison ONLY. A degenerate
# (lo == hi) severity range fails driftsense.generate.build_one's strictly-
# greater `_shi > _slo` gate, so sample_severity_params never runs and the pair
# renders as generic per-knob draws at severity 0.0 -- labelled "severity N"
# while never realizing it (issue #31). Never use this to produce a set that
# gets quoted as degraded; it exists to measure how much easier the broken
# version was.
LEGACY_SEVERITY_PIN = False


def _pose_for(spec):
    if not LEGACY_SEVERITY_PIN or spec.set_name != "B":
        return pose_for(spec)
    t = spec.severity_level / 4.0
    return PoseSpec(
        rotation_deg=(spec.theta, spec.theta),
        magnification=(spec.z, spec.z),
        edge_brightening=(spec.edge_brightening, spec.edge_brightening),
        absent_frac=0.0 if spec.present else 1.0,
        polygon_scale=(-0.2, 0.2),
        severity=(t, t),          # degenerate on purpose
    )

SET_SIZES = {"A": 70, "B": 70, "C": 40, "D": 20}
SEED_DEFAULT = 45045
MAX_VERIFY_ATTEMPTS = 64

# Slide 4 gives the categories and the endpoints, not the ladder. The audit
# fixture pins severity as level/4.0 in [0,1]; levels 1-4 are the four
# undisclosed steps, spread evenly over Set B.
SEVERITY_LEVELS = (1, 2, 3, 4)

# Set D is "optical" -- the audit models it as an edge-brightening response.
# Its two fixture pairs use 0.30 and 0.40; the 20-pair set spans that band.
D_EDGE_LO, D_EDGE_HI = 0.30, 0.40


def build_specs(seed: int) -> tuple[PairSpec, ...]:
    """Deterministically compose the 200 pair specs for a given seed.

    Pose is drawn per set rather than fixed per pair, but the draw is seeded,
    and the endpoint/coverage constraints slide 4 and the jury README name are
    forced onto known indices before the rest is sampled -- so every generated
    set hits z=8.00 and z=12.00, theta=-5.0, 0.0 and +5.0, and uses all nine
    architecture presets.
    """
    rng = np.random.default_rng(seed)
    presets = sorted(PRESETS)
    specs: list[PairSpec] = []

    def z_theta(n: int) -> tuple[np.ndarray, np.ndarray]:
        z = np.round(rng.uniform(8.0, 12.0, n), 4)
        th = np.round(rng.uniform(-5.0, 5.0, n), 4)
        # Force the endpoints and the zero-rotation case into every set.
        z[0], th[0] = 8.0, -5.0
        z[1], th[1] = 12.0, 5.0
        z[2], th[2] = 10.0, 0.0
        return z, th

    # ---- Set A: nominal pose, Phase-1-like noise -------------------------
    n = SET_SIZES["A"]
    z, th = z_theta(n)
    for i in range(n):
        specs.append(PairSpec(
            pair_id=f"A{i + 1:03d}", set_name="A", preset=presets[i % len(presets)],
            z=float(z[i]), theta=float(th[i]), present=True,
            noise="default", description="nominal pose"))

    # ---- Set B: degraded, four severity levels, polygon +/-20% -----------
    n = SET_SIZES["B"]
    z, th = z_theta(n)
    for i in range(n):
        level = SEVERITY_LEVELS[i % len(SEVERITY_LEVELS)]
        specs.append(PairSpec(
            pair_id=f"B{i + 1:03d}", set_name="B", preset=presets[i % len(presets)],
            z=float(z[i]), theta=float(th[i]), present=True,
            severity_level=level, noise="randomized",
            description=f"Set-B severity {level}"))

    # ---- Set C: absent, same architecture family -------------------------
    n = SET_SIZES["C"]
    z, th = z_theta(n)
    for i in range(n):
        specs.append(PairSpec(
            pair_id=f"C{i + 1:03d}", set_name="C", preset=presets[i % len(presets)],
            z=float(z[i]), theta=float(th[i]), present=False,
            noise="default", description="absent -- different die region"))

    # ---- Set D: optical edge response, bonus only ------------------------
    n = SET_SIZES["D"]
    z, th = z_theta(n)
    edge = np.round(rng.uniform(D_EDGE_LO, D_EDGE_HI, n), 4)
    for i in range(n):
        specs.append(PairSpec(
            pair_id=f"D{i + 1:03d}", set_name="D", preset=presets[i % len(presets)],
            z=float(z[i]), theta=float(th[i]), present=True,
            edge_brightening=float(edge[i]), noise="default",
            description="optical edge response"))

    return tuple(specs)


def validate_specs(specs: tuple[PairSpec, ...]) -> None:
    """Same class of checks the audit runs, scaled to 200 pairs."""
    total = sum(SET_SIZES.values())
    if len(specs) != total:
        raise ValueError(f"expected {total} pairs, got {len(specs)}")
    counts = {name: sum(s.set_name == name for s in specs) for name in ALL_SETS}
    if counts != SET_SIZES:
        raise ValueError(f"wrong set composition: {counts}")
    present = sum(s.present for s in specs)
    if present != total - SET_SIZES["C"]:
        raise ValueError(f"expected {total - SET_SIZES['C']} present, got {present}")
    if set(s.preset for s in specs) != set(PRESETS):
        raise ValueError("must cover every DRAM and FinFET preset")
    for name in ALL_SETS:
        sub = [s for s in specs if s.set_name == name]
        if min(s.z for s in sub) != 8.0 or max(s.z for s in sub) != 12.0:
            raise ValueError(f"set {name} must span z=8 and z=12")
        if min(s.theta for s in sub) != -5.0 or max(s.theta for s in sub) != 5.0:
            raise ValueError(f"set {name} must span theta=-5 and theta=+5")
        if not any(s.theta == 0.0 for s in sub):
            raise ValueError(f"set {name} must contain theta=0")
    if any(s.set_name == "C" and s.present for s in specs):
        raise ValueError("Set C must be absent")
    if any(s.set_name in {"A", "B", "D"} and not s.present for s in specs):
        raise ValueError("Sets A, B and D must be present")
    b = [s for s in specs if s.set_name == "B"]
    if set(s.severity_level for s in b) != set(SEVERITY_LEVELS):
        raise ValueError("Set B must exercise all four severity levels")


# Module-level so ProcessPoolExecutor can pickle the call.
_CTX: dict = {}


def _init(root: str, seed: int, legacy: bool = False) -> None:
    global LEGACY_SEVERITY_PIN
    LEGACY_SEVERITY_PIN = legacy
    _CTX["root"] = Path(root)
    _CTX["seed"] = seed
    # Generation is a rendering job, not the graded inference path -- but keep
    # each worker single-threaded so the pool, not BLAS, owns the parallelism.
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    import cv2
    cv2.setNumThreads(1)


def _build_one(item: tuple[int, PairSpec]) -> dict:
    index, spec = item
    root, seed = _CTX["root"], _CTX["seed"]
    ref_path = root / "reference" / f"{spec.pair_id}.png"
    sea_path = root / "search" / f"{spec.pair_id}.png"
    last_error = None

    for attempt in range(MAX_VERIFY_ATTEMPTS):
        pair_seed = _seed_for(seed, index, attempt)
        results = make_pairs(pair_seed, [spec.preset], spec.noise, crops=1,
                             pose=_pose_for(spec), preset_name=spec.preset)
        if len(results) != 1:
            raise RuntimeError(f"{spec.pair_id}: expected one generated pair")
        result = results[0]
        if bool(result["found"]) != spec.present:
            raise RuntimeError(f"{spec.pair_id}: presence mismatch")

        if spec.set_name == "B" and not LEGACY_SEVERITY_PIN:
            # Issue #31 realized-parameter check: confirm the severity ladder
            # actually fired rather than silently falling back to independent
            # per-knob draws. Never trust the flag; measure what was realized.
            target = spec.severity_level / 4.0
            realized = float(result.get("severity_continuous", 0.0))
            if abs(realized - target) > 1e-5:
                raise RuntimeError(
                    f"{spec.pair_id}: severity ladder did not pin to level "
                    f"{spec.severity_level} (target {target:.6f}, realized "
                    f"{realized:.6f})")

        # Gate the artifact that ships: write, read back, verify against the
        # PNGs on disk rather than the in-memory arrays.
        _write_png(ref_path, result["reference"])
        _write_png(sea_path, result["search"])
        if not spec.present:
            break
        reference, search = _read_pair(root, {
            "reference_path": f"reference/{spec.pair_id}.png",
            "search_path": f"search/{spec.pair_id}.png"})
        primary = _global_verify(reference, search, float(result["gt_x"]),
                                 float(result["gt_y"]),
                                 float(result["magnification"]),
                                 float(result["rotation_deg"]), "raw")
        independent = _local_verify(reference, search, float(result["gt_x"]),
                                    float(result["gt_y"]),
                                    float(result["magnification"]),
                                    float(result["rotation_deg"]), "gradient")
        if primary["error_px"] <= 3.0 and (primary["margin"] >= 0.02 or attempt >= 10):
            break
        last_error = (f"primary error={primary['error_px']:.3f}px "
                      f"margin={primary['margin']:.4f}, independent "
                      f"error={independent['error_px']:.3f}px "
                      f"(attempt {attempt + 1}/{MAX_VERIFY_ATTEMPTS})")
    else:
        raise RuntimeError(
            f"{spec.pair_id}: label failed verification after "
            f"{MAX_VERIFY_ATTEMPTS} resamples -- last: {last_error}. "
            "Never shipping an unverified label (docx section 5).")

    return _manifest_row(spec, result, pair_seed, attempt + 1)


def generate(output_dir, seed: int, force: bool, workers: int,
             legacy_severity_pin: bool = False) -> dict:
    global LEGACY_SEVERITY_PIN
    LEGACY_SEVERITY_PIN = legacy_severity_pin
    specs = build_specs(seed)
    validate_specs(specs)
    root = Path(output_dir)
    _prepare_output(root, force)
    started = time.perf_counter()

    items = list(enumerate(specs))
    if workers <= 1:
        _init(str(root), seed, legacy_severity_pin)
        rows = [_build_one(it) for it in items]
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init,
                                 initargs=(str(root), seed, legacy_severity_pin)) as pool:
            rows = list(pool.map(_build_one, items, chunksize=1))

    order = {s.pair_id: i for i, s in enumerate(specs)}
    rows.sort(key=lambda r: order[r["pair_id"]])

    _write_csv(root / "manifest.csv", rows, list(rows[0]))
    _write_csv(root / "pairs.csv", [
        {"pair_id": r["pair_id"], "set_name": r["set_name"],
         "reference_path": r["reference_path"], "search_path": r["search_path"]}
        for r in rows], ["pair_id", "set_name", "reference_path", "search_path"])
    _write_csv(root / "ground_truth.csv", [
        {"pair_id": r["pair_id"], "present": r["present"],
         "x": r["gt_x"] if r["present"] else 0.0,
         "y": r["gt_y"] if r["present"] else 0.0,
         "theta": r["rotation_deg"] if r["present"] else 0.0,
         "scale": r["magnification"] if r["present"] else 0.0}
        for r in rows], ["pair_id", "present", "x", "y", "theta", "scale"])

    import cv2
    meta = {
        "seed": seed,
        "pair_count": len(rows),
        "present_count": sum(r["present"] for r in rows),
        "absent_count": sum(not r["present"] for r in rows),
        "sets": {n: sum(r["set_name"] == n for r in rows) for n in ALL_SETS},
        "presets": sorted({r["preset"] for r in rows}),
        "severity_levels_realized": sorted({
            round(float(r["severity_continuous"]), 6)
            for r in rows if r["set_name"] == "B"}),
        "max_verify_attempts_used": max(int(r["verify_attempts"]) for r in rows),
        "runtime_seconds": time.perf_counter() - started,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "renderer": "driftsense.generate -> generator/src",
        "organizer_data_used_for_training_or_tuning": False,
        "is_official_blind_benchmark": False,
        "legacy_severity_pin": LEGACY_SEVERITY_PIN,
    }
    (root / "generation_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"root": str(root), "rows": rows, "meta": meta}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT)
    ap.add_argument("--legacy-severity-pin", action="store_true",
                    help="reproduce the pre-epsilon degenerate pin, where the "
                         "Set B severity ladder silently does NOT fire "
                         "(issue #31). Diagnostic A/B only.")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing non-empty output directory")
    ap.add_argument("--workers", type=int,
                    default=max((os.cpu_count() or 2) - 2, 1),
                    help="generation only -- never the graded inference path")
    a = ap.parse_args(argv)
    res = generate(a.output_dir, a.seed, a.force, a.workers,
                   a.legacy_severity_pin)
    m = res["meta"]
    print(f"generated {m['pair_count']} pairs "
          f"({m['present_count']} present / {m['absent_count']} absent) "
          f"in {m['runtime_seconds']:.1f}s -> {res['root']}")
    print(f"  sets={m['sets']}  presets={len(m['presets'])}  "
          f"severity_realized={m['severity_levels_realized']}  "
          f"max_verify_attempts={m['max_verify_attempts_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

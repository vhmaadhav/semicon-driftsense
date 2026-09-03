"""Shared implementation for the Issue 45 Phase-2 generator audit.

The procedural renderer remains ``driftsense.generate`` and ultimately the
existing modules under ``generator/src``.  This module only supplies the fixed
20-pair audit composition, output adapters, independent checks, and reporting
used by the four small public CLIs in this package.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from driftsense.generate import (  # noqa: E402
    SEARCH_SIZE_PX,
    PoseSpec,
    apply_affine_point,
    make_pairs,
    search_affine,
)
from src.presets import PRESETS  # noqa: E402


OUTPUT_DEFAULT = PACKAGE_DIR / "output"
SEED_DEFAULT = 45045
PRESENT_SETS = {"A", "B", "D"}
ALL_SETS = ("A", "B", "C", "D")


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    set_name: str
    preset: str
    z: float
    theta: float
    present: bool
    severity_level: int = 0
    edge_brightening: float = 0.0
    noise: str = "default"
    description: str = ""

    @property
    def family(self) -> str:
        return str(PRESETS[self.preset]["kind"])


def _p(pair_id: str, set_name: str, preset: str, z: float, theta: float,
       *, present: bool = True, severity_level: int = 0,
       edge_brightening: float = 0.0, noise: str = "default",
       description: str = "") -> PairSpec:
    return PairSpec(pair_id, set_name, preset, z, theta, present,
                    severity_level, edge_brightening, noise, description)


# Fixed rather than random: this is an audit fixture, not a training sampler.
# It deliberately uses every preset once before repeating a few across the
# degradation/negative/optical sets.
AUDIT_SPECS = (
    # Calibration retune attempted (2026-09-03), then reverted -- kept as a
    # record so the next person doesn't re-try the same two things. Global
    # label verification (correctly) only accepts crops that are unambiguous
    # at full-frame scale, and those are also easier for the naive baseline
    # -- pushing overall present credit from 0.550 to 0.787, out of the docx
    # section 5.1 target band (0.30-0.55). Two bounded retune passes were
    # tried, neither weakening verification: (1) Set A noise "default" ->
    # "low" plus Set B severity pushed toward 3-4 -> credit moved the WRONG
    # way, to 0.800; (2) Set A noise -> "medium" (B unchanged from (1)) ->
    # back to 0.7875, no improvement. A crop that survives global
    # verification under added noise tends to still be a strong, unambiguous
    # match -- noise doesn't reliably decouple "hittable" from "easy" once
    # the gate requires global uniqueness, which is exactly the tension the
    # docx names in section 5.1. Per instruction: do not manufacture a
    # deceptively lower baseline by weakening the verification gate or
    # picking deliberately ambiguous crops. Reverted to the original spec
    # below (the one that measures 0.787) and documented as a limitation in
    # REPORT.md section 4 instead.
    _p("A01", "A", "dram_1x", 8.0, -5.0, description="nominal pose lower endpoint"),
    _p("A02", "A", "dram_dense", 12.0, 5.0, description="nominal pose upper endpoint"),
    _p("A03", "A", "dram_loose", 10.0, 0.0, description="nominal pose zero rotation"),
    _p("A04", "A", "dram_wide", 11.5, 4.0,
       description="nominal pose periodic-ambiguity stress"),
    _p("A05", "A", "finfet_10nm", 12.0, 5.0,
       description="nominal pose periodic-ambiguity stress"),
    _p("A06", "A", "finfet_7nm", 10.0, 0.0,
       description="nominal pose periodic-ambiguity stress"),
    _p("A07", "A", "finfet_14nm", 10.5, 3.0),
    _p("A08", "A", "dram_wide", 11.5, 4.0,
       description="nominal pose periodic-ambiguity stress"),
    _p("B01", "B", "dram_compact", 8.0, -4.0, severity_level=1,
       noise="randomized", description="Set-B severity 1"),
    _p("B02", "B", "dram_legacy", 9.0, -1.0, severity_level=2,
       noise="randomized", description="Set-B severity 2"),
    _p("B03", "B", "finfet_28nm", 10.0, 0.0, severity_level=2,
       noise="randomized", description="Set-B severity 2"),
    _p("B04", "B", "finfet_45nm", 11.0, 1.0, severity_level=3,
       noise="randomized", description="Set-B severity 3"),
    _p("B05", "B", "finfet_28nm", 11.5, 4.0, severity_level=3,
       noise="randomized", description="Set-B severity 3"),
    _p("B06", "B", "finfet_45nm", 12.0, 5.0, severity_level=4,
       noise="randomized", description="Set-B severity 4"),
    _p("C01", "C", "dram_dense", 8.0, -5.0, present=False),
    _p("C02", "C", "dram_wide", 10.0, 0.0, present=False),
    _p("C03", "C", "finfet_14nm", 11.0, 3.0, present=False),
    _p("C04", "C", "finfet_22nm", 12.0, 5.0, present=False),
    _p("D01", "D", "dram_loose", 10.0, 0.0, edge_brightening=0.30,
       description="optical edge response"),
    _p("D02", "D", "finfet_7nm", 9.5, -2.0, edge_brightening=0.40,
       description="optical edge response"),
)


def validate_audit_specs(specs: tuple[PairSpec, ...] = AUDIT_SPECS) -> None:
    if len(specs) != 20:
        raise ValueError(f"audit must contain 20 pairs, got {len(specs)}")
    counts = {name: sum(s.set_name == name for s in specs) for name in ALL_SETS}
    if counts != {"A": 8, "B": 6, "C": 4, "D": 2}:
        raise ValueError(f"wrong set composition: {counts}")
    if sum(s.present for s in specs) != 16 or sum(not s.present for s in specs) != 4:
        raise ValueError("audit must contain exactly 16 present and 4 absent pairs")
    if set(s.preset for s in specs) != set(PRESETS):
        raise ValueError("audit must cover every DRAM and FinFET preset")
    if min(s.z for s in specs) != 8.0 or max(s.z for s in specs) != 12.0:
        raise ValueError("audit must cover z=8 and z=12")
    if min(s.theta for s in specs) != -5.0 or max(s.theta for s in specs) != 5.0:
        raise ValueError("audit must cover theta=-5 and theta=+5")
    if not any(s.theta == 0.0 for s in specs):
        raise ValueError("audit must contain theta=0")
    if any(s.set_name == "C" and s.present for s in specs):
        raise ValueError("Set C must be absent")
    if any(s.set_name in PRESENT_SETS and not s.present for s in specs):
        raise ValueError("Sets A, B and D must be present")


# A pinned severity LEVEL is naturally a single point (lo == hi), but
# driftsense.generate.build_one only engages sample_severity_params -- the
# function that actually draws the coherent per-knob degradation -- when
# `_shi > _slo`. A degenerate (lo == hi) range fails that strictly-greater
# gate, so a pair pinned this way silently falls back to independent
# per-knob draws (noise="randomized") instead of the requested severity
# level: it is labelled "Set-B severity N" but never actually realizes it
# (issue #31). Widening the top by SEVERITY_PIN_EPSILON keeps `_shi > _slo`
# true while leaving the realized severity indistinguishable from the pinned
# level -- sample_severity_params's own +/-8% per-knob jitter
# (SEVERITY_MARGIN) is four orders of magnitude larger than this epsilon.
SEVERITY_PIN_EPSILON = 1e-6


def pose_for(spec: PairSpec) -> PoseSpec:
    polygon = (-0.2, 0.2) if spec.set_name == "B" else (0.0, 0.0)
    target = spec.severity_level / 4.0
    severity = (target, target + SEVERITY_PIN_EPSILON) \
        if spec.set_name == "B" else (0.0, 0.0)
    return PoseSpec(
        rotation_deg=(spec.theta, spec.theta),
        magnification=(spec.z, spec.z),
        edge_brightening=(spec.edge_brightening, spec.edge_brightening),
        absent_frac=0.0 if spec.present else 1.0,
        polygon_scale=polygon,
        severity=severity,
    )


def _seed_for(seed: int, index: int, attempt: int = 0) -> int:
    # attempt=0 must reproduce byte-identically what every pair used before
    # retries existed -- SeedSequence hashes the WHOLE entropy tuple, so
    # appending attempt=0 unconditionally would reseed every pair, not just
    # a failing one. Only a retry (attempt>0) extends the entropy.
    entropy = [int(seed), int(index), 0x45]
    if attempt:
        entropy.append(int(attempt))
    state = np.random.SeedSequence(entropy)
    return int(state.generate_state(1, dtype=np.uint64)[0])


# Docx spec section 5: "If a crop fails, resample a different crop location
# and retry. Cap the retries, and if nothing passes, fail loudly." The AMP
# reference material's own generator used up to 14 attempts; this audit is a
# fixed 20-pair set rather than a bulk run, so a smaller cap is plenty.
MAX_VERIFY_ATTEMPTS = 32


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"failed to write {path}")


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _prepare_output(root: Path, force: bool) -> None:
    root = root.resolve()
    if root == Path(root.anchor) or root == REPO_ROOT.resolve():
        raise ValueError(f"refusing to use broad output path: {root}")
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(f"output is non-empty; use --force explicitly: {root}")
    if root.exists() and force:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "reference").mkdir(exist_ok=True)
    (root / "search").mkdir(exist_ok=True)


def _manifest_row(spec: PairSpec, result: dict, pair_seed: int,
                  verify_attempts: int) -> dict:
    return {
        "pair_id": spec.pair_id,
        "set_name": spec.set_name,
        "present": int(spec.present),
        "preset": spec.preset,
        "family": spec.family,
        "architecture": result["architecture"],
        "reference_path": f"reference/{spec.pair_id}.png",
        "search_path": f"search/{spec.pair_id}.png",
        "gt_x": float(result["gt_x"]),
        "gt_y": float(result["gt_y"]),
        "gt_x_raw": float(result["gt_x_raw"]),
        "gt_y_raw": float(result["gt_y_raw"]),
        "label_shift_px": float(result["label_shift_px"]),
        "magnification": float(result["magnification"]),
        "rotation_deg": float(result["rotation_deg"]),
        "severity_level": spec.severity_level,
        "severity_continuous": float(result.get("severity_continuous", 0.0)),
        "edge_brightening": spec.edge_brightening,
        "noise_profile": spec.noise,
        "sample_seed": pair_seed,
        # 1 unless the label-verification gate below needed a resample; a
        # value >1 here is a legitimate part of the record, not a defect --
        # docx section 5 requires exactly this resample-until-verified
        # behaviour and asks that it be reported.
        "verify_attempts": verify_attempts,
        "description": spec.description,
    }


def generate_audit(output_dir: str | os.PathLike[str] = OUTPUT_DEFAULT,
                   seed: int = SEED_DEFAULT, force: bool = False) -> dict:
    """Generate the exact audit package and return its manifest rows."""
    validate_audit_specs()
    root = Path(output_dir)
    _prepare_output(root, force)
    started = time.perf_counter()
    tracemalloc.start()
    rows: list[dict] = []
    for index, spec in enumerate(AUDIT_SPECS):
        ref_path = root / "reference" / f"{spec.pair_id}.png"
        sea_path = root / "search" / f"{spec.pair_id}.png"
        last_verify_error = None
        for attempt in range(MAX_VERIFY_ATTEMPTS):
            pair_seed = _seed_for(seed, index, attempt)
            results = make_pairs(
                pair_seed, [spec.preset], spec.noise, crops=1,
                pose=pose_for(spec), preset_name=spec.preset
            )
            if len(results) != 1:
                raise RuntimeError(f"{spec.pair_id}: expected one generated pair")
            result = results[0]
            if bool(result["found"]) != spec.present:
                raise RuntimeError(f"{spec.pair_id}: presence mismatch")
            if spec.set_name == "B":
                # Realized-parameter check (issue #31): confirm the severity
                # ladder actually fired for this pair rather than silently
                # falling back to independent per-knob draws. A generic
                # "randomized" pair's severity_continuous defaults to 0.0
                # (GenerationParams' own default), so this also catches the
                # degenerate-tuple regression the epsilon above exists to
                # avoid.
                target = spec.severity_level / 4.0
                realized = float(result.get("severity_continuous", 0.0))
                if abs(realized - target) > 10 * SEVERITY_PIN_EPSILON:
                    raise RuntimeError(
                        f"{spec.pair_id}: severity ladder did not pin to "
                        f"level {spec.severity_level} (target {target:.6f}, "
                        f"realized {realized:.6f}) -- the ladder likely did "
                        "not fire")

            # Docx section 5: gate the artifact that ships, not an
            # intermediate -- write to disk first, then read both PNGs back
            # and verify against those, exactly like run_score's own
            # verification does. A pair whose label the global peak cannot
            # reproduce within 3 px / margin 0.02 is resampled with a fresh
            # seed rather than shipped; only present pairs have a target to
            # verify.
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
            if primary["error_px"] <= 3.0 and primary["margin"] >= 0.02 \
                    and independent["error_px"] <= 3.0:
                break
            last_verify_error = (
                f"primary error={primary['error_px']:.3f}px "
                f"margin={primary['margin']:.4f}, "
                f"independent error={independent['error_px']:.3f}px "
                f"(attempt {attempt + 1}/{MAX_VERIFY_ATTEMPTS})")
        else:
            raise RuntimeError(
                f"{spec.pair_id}: label failed verification after "
                f"{MAX_VERIFY_ATTEMPTS} resamples -- last attempt: "
                f"{last_verify_error}. Never shipping an unverified label "
                "(docx section 5).")

        rows.append(_manifest_row(spec, result, pair_seed, attempt + 1))

    fields = list(rows[0])
    _write_csv(root / "manifest.csv", rows, fields)
    _write_csv(root / "pairs.csv", [
        {"pair_id": r["pair_id"], "set_name": r["set_name"],
         "reference_path": r["reference_path"], "search_path": r["search_path"]}
        for r in rows
    ], ["pair_id", "set_name", "reference_path", "search_path"])
    _write_csv(root / "ground_truth.csv", [
        {"pair_id": r["pair_id"], "present": r["present"],
         "x": r["gt_x"] if r["present"] else 0.0,
         "y": r["gt_y"] if r["present"] else 0.0,
         "theta": r["rotation_deg"] if r["present"] else 0.0,
         "scale": r["magnification"] if r["present"] else 0.0}
        for r in rows
    ], ["pair_id", "present", "x", "y", "theta", "scale"])
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    meta = {
        "seed": seed,
        "pair_count": len(rows),
        "present_count": sum(r["present"] for r in rows),
        "absent_count": sum(not r["present"] for r in rows),
        "sets": {name: sum(r["set_name"] == name for r in rows) for name in ALL_SETS},
        "presets": sorted({r["preset"] for r in rows}),
        "runtime_seconds": time.perf_counter() - started,
        "python_peak_tracemalloc_mb": peak / (1024 * 1024),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "renderer": "driftsense.generate -> generator/src",
        "organizer_data_used_for_training_or_tuning": False,
    }
    (root / "generation_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"root": str(root), "rows": rows, "meta": meta}


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def make_template(reference: np.ndarray, z: float, theta: float) -> np.ndarray:
    """Official-style area template reduction plus one affine pose step."""
    ref = _gray(reference)
    h, w = ref.shape[:2]
    fh, fw = h / float(z), w / float(z)
    th, tw = max(int(np.floor(fh)), 1), max(int(np.floor(fw)), 1)
    ah, aw = max(int(np.ceil(fh)), 1), max(int(np.ceil(fw)), 1)
    base = cv2.resize(ref, (aw, ah), interpolation=cv2.INTER_AREA)
    residual = fw / aw
    if theta == 0.0 and abs(residual - 1.0) < 1e-9 and (aw, ah) == (tw, th):
        return base
    matrix = cv2.getRotationMatrix2D(((aw - 1) / 2.0, (ah - 1) / 2.0), theta, residual)
    matrix[0, 2] += (tw - 1) / 2.0 - (aw - 1) / 2.0
    matrix[1, 2] += (th - 1) / 2.0 - (ah - 1) / 2.0
    return cv2.warpAffine(base, matrix, (tw, th), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _grid(lo: float, hi: float, step: float) -> tuple[float, ...]:
    count = int(round((hi - lo) / step))
    return tuple(round(lo + i * step, 6) for i in range(count + 1))


Z_GRID = _grid(8.0, 12.0, 0.5)
THETA_GRID = _grid(-5.0, 5.0, 1.0)


def baseline_match(reference: np.ndarray, search: np.ndarray) -> dict:
    """Run the official-style coarse NCC baseline over z/theta grids."""
    ref = _gray(reference)
    sea = _gray(search)
    best: dict | None = None
    for z in Z_GRID:
        for theta in THETA_GRID:
            template = make_template(ref, z, theta)
            if template.shape[0] >= sea.shape[0] or template.shape[1] >= sea.shape[1]:
                continue
            response = cv2.matchTemplate(sea, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, location = cv2.minMaxLoc(response)
            candidate = {
                "x": location[0] + template.shape[1] / 2.0,
                "y": location[1] + template.shape[0] / 2.0,
                "score": float(score),
                "z_hat": z,
                "theta_hat": theta,
                "template_width": int(template.shape[1]),
                "template_height": int(template.shape[0]),
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate
    if best is None:
        raise RuntimeError("baseline had no valid template hypothesis")
    return best


def _feature(image: np.ndarray, mode: str) -> np.ndarray:
    gray = _gray(image).astype(np.float32)
    if mode == "raw":
        return gray
    if mode == "gradient":
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)
    raise ValueError(f"unknown verifier feature: {mode}")


def _global_verify(reference: np.ndarray, search: np.ndarray, gt_x: float,
                   gt_y: float, z: float, theta: float, mode: str = "raw") -> dict:
    """Primary label verification: the GLOBAL correlation peak over the
    entire search frame at the fixed labelled pose (docx section 5: "The
    global correlation peak must land within 3 px of your label").

    On a periodic layout a crop can look perfect within any local window
    while a stronger repeat elsewhere in the full 1000x1000 frame wins
    globally -- exactly the failure mode section 5 exists to catch, and a
    windowed search cannot see it by construction. Margin is measured
    against the best competing peak outside a small exclusion window around
    the winner, over the SAME full-frame response the winner came from.
    """
    template = _feature(make_template(reference, z, theta), mode)
    frame = _feature(search, mode)
    tw, th = template.shape[1], template.shape[0]
    response = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, location = cv2.minMaxLoc(response)
    px = location[0] + tw / 2.0
    py = location[1] + th / 2.0
    error = float(np.hypot(px - gt_x, py - gt_y))
    suppressed = response.copy()
    cv2.circle(suppressed, location, 9, -np.inf, -1)
    runner = float(np.max(suppressed)) if suppressed.size else float("-inf")
    margin = float(peak - runner) if np.isfinite(runner) else float(peak)
    return {
        "x": float(px), "y": float(py), "score": float(peak),
        "error_px": error, "margin": margin, "mode": mode,
    }


def _local_verify(reference: np.ndarray, search: np.ndarray, gt_x: float,
                  gt_y: float, z: float, theta: float, mode: str) -> dict:
    """Independent cross-check: a deliberately different template/feature
    path (gradient magnitude, not raw intensity) confirmed near the already
    globally-verified label, per docx section 5's "cross-check with a
    second, deliberately different verifier". Windowed, not global -- it is
    corroborating _global_verify's answer, not independently establishing
    global uniqueness, which is _global_verify's job.
    """
    template = _feature(make_template(reference, z, theta), mode)
    frame = _feature(search, mode)
    tw, th = template.shape[1], template.shape[0]
    expected_x = int(round(gt_x - tw / 2.0))
    expected_y = int(round(gt_y - th / 2.0))
    radius = 14
    left = max(0, expected_x - radius)
    top = max(0, expected_y - radius)
    right = min(frame.shape[1] - tw, expected_x + radius)
    bottom = min(frame.shape[0] - th, expected_y + radius)
    if right < left or bottom < top:
        raise ValueError("ground-truth verification window is outside image")
    region = frame[top:bottom + th, left:right + tw]
    response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, location = cv2.minMaxLoc(response)
    px = left + location[0] + tw / 2.0
    py = top + location[1] + th / 2.0
    error = float(np.hypot(px - gt_x, py - gt_y))
    # Suppress a small neighbourhood around the winner before measuring the
    # runner-up. This is a local distinctiveness margin, not a global periodic
    # uniqueness claim -- _global_verify (the primary check) is what
    # establishes global uniqueness; this is corroboration only.
    suppressed = response.copy()
    cv2.circle(suppressed, location, 9, -np.inf, -1)
    runner = float(np.max(suppressed)) if suppressed.size else float("-inf")
    margin = float(peak - runner) if np.isfinite(runner) else float(peak)
    return {
        "x": float(px), "y": float(py), "score": float(peak),
        "error_px": error, "margin": margin, "mode": mode,
    }


def _load_rows(root: Path) -> list[dict]:
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_pair(root: Path, row: dict) -> tuple[np.ndarray, np.ndarray]:
    reference = cv2.imread(str(root / row["reference_path"]), cv2.IMREAD_UNCHANGED)
    search = cv2.imread(str(root / row["search_path"]), cv2.IMREAD_UNCHANGED)
    if reference is None or search is None:
        raise FileNotFoundError(f"unreadable pair {row['pair_id']}")
    return reference, search


def _credit(error: float) -> float:
    if error <= 1.0:
        return 1.0
    if error <= 2.0:
        return 0.8
    if error <= 3.0:
        return 0.6
    if error <= 5.0:
        return 0.4
    return 0.0


def _precision_recall_f1(rows: list[dict], threshold: float) -> dict:
    tp = sum(r["present"] and r["predicted_present"] for r in rows)
    fp = sum((not r["present"]) and r["predicted_present"] for r in rows)
    fn = sum(r["present"] and (not r["predicted_present"]) for r in rows)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _spectral_fraction(image: np.ndarray) -> float:
    values = image.astype(np.float64) - float(np.mean(image))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(values))) ** 2
    h, w = image.shape[:2]
    yy, xx = np.mgrid[-0.5:0.5:complex(h), -0.5:0.5:complex(w)]
    high = spectrum[np.sqrt(xx * xx + yy * yy) > 0.25].sum()
    return float(high / max(float(spectrum.sum()), 1e-12))


def _resampling_field(size: int = 4096) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    x = xx / size
    y = yy / size
    field = 35.0 + 70.0 * (np.sin(2 * np.pi * (x * 21.0 + y * 0.7)) > 0)
    field += 85.0 * (np.sin(2 * np.pi * (y * 17.0 - x * 0.4)) > 0)
    field += 45.0 * (((xx % 173) < 24) & ((yy % 137) < 18))
    return np.clip(field, 0, 255).astype(np.uint8)


def _resampling_case(z: float, theta: float) -> dict:
    source = _resampling_field()
    target = 256
    production_matrix = search_affine(source.shape[0], target, z, theta)
    production = cv2.GaussianBlur(source, (0, 0), max(z / 2.0, 0.5),
                                   borderType=cv2.BORDER_REPLICATE)
    production = cv2.warpAffine(production, production_matrix, (target, target),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    # Independent 2x supersampled path. It uses a different source sampling
    # grid and is intentionally not a call to image_search_traced.
    high = cv2.resize(source, (source.shape[1] * 2, source.shape[0] * 2),
                      interpolation=cv2.INTER_CUBIC)
    high = cv2.GaussianBlur(high, (0, 0), max(z, 1.0),
                            borderType=cv2.BORDER_REPLICATE)
    truth_matrix = search_affine(high.shape[0], target, z * 2.0, theta)
    truth = cv2.warpAffine(high, truth_matrix, (target, target),
                           flags=cv2.INTER_AREA,
                           borderMode=cv2.BORDER_REPLICATE)
    no_aa = cv2.warpAffine(source, production_matrix, (target, target),
                           flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REPLICATE)

    def measure(image: np.ndarray) -> tuple[float, float, float]:
        error = image.astype(np.float32) - truth.astype(np.float32)
        mae = float(np.mean(np.abs(error)))
        mse = float(np.mean(error * error))
        psnr = float(20 * np.log10(255.0 / math.sqrt(max(mse, 1e-12))))
        return mae, psnr, _spectral_fraction(image)

    pm, pp, ps = measure(production)
    cm, cp, cs = measure(no_aa)
    return {
        "z": z, "theta": theta,
        "production_mae": pm, "production_psnr": pp,
        "control_mae": cm, "control_psnr": cp,
        "production_spectral": ps, "control_spectral": cs,
        "truth_spectral": _spectral_fraction(truth),
        "production_better": bool(pm < cm and pp > cp),
    }


def resampling_audit() -> list[dict]:
    return [_resampling_case(12.0, 5.0), _resampling_case(11.5, 2.7)]


def _geometry_metrics(rows: list[dict]) -> dict:
    round_trip = []
    source_lows, source_highs = [], []
    footprint_ok = True
    label_shifts = []
    for row in rows:
        z = float(row["magnification"])
        theta = float(row["rotation_deg"])
        canvas_px = PoseSpec(rotation_deg=(theta, theta),
                             magnification=(z, z)).required_canvas_px()
        matrix = search_affine(canvas_px, SEARCH_SIZE_PX, z, theta)
        inverse = cv2.invertAffineTransform(matrix)
        corners = np.array([[0, 0], [999, 0], [0, 999], [999, 999]], dtype=np.float64)
        mapped = np.array([apply_affine_point(inverse, x, y) for x, y in corners])
        source_lows.append(mapped.min(axis=0))
        source_highs.append(mapped.max(axis=0))
        # Round-trip of the affine's source corners is the numerical R1 proof.
        source_points = np.array([[0.0, 0.0], [1234.5, 9876.25],
                                  [canvas_px - 1.0, canvas_px - 1.0]])
        forward = np.array([apply_affine_point(matrix, x, y) for x, y in source_points])
        back = np.array([apply_affine_point(inverse, x, y) for x, y in forward])
        round_trip.append(float(np.max(np.linalg.norm(back - source_points, axis=1))))
        if int(row["present"]):
            x, y = float(row["gt_x"]), float(row["gt_y"])
            half = 0.5 * (1000.0 / z) * (
                abs(math.cos(math.radians(theta))) + abs(math.sin(math.radians(theta)))
            )
            footprint_ok = footprint_ok and half <= x <= 1000.0 - half \
                and half <= y <= 1000.0 - half
        label_shifts.append(float(row.get("label_shift_px", 0.0) or 0.0))
    return {
        "R1_max_round_trip_px": max(round_trip),
        "R2_pose_endpoints": {
            "z_min": min(float(r["magnification"]) for r in rows),
            "z_max": max(float(r["magnification"]) for r in rows),
            "theta_min": min(float(r["rotation_deg"]) for r in rows),
            "theta_max": max(float(r["rotation_deg"]) for r in rows),
            "theta_zero_present": any(float(r["rotation_deg"]) == 0.0 for r in rows),
        },
        "R3_source_bounds": [
            np.min(np.asarray(source_lows), axis=0).tolist(),
            np.max(np.asarray(source_highs), axis=0).tolist(),
        ],
        "R4_visible_present_footprints": footprint_ok,
        "R5_max_label_shift_px": max(label_shifts) if label_shifts else 0.0,
    }


def run_baseline(output_dir: str | os.PathLike[str] = OUTPUT_DEFAULT,
                 threshold: float = 0.55) -> dict:
    root = Path(output_dir)
    rows = _load_rows(root)
    calibrated = []
    for row in rows:
        reference, search = _read_pair(root, row)
        match = baseline_match(reference, search)
        present = bool(int(row["present"]))
        error = (float(np.hypot(match["x"] - float(row["gt_x"]),
                                match["y"] - float(row["gt_y"])))
                 if present else None)
        calibrated.append({
            "pair_id": row["pair_id"], "set_name": row["set_name"],
            "severity_level": int(row["severity_level"]), "present": present,
            "score": match["score"], "predicted_present": match["score"] >= threshold,
            "error_px": error,
            "credit": (_credit(error) if match["score"] >= threshold else 0.0)
            if present else 0.0,
            "z_hat": match["z_hat"], "theta_hat": match["theta_hat"],
            "x_hat": match["x"], "y_hat": match["y"],
        })
    _write_calibration(root, calibrated)
    return {"threshold": threshold, "rows": calibrated}


def _write_calibration(root: Path, rows: list[dict]) -> None:
    with (root / "baseline_calibration.txt").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _similarity_audit(rows: list[dict], calibrated: list[dict]) -> dict:
    absent = [r for r in calibrated if not r["present"]]
    same_family = all(row["family"] == PRESETS[row["preset"]]["kind"]
                      for row in rows if row["set_name"] == "C")
    scores = [r["score"] for r in absent]
    return {
        "same_family_decoys": same_family,
        "absent_count": len(absent),
        "absent_score_min": min(scores) if scores else None,
        "absent_score_max": max(scores) if scores else None,
        "semantic_absence_flagged_in_manifest": all(not int(r["present"])
                                                      for r in rows if r["set_name"] == "C"),
        "note": "Set C uses the renderer's independent same-family decoy canvas; global NCC score is reported as difficulty evidence, not as an absence proof.",
    }


def run_score(output_dir: str | os.PathLike[str] = OUTPUT_DEFAULT,
              threshold: float = 0.55, write_report: bool = True) -> dict:
    root = Path(output_dir)
    rows = _load_rows(root)
    baseline = run_baseline(root, threshold)["rows"]
    by_id = {r["pair_id"]: r for r in baseline}
    verification = []
    for row in rows:
        if not int(row["present"]):
            continue
        reference, search = _read_pair(root, row)
        primary = _global_verify(reference, search, float(row["gt_x"]), float(row["gt_y"]),
                                 float(row["magnification"]), float(row["rotation_deg"]), "raw")
        independent = _local_verify(reference, search, float(row["gt_x"]), float(row["gt_y"]),
                                    float(row["magnification"]), float(row["rotation_deg"]), "gradient")
        verification.append({
            "pair_id": row["pair_id"], "primary": primary,
            "independent": independent,
            "pass": primary["error_px"] <= 3.0 and primary["margin"] >= 0.02
            and independent["error_px"] <= 3.0,
        })
    present = [r for r in baseline if r["present"]]
    per_set = {}
    for name in ALL_SETS:
        subset = [r for r in present if r["set_name"] == name]
        per_set[name] = {
            "count": len(subset),
            "mean_credit": float(np.mean([r["credit"] for r in subset])) if subset else None,
            "median_error_px": float(np.median([r["error_px"] for r in subset])) if subset else None,
        }
    severity = {}
    for level in (1, 2, 3, 4):
        errors = [r["error_px"] for r in baseline
                  if r["set_name"] == "B" and r["severity_level"] == level and r["error_px"] is not None]
        severity[str(level)] = float(np.median(errors)) if errors else None
    severity_values = [v for v in severity.values() if v is not None]
    severity_monotone = all(a < b for a, b in zip(severity_values, severity_values[1:]))
    overall_credit = float(np.mean([r["credit"] for r in present]))
    metrics = {
        "baseline_threshold": threshold,
        "baseline_rows": baseline,
        "present_score_range": [min(r["score"] for r in present), max(r["score"] for r in present)],
        "absent_score_range": [min(r["score"] for r in baseline if not r["present"]),
                               max(r["score"] for r in baseline if not r["present"])],
        "per_set": per_set,
        "severity_median_error_px": severity,
        "severity_strictly_monotone": severity_monotone,
        "overall_present_credit": overall_credit,
        "target_band_0_30_to_0_55": 0.30 <= overall_credit <= 0.55,
        "classification": _precision_recall_f1(baseline, threshold),
        "verification": verification,
        "all_present_verification_pass": all(v["pass"] for v in verification),
        "similarity_audit": _similarity_audit(rows, baseline),
        "geometry": _geometry_metrics(rows),
        "resampling": resampling_audit(),
        "numpy": np.__version__, "opencv": cv2.__version__,
    }
    (root / "score.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n",
                                        encoding="utf-8")
    if write_report:
        (root / "REPORT.md").write_text(render_report(metrics, rows), encoding="utf-8")
    return metrics


def render_report(metrics: dict, rows: list[dict]) -> str:
    geometry = metrics["geometry"]
    checks = {
        "exact A8/B6/C4/D2 composition": len(rows) == 20,
        "16 present / 4 absent": sum(int(r["present"]) for r in rows) == 16,
        "all present pairs pass both verifiers": metrics["all_present_verification_pass"],
        "Set-B severity medians strictly increase": metrics["severity_strictly_monotone"],
        "resampling production beats no-AA control": all(r["production_better"] for r in metrics["resampling"]),
        "organizer data excluded from tuning": True,
    }
    lines = [
        "# Drift-Sense Phase 2 generator audit",
        "",
        f"Stack: Python runtime, NumPy {metrics['numpy']}, OpenCV {metrics['opencv']}.",
        "The audit delegates scene generation to `driftsense.generate`, which delegates structural rendering to `generator/src`.",
        "",
        "## 1. Transform, labels, and R1–R5",
        "",
        "Reference crops are mapped into the search frame with the same affine used for rendering; raster-drift correction is taken from the realised traced warp. Absent pairs carry `present=0` and no valid target centre.",
        f"- R1 maximum affine round-trip error: {geometry['R1_max_round_trip_px']:.3e} px.",
        f"- R2 pose coverage: z={geometry['R2_pose_endpoints']['z_min']:.1f}..{geometry['R2_pose_endpoints']['z_max']:.1f}, theta={geometry['R2_pose_endpoints']['theta_min']:.1f}..{geometry['R2_pose_endpoints']['theta_max']:.1f} degrees, theta=0 present={geometry['R2_pose_endpoints']['theta_zero_present']}.",
        f"- R3 traced source bounds: {geometry['R3_source_bounds']}.",
        f"- R4 visible present footprints: {geometry['R4_visible_present_footprints']}.",
        f"- R5 maximum corrected label shift: {geometry['R5_max_label_shift_px']:.3f} px.",
        "",
        "## 2. Verification and resampling",
        "",
        f"Primary verification is the GLOBAL correlation peak over the full search frame at the labelled pose, requiring error <=3 px and margin >=0.02 against the best competing peak; independent verification cross-checks near that label with gradient magnitude rather than raw intensity. All present pairs passed: {metrics['all_present_verification_pass']}.",
        "",
        "| pair_id | primary error (px) | primary margin | independent error (px) | pass |",
        "|---|---:|---:|---:|:---:|",
    ]
    for v in metrics["verification"]:
        lines.append(
            f"| {v['pair_id']} | {v['primary']['error_px']:.3f} | "
            f"{v['primary']['margin']:.4f} | {v['independent']['error_px']:.3f} | "
            f"{'PASS' if v['pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "Resampling compares the production blurred affine path with an independent 2x supersampled path and a nearest-neighbour no-antialiasing control:",
    ]
    for result in metrics["resampling"]:
        lines.append(
            f"- z={result['z']:.1f}, theta={result['theta']:+.1f}: production MAE={result['production_mae']:.3f}, PSNR={result['production_psnr']:.3f}; no-AA MAE={result['control_mae']:.3f}, PSNR={result['control_psnr']:.3f}; production_better={result['production_better']}."
        )
    lines += [
        "",
        "## 3. Baseline calibration",
        "",
        f"Official-style coarse NCC searches z in [8,12] by 0.5 and theta in [-5,5] by 1 degree at threshold {metrics['baseline_threshold']:.2f}.",
        f"Per-set results: {json.dumps(metrics['per_set'], sort_keys=True)}.",
        f"Present score range={metrics['present_score_range']}; absent score range={metrics['absent_score_range']}; classification={json.dumps(metrics['classification'], sort_keys=True)}.",
        f"Set-B severity median errors={json.dumps(metrics['severity_median_error_px'], sort_keys=True)}; strictly monotone={metrics['severity_strictly_monotone']}.",
        f"Overall present credit={metrics['overall_present_credit']:.3f}; target band status={metrics['target_band_0_30_to_0_55']}.",
        "",
        "## 4. Set-C design and limitations",
        "",
        "Set C references are generated from an independent same-family decoy canvas with the renderer's pitch-offset rule; the search canvas is separate, so no true reference instance is inserted. The similarity audit reports global NCC scores as difficulty evidence and retains the semantic absence flag as the actual label contract.",
        "The procedural DRAM/FinFET patterns are illustrative rather than proprietary fab geometry. The independent resampling truth is a supersampled validation field, not a metrology instrument. The NCC baseline remains vulnerable to periodic repeats; its score range is reported rather than hidden.",
        "",
        "**Calibration band limitation.** Global post-write verification (the "
        "GLOBAL correlation peak over the full search frame, per section 5) "
        "exposed periodic ambiguities a windowed local verifier had been "
        "hiding: enforcing the required global-peak gate raised naive-"
        "baseline present credit from 0.550 to 0.787, above the section "
        "5.1 target band (0.30-0.55). Two bounded retune passes (Set A "
        "noise floor raised, Set B severity pushed toward levels 3-4) were "
        "tried and neither brought the set back into band -- a crop that "
        "survives global verification under added noise tends to still be "
        "a strong, unambiguous match, so noise does not reliably decouple "
        "\"hittable\" from \"easy\" once the gate requires global "
        "uniqueness. We retained globally verifiable labels rather than "
        "weakening the gate, or selecting deliberately ambiguous crops, to "
        "force the calibration target. Severity remains the "
        "presence-detection/difficulty lever it is used as elsewhere in "
        "this report; the remaining calibration deviation is reported here "
        "as a known limitation, not resolved.",
        "",
        "## 5. Acceptance snapshot",
        "",
    ]
    lines.extend(f"- [{'x' if value else ' '}] {label}" for label, value in checks.items())
    lines += [
        "",
        "Organizer reference/sample material was not used for training, fine-tuning, threshold fitting, or generator tuning.",
        "",
    ]
    return "\n".join(lines)


def make_contact_sheet(output_dir: str | os.PathLike[str] = OUTPUT_DEFAULT) -> Path:
    root = Path(output_dir)
    rows = _load_rows(root)
    tile_w, tile_h = 320, 300
    sheet = np.full((4 * tile_h, 5 * tile_w, 3), 255, dtype=np.uint8)
    for index, row in enumerate(rows):
        reference, search = _read_pair(root, row)
        thumb = cv2.resize(_gray(search), (260, 240), interpolation=cv2.INTER_AREA)
        tile = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        if int(row["present"]):
            x = int(round(float(row["gt_x"]) * 260.0 / 1000.0))
            y = int(round(float(row["gt_y"]) * 240.0 / 1000.0))
            cv2.circle(tile, (x, y), 7, (0, 0, 255), 2)
        else:
            cv2.putText(tile, "ABSENT", (90, 125), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 255), 2, cv2.LINE_AA)
        ref_thumb = cv2.resize(_gray(reference), (64, 64), interpolation=cv2.INTER_AREA)
        tile[4:68, 4:68] = cv2.cvtColor(ref_thumb, cv2.COLOR_GRAY2BGR)
        x0, y0 = (index % 5) * tile_w, (index // 5) * tile_h
        sheet[y0:y0 + 240, x0:x0 + 260] = tile
        label = f"{row['pair_id']} Set {row['set_name']} z={float(row['magnification']):.1f} th={float(row['rotation_deg']):+.1f}"
        cv2.putText(sheet, label, (x0 + 4, y0 + 263), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (0, 0, 0), 1, cv2.LINE_AA)
    path = root / "contact_sheet.png"
    _write_png(path, sheet)
    return path


def add_common_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))


def parse_output_dir(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_common_cli_arguments(parser)
    return parser.parse_args(argv)

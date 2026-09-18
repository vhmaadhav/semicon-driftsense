#!/usr/bin/env python3
"""Phase 3 entry point: GDS reference -> predictions.

Same command shape as ``register.py`` (``--input pairs.csv --output
predictions.csv``), same output contract, same scoring. The single difference
is what the reference *is*: a GDSII design file rather than an SEM image.

    python phase3.py --input pairs.csv --output predictions.csv

``register.py`` is deliberately left untouched. Everything from Phase 1 and
Phase 2 still applies, and the Phase 2 entry point remains the graded one until
Phase 3 actually opens.

What this reuses, rather than reimplements
------------------------------------------

The hard-won Phase 2 properties are all inherited by calling ``register``'s own
code:

* per-pair zero-fill on decline (a missing row scores zero; a declined one does
  not, so every pair always gets a row),
* one row per ``pair_id``, in input order,
* fail-closed weight loading (never silently degrade to the classical matcher),
* the 4-core thread cap,
* the mass-failure alarm -- a systematic failure must not produce a
  well-formed, exit-0, all-declined CSV. That alarm was written for exactly the
  failure shape a schema mistake produces, and the Phase 3 header is the most
  likely way to trigger it.

What is genuinely new
---------------------

* ``driftsense.pairs3`` -- explicit, asserted column resolution. The Phase 3
  header has TWO columns matching "reference" (``reference_gds_path`` and
  ``reference_sem_path``), and ``register.pick_column``'s substring fallback
  would silently pick the GDS one and hand it to ``cv2.imread``.
* ``driftsense.gds`` -- the ``.gds`` -> raster step.
* A refusal to run at all if the input is not a Phase 3 pairs file, so the
  schema error is loud and happens before any row is written.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

# register.py must be importable for its shared helpers. It lives beside this
# file and puts its own directory on sys.path, so an explicit insert here keeps
# `python phase3.py` working from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import register as R  # noqa: E402
from driftsense import cad_anchor  # noqa: E402
from driftsense import gds  # noqa: E402
from driftsense import pairs3  # noqa: E402
from driftsense.config import (  # noqa: E402
    PHASE3_COARSE_ROTATIONS, PHASE3_FALLBACK_SHEAR_PX, PHASE3_LABEL_CONVENTION, PHASE3_ROTATION_BOUNDS,
    PHASE3_SUBPIXEL_ROWS, PHASE3_THRESHOLD,
    SHIPPED_BAND, SHIPPED_STRIP_ROTATION, SHIPPED_VERIFICATION,
)
from driftsense.matching import PHASE2_SCALE_BOUNDS, locate_phase2  # noqa: E402

import infer as I  # noqa: E402

OUT_FIELDS = R.OUT_FIELDS
DEFAULT_FOUND_THRESHOLD = PHASE3_THRESHOLD

# Mass-failure thresholds. register.py grew these on the private development
# trunk; this branch's base (origin/main) predates them, and the two trunks
# have diverged. Read them from register when present so the two entry points
# cannot drift apart, and fall back to the same documented values when they are
# absent rather than importing something that does not exist.
MASS_FAILURE_ERROR_FRAC = getattr(R, "MASS_FAILURE_ERROR_FRAC", 0.20)
MASS_FAILURE_FOUND_FRAC = getattr(R, "MASS_FAILURE_FOUND_FRAC", 0.30)
MASS_FAILURE_MIN_PAIRS = getattr(R, "MASS_FAILURE_MIN_PAIRS", 8)

# Phase 3 discloses a different absent rate from Phase 2: about one site in
# twelve has no true match (~8.3%), so ~92% present, versus Phase 2's ~80%.
# The inherited FOUND_FRAC of 0.30 is far below either bound and stays valid.
PHASE3_EXPECTED_PRESENT_FRAC = 0.92


# The Phase 3 decode, as keyword arguments to locate_phase2. ONE definition:
# main() and the evaluation harness (scripts/phase3_eval.py) both go through
# decode(), so a measured configuration is the shipped one.
DECODE = dict(
    refine=True,
    verification=SHIPPED_VERIFICATION,
    band=SHIPPED_BAND,
    subpixel_rows=PHASE3_SUBPIXEL_ROWS,
    strip_rot=SHIPPED_STRIP_ROTATION,
    label_convention=PHASE3_LABEL_CONVENTION,
    scale_bounds=PHASE2_SCALE_BOUNDS,
    rotation_bounds=PHASE3_ROTATION_BOUNDS,
    coarse_rotations=PHASE3_COARSE_ROTATIONS,
)


def decode(model, device, ref, sea, **overrides) -> dict:
    """Pose and confidence for one rendered reference against one search
    frame. Returns locate_phase2's dict; `overrides` replace DECODE entries
    (measurement only -- phase3.py itself passes the defaults)."""
    kw = dict(DECODE)
    kw.update(overrides)
    return locate_phase2(model, ref, sea, device, **kw)


def predict_pair(model, device, ref_gds: str, search_png: str, search_gds: str, *,
                 threshold: float = None, verification: str = SHIPPED_VERIFICATION,
                 render_size: int = gds.REF_SIZE, min_layer: int = 0,
                 use_cad: bool = True, drift_prior_px: float = None, **overrides) -> dict:
    """One pair's raw answer: pose (always filled), found, score, and how.

    Primary path -- the search CAD is on the blind split, so register through
    it (driftsense.cad_anchor): find the reference in the search design
    exactly, then fit the design-to-image rotation over the whole frame.
    `found` there is the CAD-to-CAD decision; `score` its confidence.

    Fallback -- no usable search CAD, or the frame does not align: render the
    reference and run the Phase 2 image matcher (decode()), found = score >=
    threshold, exactly as before.
    """
    if threshold is None:
        threshold = DEFAULT_FOUND_THRESHOLD
    sea = I.read_gray(search_png)
    note = ""
    if use_cad and search_gds:
        try:
            r = cad_anchor.register(ref_gds, search_gds, sea, rotation_bounds=PHASE3_ROTATION_BOUNDS,
                                    ref_size=float(render_size))
            return {"x": r.x, "y": r.y, "theta": r.theta, "scale": r.scale,
                    "found": int(r.found), "score": float(r.score), "method": "cad",
                    "support": r.support, "coarse_peak": r.coarse_peak, "n_interior": r.n_interior,
                    "theta_coarse": r.theta_coarse, "tiles_used": r.tiles_used,
                    "tiles_total": r.tiles_total, "tile_ncc": r.tile_ncc,
                    "tile_resid_px": r.tile_resid_px, "yield_r2": r.yield_r2,
                    "magnification": r.magnification, "note": r.reason}
        except cad_anchor.CadAnchorUnavailable as exc:
            note = f"cad unavailable: {exc}"
    ref = gds.render_reference(ref_gds, size=render_size, min_layer=min_layer)
    if model is None:
        res = I.zncc_fallback(ref, sea)
        thr = R.LEGACY_FALLBACK_THRESHOLD
    else:
        res = decode(model, device, ref, sea, verification=verification, **overrides)
        thr = threshold
    score = float(res.get("confidence", res.get("score", 0.0)))
    # The CAD generator labels the undrifted position; raster shear moves the
    # imaged content left by shear * y / (h - 1) on average, so the matched
    # content sits left of the label by that much. Add the expected shift.
    if drift_prior_px is None:
        drift_prior_px = PHASE3_FALLBACK_SHEAR_PX
    x_img = float(res["x"]) + drift_prior_px * float(res["y"]) / max(sea.shape[0] - 1, 1)
    out = {"x": x_img, "y": float(res["y"]),
           "theta": float(res.get("theta", 0.0)), "scale": float(res.get("scale", 10.0)),
           "found": int(score >= thr), "score": score, "method": "image", "note": note}
    for k, v in res.items():
        if k not in out and isinstance(v, (int, float)):
            out[k] = float(v)
    return out


def _existing(resolved: str, raw: str) -> str:
    """The brief says paths are "relative to the dataset root"; pairs3
    resolves them against the CSV's directory, which is the root when
    pairs.csv sits there. If it does not and the grader runs from the root,
    the working directory is the other reading -- try it before giving up."""
    if not resolved or os.path.exists(resolved):
        return resolved
    raw = (raw or "").strip()
    if raw and not os.path.isabs(raw) and os.path.exists(raw):
        return os.path.abspath(raw)
    return resolved


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Phase 3: register a GDS reference against a search image.")
    ap.add_argument("--input", required=True, help="pairs.csv (Phase 3 layout)")
    ap.add_argument("--output", required=True, help="predictions.csv")
    ap.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_FOUND_THRESHOLD,
                    help="found = score >= threshold (default %(default)s)")
    ap.add_argument("--verification", default=SHIPPED_VERIFICATION)
    ap.add_argument("--threads", type=int, default=0,
                    help="torch/OpenCV thread cap; 0 auto-caps to "
                         "min(4, cores) to match the 4-core reference machine")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="if the learned model cannot load, decode the batch "
                         "with the classical fallback instead of aborting")
    ap.add_argument("--render-size", type=int, default=gds.REF_SIZE,
                    help="reference raster size (default %(default)s)")
    ap.add_argument("--min-layer", type=int, default=0,
                    help="drop design layers below this index when rendering "
                         "the reference")
    ap.add_argument("--no-cad", action="store_true",
                    help="skip the CAD-anchored path and match the rendered reference "
                         "against the image only (measurement / debugging)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    R.cap_threads(a.threads)

    # ---- Read the Phase 3 schema, loudly ---------------------------------
    # Deliberately before anything else that can fail: a schema mistake must
    # abort with an actionable message and NO output file, rather than degrade
    # into per-pair declines. This is the whole reason pairs3 exists.
    try:
        rows = pairs3.read_pairs(a.input, absolute_paths=True)
    except pairs3.PairsSchemaError as exc:
        raise SystemExit(
            f"phase3: {a.input} is not a readable Phase 3 pairs.csv -- {exc}\n"
            "Expected columns: "
            + ", ".join(pairs3.PHASE3_FIELDS)
        ) from exc
    if not rows:
        raise SystemExit(f"{a.input}: no rows")

    model, device = I.load_model(a.weights) or (None, None)
    if model is None and not a.allow_fallback:
        raise SystemExit(
            "FATAL: learned model failed to load from "
            f"{a.weights!r} -- refusing to write {a.output!r} with the "
            "classical fallback (issue #36). Pass --allow-fallback to decode "
            "with the classical fallback instead (local/debug only).")

    out_dir = os.path.dirname(os.path.abspath(a.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    times = []
    found_count = 0
    error_count = 0
    t_start = time.perf_counter()
    total = len(rows)
    mass_failure_warned = False

    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for n, r in enumerate(rows):
            pid = r.pair_id
            out = {"pair_id": pid, "x": 0, "y": 0, "theta": 0, "scale": 0,
                   "found": 0, "score": 0.0}
            t0 = time.perf_counter()
            try:
                # A GdsError or unreadable image here is a per-pair failure,
                # which the Phase 2 contract already handles: the row is
                # written declined.
                src = r.source
                res = predict_pair(model, device,
                                   _existing(r.reference_gds_path, src.get("reference_gds_path")),
                                   _existing(r.search_path, src.get("search_path")),
                                   _existing(r.search_gds_path, src.get("search_gds_path")),
                                   threshold=a.threshold,
                                   verification=a.verification, render_size=a.render_size,
                                   min_layer=a.min_layer, use_cad=not a.no_cad)
                found = int(res["found"])
                out.update({
                    "x": f'{res["x"]:.4f}' if found else 0,
                    "y": f'{res["y"]:.4f}' if found else 0,
                    "theta": f'{res["theta"]:.4f}' if found else 0,
                    "scale": f'{res["scale"]:.4f}' if found else 0,
                    "found": found,
                    "score": f'{res["score"]:.6f}',
                })
            except (Exception, SystemExit) as e:      # noqa: BLE001
                error_count += 1
                print(f"[warn] pair {pid}: {type(e).__name__}: {e}",
                      file=sys.stderr)
            w.writerow(out)
            if out.get("found"):
                found_count += 1
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"# t,{pid},{dt:.3f}", file=sys.stderr, flush=True)

            if (not mass_failure_warned and n + 1 >= MASS_FAILURE_MIN_PAIRS
                    and error_count >= MASS_FAILURE_ERROR_FRAC * (n + 1)):
                mass_failure_warned = True
                print("=" * 72, file=sys.stderr)
                print(f"[MASS FAILURE] {error_count} of the first {n + 1} "
                      "pair(s) raised. This is a systematic failure, not bad "
                      "luck -- check the GDS paths, the weights and the "
                      "render size. Rows are still being written, but they "
                      "are declines, not answers.", file=sys.stderr)
                print("=" * 72, file=sys.stderr)

    # ---- End-of-run summary + mass-failure banner ------------------------
    if total:
        err_frac = error_count / total
        found_frac = found_count / total
        reasons = []
        if err_frac >= MASS_FAILURE_ERROR_FRAC:
            reasons.append(f"{error_count}/{total} pair(s) raised "
                           f"({err_frac:.0%}, threshold "
                           f"{MASS_FAILURE_ERROR_FRAC:.0%})")
        if found_frac < MASS_FAILURE_FOUND_FRAC:
            reasons.append(f"only {found_count}/{total} reported found "
                           f"({found_frac:.0%}, expected ~92% present)")
        if reasons:
            print(f"# mass_failure: errors={error_count} found={found_count} "
                  f"n={total}", file=sys.stderr)
            print("=" * 72, file=sys.stderr)
            print("[MASS FAILURE] This run does not look like a successful "
                  "decode:", file=sys.stderr)
            for why in reasons:
                print(f"  - {why}", file=sys.stderr)
            print("=" * 72, file=sys.stderr)

    if times:
        srt = sorted(times)
        med = srt[len(srt) // 2]
        p90 = srt[min(int(0.9 * len(srt)), len(srt) - 1)]
        print(f"# runtime: median {med:.3f} p90 {p90:.3f} max {max(times):.3f} "
              f"n={len(times)}", file=sys.stderr)
    print(f"wrote {total} rows to {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

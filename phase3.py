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
from driftsense import gds  # noqa: E402
from driftsense import pairs3  # noqa: E402
from driftsense.config import (  # noqa: E402
    PHASE3_CONFIDENCE,
    PHASE3_REFERENCE_BLUR,
    PHASE3_LABEL_CONVENTION,
    PHASE3_SUBPIXEL_ROWS,
    PHASE3_THRESHOLD,
    SHIPPED_BAND,
    SHIPPED_LABEL_CONVENTION,
    SHIPPED_STRIP_ROTATION,
    SHIPPED_SUBPIXEL_ROWS,
    SHIPPED_VERIFICATION,
)
from driftsense.matching import (  # noqa: E402
    LABEL_CONVENTIONS,
    PHASE3_ROTATION_BOUNDS,
    PHASE3_SCALE_BOUNDS,
    locate_phase2,
)

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
    ap.add_argument("--label-convention", default=PHASE3_LABEL_CONVENTION,
                    choices=LABEL_CONVENTIONS,
                    help="pixel convention the x, y columns are written in, "
                         "matching the grader's labels (default: %(default)s). "
                         "'center': pixel i spans [i-0.5, i+0.5]. 'edge': "
                         "pixel i spans [i, i+1).")
    # The pose search box is Phase 3's, not Phase 2's -- see
    # driftsense.matching.PHASE3_*_BOUNDS for why each differs. Exposed as
    # flags so a clarification from the organizers about the stage rotation
    # cap or the magnification spread is a command line, not a code change.
    ap.add_argument("--rotation-bounds", type=float, nargs=2,
                    metavar=("LO", "HI"), default=list(PHASE3_ROTATION_BOUNDS),
                    help="stage rotation search range in degrees "
                         "(default: %(default)s; Phase 2 was -5 5)")
    ap.add_argument("--scale-bounds", type=float, nargs=2,
                    metavar=("LO", "HI"), default=list(PHASE3_SCALE_BOUNDS),
                    help="magnification search range (default: %(default)s; "
                         "Phase 2 was 8 12, nominal here is 10)")
    # Phase 2's drift-row refinement moves ONLY x (driftsense.config:
    # "the correction moves only x"). It models the SEM's slow-scan raster
    # drift, which the Phase 2 search frames carry and a CAD reference does
    # not, so on Phase 3 it is a correction applied to a distortion that is
    # not there -- and it shows up as an x-only bias. Toggleable, and
    # measured rather than assumed.
    ap.add_argument("--subpixel-rows", dest="subpixel_rows",
                    action="store_true", default=None,
                    help="force the Phase 2 drift-row x refinement on")
    ap.add_argument("--no-subpixel-rows", dest="subpixel_rows",
                    action="store_false",
                    help="disable the Phase 2 drift-row x refinement")
    # Which statistic goes in the `score` column. Phase 2's "legacy_min" takes
    # min(network score, native ZNCC); on Phase 3 the network is out of domain
    # (it was trained SEM-against-SEM, and the reference here is a rendered
    # design), so the min drags the stronger signal down. See
    # driftsense.config.PHASE3_CONFIDENCE for the measured AUC table.
    ap.add_argument("--confidence", default=PHASE3_CONFIDENCE,
                    choices=("legacy_min", "zncc"),
                    help="statistic written to the score column "
                         "(default: %(default)s)")
    # Beam-PSF match. The search frame is Gaussian-blurred by
    # sigma = beam_spot_size_nm / 1 nm-per-px at REFERENCE resolution and only
    # then area-downsampled 10x (i4c src/sem_imaging.py:19-33). Our template
    # gets the area-average and not the Gaussian, so the two sides differ by
    # exactly that blur. Applying it to the rendered reference is a forward-model
    # match, not a filter: it is blur only, with none of the noise that made the
    # full cad2sem chain a wash in docs/PHASE3_MEASUREMENT.md.
    ap.add_argument("--reference-blur", type=float, default=PHASE3_REFERENCE_BLUR,
                    metavar="SIGMA",
                    help="Gaussian sigma in reference px applied to the rendered "
                         "GDS before matching; 0 disables (default: %(default)s)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    R.cap_threads(a.threads)

    subpixel_rows = (PHASE3_SUBPIXEL_ROWS if a.subpixel_rows is None
                     else bool(a.subpixel_rows))
    scale_bounds = (float(a.scale_bounds[0]), float(a.scale_bounds[1]))
    rotation_bounds = (float(a.rotation_bounds[0]), float(a.rotation_bounds[1]))
    for name, (lo, hi) in (("--scale-bounds", scale_bounds),
                           ("--rotation-bounds", rotation_bounds)):
        if lo > hi:
            raise SystemExit(f"phase3: {name} lower bound {lo} exceeds upper "
                             f"bound {hi}")
    if scale_bounds[0] <= 0:
        raise SystemExit(f"phase3: --scale-bounds must be positive, got "
                         f"{scale_bounds}")

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
                # The one line that differs from register.py: the reference is
                # rendered from the design file instead of read as an image.
                # A GdsError here is a per-pair failure like an unreadable
                # image, which the Phase 2 contract already handles.
                ref = gds.render_reference(r.reference_gds_path,
                                           size=a.render_size,
                                           min_layer=a.min_layer)
                if a.reference_blur > 0:
                    import cv2  # noqa: PLC0415
                    ref = cv2.GaussianBlur(ref, (0, 0), a.reference_blur)
                sea = I.read_gray(r.search_path)
                if model is None:
                    res = I.zncc_fallback(ref, sea)
                    threshold = R.LEGACY_FALLBACK_THRESHOLD
                    if a.label_convention == "center":
                        res["x"] = float(res["x"]) - 0.5
                        res["y"] = float(res["y"]) - 0.5
                else:
                    threshold = a.threshold
                    res = locate_phase2(model, ref, sea, device, refine=True,
                                        verification=a.verification,
                                        band=SHIPPED_BAND,
                                        subpixel_rows=subpixel_rows,
                                        strip_rot=SHIPPED_STRIP_ROTATION,
                                        label_convention=a.label_convention,
                                        scale_bounds=scale_bounds,
                                        rotation_bounds=rotation_bounds)
                if a.confidence == "zncc":
                    # Fall back to the shipped statistic if the verification
                    # stage did not produce a native ZNCC for this pair, so a
                    # missing feature is a weaker score, never a crash.
                    score = float(res.get("zncc", res.get(
                        "confidence", res.get("score", 0.0))))
                else:
                    score = float(res.get("confidence", res.get("score", 0.0)))
                found = int(score >= threshold)
                out.update({
                    "x": f'{float(res["x"]):.4f}' if found else 0,
                    "y": f'{float(res["y"]):.4f}' if found else 0,
                    "theta": f'{float(res.get("theta", 0.0)):.4f}' if found else 0,
                    "scale": f'{float(res.get("scale", 10.0)):.4f}' if found else 0,
                    "found": found,
                    "score": f"{score:.6f}",
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

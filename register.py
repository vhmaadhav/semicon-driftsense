#!/usr/bin/env python3
"""Phase 2 batch entry point.

    python register.py --input pairs.csv --output predictions.csv

Writes one row per input pair, in input order, with the columns the Phase 2
contract names: pair_id, x, y, theta, scale, found, score.

Scale semantics, fixed by the Phase 2 task material (slide 5 + prompt section
2.3; see .agents/ORGANIZER_PHASE2_GROUND_TRUTH.md section 5): `scale` is the
recovered down-scaling factor z -- nominally in [8, 12], i.e. the search
image's nm/px -- NOT the reference-to-search linear factor 1/z (the two
readings differ by ~100x). theta is degrees, CCW positive as displayed,
about the match centre.

Two properties are treated as non-negotiable, because the scoring rules make
them expensive to get wrong:

* **Every pair_id appears exactly once.** A missing row scores zero, so a pair
  that raises, times out, or has unreadable images still emits a row -- a
  declined answer (`found=0`) rather than no answer.
* **No network, no downloads.** Weights load from the local `weights/`
  directory that ships inside the ZIP.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import infer as I  # noqa: E402
from driftsense.matching import locate_phase2  # noqa: E402

# The shipped Phase 2 operating point lives in driftsense.config (the ONE
# definition of the shipped decode config, so eval_ext.py and the parity tests
# consume the same value register.py does). Re-exported under the historical
# name for backwards compatibility -- every caller in this repo imports it
# from here.
from driftsense.config import SHIPPED_BAND, SHIPPED_THRESHOLD  # noqa: E402
from driftsense.config import SHIPPED_VERIFICATION  # noqa: E402
from driftsense.config import SHIPPED_SUBPIXEL_ROWS  # noqa: E402
from driftsense.config import LEGACY_FALLBACK_THRESHOLD  # noqa: E402,F401

DEFAULT_FOUND_THRESHOLD = SHIPPED_THRESHOLD

OUT_FIELDS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]

# Candidate spellings for the two image columns. The addendum fixes `pair_id`
# but publishes the rest of the pairs.csv layout separately, so accept the
# plausible spellings rather than guess one and fail the whole run.
REF_KEYS = ("reference", "reference_path", "ref", "ref_path", "reference_image",
            "template", "template_path", "high_res", "highres")
SEA_KEYS = ("search", "search_path", "sea", "search_image", "wide", "wide_path",
            "low_res", "lowres")


def cap_threads(requested=0):
    """Cap the torch and OpenCV thread pools to one sane value.

    The judge box is 4-core CPU-only. torch's intra-op default and OpenCV's
    pool both size themselves to every physical core, so without an explicit
    cap they oversubscribe 4 cores catastrophically (grader harness measured
    7.08 s/pair against 1.58 s in a tuned env on the same pairs). Default:
    min(4, os.cpu_count()); the --threads flag overrides for experiments.
    set_flush_denormal is x86-flavoured (avoids the denormal stalls of FP32
    near-zero activation outputs) and is best-effort: on platforms where it
    raises (e.g. ARM) we simply keep denormals.
    """
    n = requested if requested else min(4, os.cpu_count() or 4)
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_flush_denormal(True)
        except Exception:                        # noqa: BLE001
            pass
    except Exception:                            # noqa: BLE001
        pass
    try:
        import cv2
        cv2.setNumThreads(n)                     # noqa: N806 (cv2 spelling)
    except Exception:                            # noqa: BLE001
        pass
    return n


def pick_column(fieldnames, candidates, role):
    lowered = {f.lower().strip(): f for f in fieldnames}
    for c in candidates:
        if c in lowered:
            return lowered[c]
    for f in fieldnames:                      # substring fallback
        if any(c.split("_")[0] in f.lower() for c in candidates):
            return f
    raise SystemExit(f"pairs.csv: could not find the {role} column among {fieldnames}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="pairs.csv")
    ap.add_argument("--output", required=True, help="predictions.csv")
    ap.add_argument("--weights", default=I.DEFAULT_WEIGHTS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_FOUND_THRESHOLD,
                    help="confidence at or above which a pair is reported found. "
                         "Applies to the LEARNED path only: the statistic and the "
                         "threshold are one unit system, so when the weights cannot "
                         "load and the ZNCC fallback runs, this value is ignored and "
                         "the fallback uses its own calibrated "
                         "LEGACY_FALLBACK_THRESHOLD instead -- a fused6 threshold "
                         "applied to a raw NCC would decide nothing meaningful")
    ap.add_argument("--verification", default=SHIPPED_VERIFICATION,
                    help="hypothesis selector: zncc (default) | consensus | majority. "
                         "consensus overrides the native-ZNCC winner only when the rank "
                         "and band scores pick the same different hypothesis; it was "
                         "measured +2/0 and +1/0 rescued/broken on the PR #3 proxy; "
                         "full 2,250-pair A/B (issue #9): +0.11 total, paired CI "
                         "spans zero, 5 broken / 6 rescued -- real but under the "
                         "promotion gate, so zncc stays the default")
    ap.add_argument("--threads", type=int, default=0,
                    help="torch/OpenCV thread cap. 0 (default) auto-caps to "
                         "min(4, CPU cores) to match the 4-core reference "
                         "machine -- it does NOT leave the library defaults, "
                         "because an uncapped pool oversubscribes the grader's "
                         "box and inflates every per-pair time. Pass a positive "
                         "value to override.")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    if a.threads:
        import torch
        torch.set_num_threads(a.threads)

    # Thread sanity before any heavy work (model load touches torch and
    # conv2d; the coarse sweep touches cv2). --threads overrides the default.
    cap_threads(a.threads)

    with open(a.input, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{a.input}: no rows")

    fields = list(rows[0].keys())
    id_col = pick_column(fields, ("pair_id", "id", "pair"), "pair_id")
    ref_col = pick_column(fields, REF_KEYS, "reference")
    sea_col = pick_column(fields, SEA_KEYS, "search")
    base = os.path.dirname(os.path.abspath(a.input))

    def resolve(p):
        p = (p or "").strip()
        return p if os.path.isabs(p) else os.path.join(base, p)

    model, device = I.load_model(a.weights) or (None, None)

    os.makedirs(os.path.dirname(os.path.abspath(a.output)) or ".", exist_ok=True)
    times = []
    # Per-pair timing metadata goes to stderr only: stdout stays the human
    # progress stream and the predictions file stays byte-identical.
    print("# per-pair seconds", file=sys.stderr)
    with open(a.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=OUT_FIELDS)
        w.writeheader()
        for n, r in enumerate(rows):
            pid = r[id_col]
            # Declined answer, overwritten below on success. Constructed first
            # so that any failure path still has a complete row to write.
            out = {"pair_id": pid, "x": 0, "y": 0, "theta": 0, "scale": 0,
                   "found": 0, "score": 0.0}
            t0 = time.perf_counter()
            try:
                ref = I.read_gray(resolve(r[ref_col]))
                sea = I.read_gray(resolve(r[sea_col]))
                if model is None:
                    res = I.zncc_fallback(ref, sea)
                    res.setdefault("scale", 10.0)
                    res.setdefault("theta", 0.0)
                    # The fallback's score is raw ZNCC, not the fused
                    # calibrated P(present): it gates at the LEGACY threshold
                    # (driftsense.config.LEGACY_FALLBACK_THRESHOLD), not at
                    # the shipped fused-units one. Only reachable when the
                    # weights/torch are unavailable -- on the grader box they
                    # ship inside the ZIP, so this is a degraded-mode guard.
                    threshold = LEGACY_FALLBACK_THRESHOLD
                else:
                    threshold = a.threshold
                    # band=False: the difference-of-Gaussians pre-filter on
                    # the coarse sweep costs points on both architectures.
                    # Measured independently here at +0.439 (95% CI
                    # [+0.132, +0.767], P 99.8%) on the 0.456M model and +0.509
                    # (P 94.2%) on the shipped 1.02M one; PR #18 reached the
                    # same conclusion separately. The value is the shipped
                    # decode config (driftsense.config), shared with
                    # eval_ext.py so the evaluator decodes identically.
                    res = locate_phase2(model, ref, sea, device, refine=True,
                                        verification=a.verification,
                                        band=SHIPPED_BAND,
                                        subpixel_rows=SHIPPED_SUBPIXEL_ROWS)
                # The reported confidence (see locate_phase2): fused 6-feature
                # calibrated P(present) on the model path, raw ZNCC on the
                # fallback path.
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
            except Exception as e:                      # noqa: BLE001
                # Never let one bad pair cost the rest of the run, and never
                # drop the row. SystemExit is caught too: read_gray raises
                # SystemExit for an unreadable image, and that must zero-fill
                # THIS row only -- not kill the whole batch.
                print(f"[warn] pair {pid}: {type(e).__name__}: {e}", file=sys.stderr)
            except SystemExit as e:
                print(f"[warn] pair {pid}: SystemExit: {e}", file=sys.stderr)
            w.writerow(out)
            dt = time.perf_counter() - t0
            times.append(dt)
            print(f"# t,{pid},{dt:.3f}", file=sys.stderr)
            if not a.quiet and (n + 1) % 25 == 0:
                print(f"  {n+1}/{len(rows)}  median {np.median(times):.2f}s", flush=True)
                f.flush()

    if not a.quiet:
        t = np.array(times)
        print(f"wrote {len(rows)} rows to {a.output}")
        print(f"runtime: median {np.median(t):.2f}s  p90 {np.percentile(t,90):.2f}s  "
              f"max {t.max():.2f}s  total {t.sum()/60:.1f} min")
        if t.max() > 20:
            print(f"WARNING: {int((t>20).sum())} pair(s) exceeded the 20 s hard timeout",
                  file=sys.stderr)
    # Machine-readable runtime summary: stderr, same numbers as the stdout
    # line, for the judge harness to parse without scraping progress text.
    t = np.array(times)
    print(f"# runtime: median {np.median(t):.2f} p90 {np.percentile(t,90):.2f} "
          f"max {t.max():.2f} n={len(t)}", file=sys.stderr)


if __name__ == "__main__":
    main()

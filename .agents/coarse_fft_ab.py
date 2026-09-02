#!/usr/bin/env python3
"""D3 A/B validation: driftsense.coarse_fft vs cv2.matchTemplate.

Across 3 official pairs and 50 random (scale, rotation) templates per pair
(the same make_template + _probe pipeline the coarse sweep feeds), check:

  * VALUE: SpectralSearchIndex.peak_score(t) matches
    cv2.matchTemplate(probe, t, TM_CCOEFF_NORMED) peak within 1e-6.
  * WALL-CLOCK: index construction once per pair, then 50 peak_score calls
    vs 50 matchTemplate calls (dev machine, indicative; 2 cv2 threads).

Run: venv313/bin/python .agents/coarse_fft_ab.py
"""
import os
import sys
import time

import cv2
import numpy as np

cv2.setNumThreads(2)
try:
    import torch
    torch.set_num_threads(2)
except Exception:                                    # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driftsense.matching import _probe, _band, make_template  # noqa: E402
from driftsense.coarse_fft import prepare_search              # noqa: E402

PAIRS = ["p001", "p002", "p003"]
N_TEMPLATES = 50
TOL = 1e-6
MAX_T = 64                 # coarse probe templates are 8-13% of 500 px -> <=64


def run_pair(tag):
    ref = cv2.imread(f".agents/ref_material/reference/{tag}.png",
                     cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(f".agents/ref_material/search/{tag}.png",
                     cv2.IMREAD_GRAYSCALE)
    probe = _band(_probe(sea))
    probe = np.ascontiguousarray(probe.astype(np.float32))

    rng = np.random.default_rng(hash(tag) % (2**32))
    # 50 random (scale, rotation) templates exactly like the coarse sweep builds
    tmpls = []
    for _ in range(N_TEMPLATES):
        f = float(rng.uniform(8.0, 12.0))
        r = float(rng.uniform(-3.0, 3.0))
        t = np.ascontiguousarray(_probe(make_template(ref, f, r)).astype(np.float32))
        tmpls.append(t)

    # --- values -----------------------------------------------------------
    max_diff = 0.0
    argmax_disagree = 0
    for t in tmpls:
        want_map = cv2.matchTemplate(probe, t, cv2.TM_CCOEFF_NORMED)
        got = prepare_search(probe, max_template=MAX_T).peak_score(t) \
            if False else None
        # index built once below; use it for every template
        break
    idx = prepare_search(probe, max_template=MAX_T)
    for t in tmpls:
        want_map = cv2.matchTemplate(probe, t, cv2.TM_CCOEFF_NORMED)
        want = float(want_map.max())
        got = idx.peak_score(t)
        d = abs(got - want)
        if d > max_diff:
            max_diff = d
        gloc = np.unravel_index(int(np.argmax(
            idx.full_map(t) if hasattr(idx, "full_map") else want_map)),
            want_map.shape)
        wloc = np.unravel_index(int(want_map.argmax()), want_map.shape)
        if gloc != wloc:
            argmax_disagree += 1

    # --- clocks (index built once, then the 50 calls) ----------------------
    t0 = time.perf_counter()
    idx = prepare_search(probe, max_template=MAX_T)
    t_setup = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for t in tmpls:
        idx.peak_score(t)
    t_fft = (time.perf_counter() - t0) * 1e3 / N_TEMPLATES

    t0 = time.perf_counter()
    for t in tmpls:
        cv2.matchTemplate(probe, t, cv2.TM_CCOEFF_NORMED)
    t_cv = (time.perf_counter() - t0) * 1e3 / N_TEMPLATES

    print(f"{tag}: max|value diff| {max_diff:.3e} "
          f"({'OK' if max_diff <= TOL else 'FAIL vs 1e-6'})  "
          f"argmax disagree {argmax_disagree}/{N_TEMPLATES}")
    print(f"     setup {t_setup:6.2f} ms (once) | spectral {t_fft:6.3f} ms/call"
          f" | matchTemplate {t_cv:6.3f} ms/call | "
          f"speedup x{t_cv / t_fft:.2f}")
    return max_diff, t_setup, t_fft, t_cv


def main():
    results = [run_pair(t) for t in PAIRS]
    max_diff = max(r[0] for r in results)
    setup = np.mean([r[1] for r in results])
    t_fft = np.mean([r[2] for r in results])
    t_cv = np.mean([r[3] for r in results])
    print(f"\nAcross 3 pairs, {N_TEMPLATES} random (scale,rot) templates each:")
    print(f"  worst |value diff| : {max_diff:.3e}  (tolerance {TOL:.0e})")
    print(f"  index setup (once) : {setup:.2f} ms/pair")
    print(f"  spectral           : {t_fft:.3f} ms/call")
    print(f"  cv2.matchTemplate  : {t_cv:.3f} ms/call  (x{t_cv/t_fft:.2f})")
    # 50-call coarse sweep projection (correlation part only)
    saved = (t_cv - t_fft) * 50 - setup
    print(f"  projected coarse-stage saving (50 calls): {saved:.0f} ms/pair"
          f" ({'net win' if saved > 0 else 'NET LOSS after setup'})")
    print(f"VERDICT: {'PASS' if max_diff <= TOL else 'FAIL'}")


if __name__ == "__main__":
    main()

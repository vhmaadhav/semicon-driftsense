#!/usr/bin/env python3
"""D2 microbenchmark: is there a real ceiling in eliminating the redundant
probe-search DFT of the coarse sweep?

The coarse stage calls cv2.matchTemplate(500x500 probe_search, ~50x50
template) ~207 times per pair, and every call recomputes the DFT of the SAME
probe search. A precomputed-search spectral scorer would pay the search DFT
once and then only mulSpectrums+idft per template. This script measures, on 3
official pairs:

  (a) 100x cv2.matchTemplate(probe_search, template)          -- current cost
  (b) 1x dft(probe) + 100x (dft(tmpl) + mulSpectrums + idft)  -- spectral cost
  (c) projected per-pair saving for the ~100-call coarse sweep

Plain cross-correlation via FFT only -- ZNCC normalization is NOT applied
(timing is the question, not values). Timings: dev machine, indicative.

Run: venv313/bin/python .agents/fft_ceiling_tmp.py
"""
import cv2
import numpy as np
import os
import sys
import time

cv2.setNumThreads(2)
try:
    import torch
    torch.set_num_threads(2)
except Exception:                                    # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from driftsense.matching import _probe, _band, make_template  # noqa: E402

N_CALLS = 100
COARSE_CALLS_PER_PAIR = 100          # the ~100-call coarse sweep (of ~207 total)
PAIR_MEDIAN_S = 3.09                 # official-20 dev median (reg_after.err)
COARSE_SHARE = 0.614                 # coarse stage share of pair time (stage profile)

PAIRS = ["p001", "p002", "p003"]


def spectral_corr(search_f32, tmpl, F_search, pad_shape):
    """One plain FFT cross-correlation with the search DFT precomputed.

    Template at the TOP-LEFT of the canvas + conjB=True gives
    cc[y, x] = sum_u search[y+u, x+v] * tmpl[u, v] directly at offset (y, x)
    (verified against a naive sliding dot on a random toy and against
    cv2.matchTemplate TM_CCORR on the real probes). The extract region is
    wrap-free as long as pad_h - tmpl_h >= H - tmpl_h, i.e. any pad >= H.
    Returns the map cropped to the valid region (matchTemplate's shape)."""
    th, tw = tmpl.shape
    H, W = search_f32.shape
    t_pad = np.zeros(pad_shape, np.float32)
    t_pad[:th, :tw] = tmpl
    F_t = cv2.dft(t_pad, flags=cv2.DFT_COMPLEX_OUTPUT)
    prod = cv2.mulSpectrums(F_search, F_t, 0, conjB=True)
    corr = cv2.idft(prod, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return corr[:H - th + 1, :W - tw + 1]


def bench_one(tag):
    ref = cv2.imread(f".agents/ref_material/reference/{tag}.png",
                     cv2.IMREAD_GRAYSCALE)
    sea = cv2.imread(f".agents/ref_material/search/{tag}.png",
                     cv2.IMREAD_GRAYSCALE)
    probe = _band(_probe(sea))
    probe = np.ascontiguousarray(probe.astype(np.float32))
    # realistic template: nominal factor 10, probed like the coarse sweep does
    tmpl = np.ascontiguousarray(_probe(make_template(ref, 10.0)).astype(np.float32))
    # a few more templates across the scale grid, like the sweep
    tmpls = [np.ascontiguousarray(_probe(make_template(ref, f)).astype(np.float32))
             for f in (8.0, 9.0, 10.0, 11.0, 12.0)]

    # (a) current: 100x matchTemplate
    t0 = time.perf_counter()
    for i in range(N_CALLS):
        cv2.matchTemplate(probe, tmpls[i % len(tmpls)], cv2.TM_CCOEFF_NORMED)
    ta = (time.perf_counter() - t0) / N_CALLS * 1e3

    # (b) spectral: 1x dft(probe) + N x (dft(tmpl) + mul + idft)
    H, W = probe.shape
    fh = cv2.getOptimalDFTSize(H + tmpl.shape[0] - 1)
    fw = cv2.getOptimalDFTSize(W + tmpl.shape[1] - 1)
    padded = np.zeros((fh, fw), np.float32)
    padded[:H, :W] = probe
    t0 = time.perf_counter()
    F_search = cv2.dft(padded, flags=cv2.DFT_COMPLEX_OUTPUT)
    t_dft_search = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    for i in range(N_CALLS):
        spectral_corr(probe, tmpls[i % len(tmpls)], F_search, (fh, fw))
    tb_loop = (time.perf_counter() - t0) / N_CALLS * 1e3
    tb = tb_loop + t_dft_search / N_CALLS

    # sanity: plain-correlation values must match cv2.matchTemplate(TM_CCORR)
    # bit-tightly (same linear correlation, different summation order), and
    # the peak location must agree exactly.
    mt = cv2.matchTemplate(probe, tmpl, cv2.TM_CCORR)
    sp = spectral_corr(probe, tmpl, F_search, (fh, fw))
    val_diff = float(np.abs(mt - sp).max())
    argmax_agree = np.unravel_index(np.argmax(mt), mt.shape) == \
        np.unravel_index(np.argmax(sp), sp.shape)

    print(f"{tag}: matchTemplate {ta:6.3f} ms/call   "
          f"precomputed-FFT {tb:6.3f} ms/call   "
          f"(search dft {t_dft_search:6.2f} ms amortized; per-call spectral loop "
          f"{tb_loop:6.3f} ms)")
    print(f"     sanity vs TM_CCORR: max|diff| {val_diff:.3e} "
          f"(rel {val_diff / np.abs(mt).max():.2e}), argmax agree: {argmax_agree}")
    return ta, tb


def main():
    results = [bench_one(t) for t in PAIRS]
    ta = np.mean([a for a, _ in results])
    tb = np.mean([b for _, b in results])

    per_call_saving = ta - tb
    # coarse stage today: share of pair median
    coarse_ms = PAIR_MEDIAN_S * 1000 * COARSE_SHARE
    # matchTemplate per call in the sweep vs our (a) microbench may differ
    # slightly (real templates vary in size), but (a) IS the same op at the
    # same sizes, so scale directly:
    projected_saving_ms = per_call_saving * COARSE_CALLS_PER_PAIR
    projected_stage_pct = 100 * projected_saving_ms / coarse_ms

    print(f"\nmean matchTemplate      : {ta:.3f} ms/call")
    print(f"mean precomputed-FFT    : {tb:.3f} ms/call")
    print(f"per-call saving         : {per_call_saving:.3f} ms "
          f"({100*per_call_saving/ta:.1f}% of the call)")
    print(f"coarse stage today      : {coarse_ms:.0f} ms/pair "
          f"({PAIR_MEDIAN_S}s median x {COARSE_SHARE:.1%})")
    print(f"projected saving        : {projected_saving_ms:.0f} ms/pair "
          f"= {projected_stage_pct:.1f}% of the coarse stage")
    verdict = ("GO (>=15% of coarse stage) -> build driftsense/coarse_fft.py"
               if projected_stage_pct >= 15 else
               "NO-GO (<15% of coarse stage) -> record negative result, skip D3")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()

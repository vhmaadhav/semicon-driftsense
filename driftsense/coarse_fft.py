"""Spectral precomputed-search coarse scorer (workstream A, deliverable 3).

The pose coarse sweep correlates ~50 templates against the SAME half-resolution
probe search image. Every ``cv2.matchTemplate`` call re-computes the DFT of that
500x500 probe, which the microbenchmark in ``.agents/fft_ceiling_tmp.py``
measured at roughly half the per-call cost. This module hoists the search DFT
out of the loop: ``prepare_search(search)`` pays the search-side DFT (plus f64
integral tables for the window moments) exactly once, and each
``index.peak_score(template)`` afterwards costs one small template DFT, one
``mulSpectrums`` and one ``idft`` -- returning the same peak ZNCC value
``cv2.matchTemplate(search, template, TM_CCOEFF_NORMED)`` would report.

Value fidelity: the DFT cross-correlation runs in float32 (as cv2's own DFT
path does) but the ZNCC moments -- window sums and sums of squares -- come from
precomputed float64 integral tables, so the assembled statistic matches cv2's
TM_CCOEFF_NORMED to ~1e-7 (verified in tests and in
``.agents/coarse_fft_ab.py``). Zero-variance windows (flat regions) are scored
0.0, mirroring templatch's den<=0 convention; a template larger than the
search image scores -inf, mirroring ``matching._peak_score``.

STATUS: flag-gated experiment, default OFF, deliberately NOT wired into
``locate_phase2`` -- the integrator audits and wires it (workstream A has no
mandate to change the shipped decode).

All numbers quoted here are dev machine, indicative (Compute protocol: the
integrator re-times on the idle 4-core x86 reference box).
"""
from __future__ import annotations

import cv2
import numpy as np

__all__ = ["SpectralSearchIndex", "prepare_search"]


class SpectralSearchIndex:
    """A search image whose DFT and moment tables are computed once.

    Parameters
    ----------
    search:
        2-D float32 search image (the coarse probe, e.g. the banded
        half-resolution frame the coarse sweep correlates against).
    max_template:
        Largest template side the fast path supports. The search is padded to
        ``getOptimalDFTSize(H + max_template)``, which is wrap-free for every
        valid correlation offset of any template up to that size. Templates
        larger than this fall back to a per-call pad (correct, not fast);
        templates larger than the search image are rejected with -inf.
    """

    def __init__(self, search: np.ndarray, max_template: int = 64):
        search = np.ascontiguousarray(search, dtype=np.float32)
        if search.ndim != 2:
            raise ValueError(f"search must be 2-D, got shape {search.shape}")
        h, w = search.shape
        if h < 1 or w < 1:
            raise ValueError(f"search too small: {search.shape}")
        self.search_shape = (h, w)
        self._search = search
        self.max_template = int(max_template)

        ph = cv2.getOptimalDFTSize(h + self.max_template)
        pw = cv2.getOptimalDFTSize(w + self.max_template)
        self.pad_shape = (ph, pw)
        pad = np.zeros((ph, pw), np.float64)
        pad[:h, :w] = search
        # The one DFT the whole coarse sweep used to pay ~50 times. Kept in
        # float64 (the per-call float32 DFT roundoff, ~2e-6 relative on the
        # numerator, breaks the 1e-6 ZNCC contract) and CCS-packed
        # (DFT_REAL_OUTPUT: half the memory traffic of the complex spectrum,
        # ~1.2 ms/call cheaper, bit-equivalent to the complex route).
        self.F_search = cv2.dft(pad, flags=cv2.DFT_REAL_OUTPUT)

        # float64 moment tables for the ZNCC denominator/numerator correction:
        # SX and SXX of every window, from integral images (O(1) per window).
        s64 = search.astype(np.float64)
        self._I1 = cv2.integral(s64)                 # sums
        self._I2 = cv2.integral(s64 * s64)           # sums of squares

    # -- moment helpers -----------------------------------------------------

    def _window_moments(self, th: int, tw: int) -> tuple[np.ndarray, np.ndarray]:
        """(SX, SXX) of every th x tw window, float64, shape (H-th+1, W-tw+1)."""
        i1, i2 = self._I1, self._I2
        sx = i1[th:, tw:] - i1[:-th, tw:] - i1[th:, :-tw] + i1[:-th, :-tw]
        sxx = i2[th:, tw:] - i2[:-th, tw:] - i2[th:, :-tw] + i2[:-th, :-tw]
        return sx, sxx

    # -- scoring ------------------------------------------------------------

    def peak_score(self, template: np.ndarray) -> float:
        """Max ZNCC of `template` over the search, matching cv2's
        TM_CCOEFF_NORMED peak. Templates larger than the search score -inf
        (same convention as matching._peak_score)."""
        template = np.ascontiguousarray(template, dtype=np.float32)
        if template.ndim != 2:
            raise ValueError(f"template must be 2-D, got shape {template.shape}")
        th, tw = template.shape
        h, w = self.search_shape
        if th >= h or tw >= w:
            return -np.inf
        if th > self.max_template or tw > self.max_template:
            # Correctness path: a fresh index sized for this template. The
            # coarse sweep never hits it (probe templates are <= ~64 px).
            return SpectralSearchIndex(
                self._search, max_template=max(th, tw)).peak_score(template)

        n = th * tw
        t64 = template.astype(np.float64)
        st = float(t64.sum())
        var_t = float((t64 * t64).sum()) - st * st / n
        if var_t <= 0.0:
            # A constant template has zero correlation variance; cv2 scores
            # every window 0 in this case (templmatch's den<=0 convention).
            return 0.0

        # Cross-correlation in the spectral domain, against the MEAN-SUBTRACTED
        # template. Identity (exact, not an approximation):
        #   num = sum_uv s[y+u,x+v]*t[u,v] - SX[y,x]*ST/n
        #       = corr(s, t - t_mean)[y, x]      (since sum(t - t_mean) = 0)
        # so the spectral map IS the ZNCC numerator. Centering also removes the
        # DC term from the product, keeping float32 DFT magnitudes at the
        # fluctuation scale (~1e5 instead of ~1.6e7 on real probes) -- without
        # it the absolute float32 error (~1.0) is a 1e-5 ZNCC error; with it,
        # ~1e-8. Template at the TOP-LEFT of the canvas + conjB=True makes
        # cc[y, x] = sum_u s[y+u, x+v] * t[u, v] directly at offset (y, x),
        # and the extract region below is wrap-free because
        # y+u <= (H-th) + (th-1) = H-1 < pad (verified against a naive sliding
        # dot and against cv2.matchTemplate in the tests).
        t_mean = float(t64.mean())
        t_pad = np.zeros(self.pad_shape, np.float64)
        t_pad[:th, :tw] = t64 - t_mean
        f_t = cv2.dft(t_pad, flags=cv2.DFT_REAL_OUTPUT)
        prod = cv2.mulSpectrums(self.F_search, f_t, 0, conjB=True)
        cc = cv2.idft(prod, flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)
        num = cc[:h - th + 1, :w - tw + 1]

        sx, sxx = self._window_moments(th, tw)
        var_w = sxx - sx * sx / n
        den2 = var_w * var_t
        with np.errstate(divide="ignore", invalid="ignore"):
            zncc = np.where(den2 > 0.0, num / np.sqrt(np.maximum(den2, 0.0)), 0.0)
        score = float(np.max(zncc))
        return score if np.isfinite(score) else 0.0


def prepare_search(search: np.ndarray, max_template: int = 64) -> SpectralSearchIndex:
    """Pay the search DFT once; reuse the index across the whole coarse sweep."""
    return SpectralSearchIndex(search, max_template=max_template)

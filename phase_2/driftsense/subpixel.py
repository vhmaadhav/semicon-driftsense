"""Sub-pixel localisation refinement for the Phase 2 decode.

Two drop-in replacements for the 1-D parabolic fit inside
``driftsense.matching.refine_zncc``. Both share that function's contract::

    variant(search_window, template, cx, cy) -> (x, y, score)

where ``(cx, cy)`` is the coarse template-centre in ``search_window``
coordinates and the returned ``(x, y)`` uses the same frame. Neither function
touches global state; both are deterministic and pure numpy/cv2.

Why: the shipped decode places the template with an integer-precision
``matchTemplate`` snap plus a 1-D parabola through the correlation peak. A
parabola is the wrong model for a ZNCC peak (it is not quadratic), and the
error budget at the 1.00 px credit tier is dominated by exactly this bias.
Literature grounding:

* Debella-Gilo & Kääb (2011), Remote Sens. Environ., DOI
  10.1016/j.rse.2010.08.012 -- bicubic interpolation of the correlation
  surface beats 1-D parabolic peak fits; cutting error by 40-80% on
  upscaled imagery.
* Guizar-Sicairos, Thurman & Fienup (2008), Opt. Lett. 33, DOI
  10.1364/OL.33.000156 -- upsampled-DFT cross-correlation: evaluate the
  DFT of the correlation surface on a fine grid around the integer peak by
  a matrix-multiply DFT, with no interpolation error at all. NoRMCorre
  (Pnevmatikakis & Giovannucci 2017, DOI 10.1016/j.jneumeth.2017.07.031)
  uses the same routine for motion correction.
"""
from __future__ import annotations

import cv2
import numpy as np

# Window geometry copied from driftsense.matching so a variant sees exactly
# the pixels refine_zncc sees (locate_phase2 calls it with radius=4, the
# module-level REFINE_RADIUS there).
REFINE_RADIUS = 4
# Sub-pixel grid: 40x upsample of a 7x7 (bicubic) neighbourhood around the
# integer peak -- a 0.025 px grid, well below the 1 px credit tier and enough
# for the 0.05 px accuracy the tests assert. The DFT variant uses a 1/20 px
# grid by the same argument (its residual error is dominated by the data, not
# the grid).
UP_FACTOR = 40


def parabola_1d(vm1: float, v0: float, vp1: float) -> float:
    """Vertex offset of the parabola through (-1, vm1), (0, v0), (1, vp1).

    Exposed so tests can reproduce the shipped 1-D parabolic baseline.
    """
    denom = vm1 - 2.0 * v0 + vp1
    if denom == 0.0:
        return 0.0
    return 0.5 * (vm1 - vp1) / denom


def _crop_window(search: np.ndarray, template: np.ndarray,
                 cx: float, cy: float, radius: int):
    """Replicates refine_zncc's window crop exactly.

    Returns (window, x0c, y0c, bx, by) or None when the window degenerates.
    ``bx``/``by`` is the continuous template top-left implied by (cx, cy).
    """
    h, w = search.shape
    th, tw = template.shape
    bx, by = cx - tw / 2.0, cy - th / 2.0

    x0, y0 = int(round(bx)) - radius, int(round(by)) - radius
    x1, y1 = x0 + tw + 2 * radius, y0 + th + 2 * radius

    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, w), min(y1, h)
    if x1c - x0c < tw + 1 or y1c - y0c < th + 1:
        return None
    return search[y0c:y1c, x0c:x1c], x0c, y0c, bx, by


def _zncc_surface(window: np.ndarray, template: np.ndarray) -> np.ndarray:
    """The correlation surface refine_zncc computes on the same window."""
    return cv2.matchTemplate(np.ascontiguousarray(window, dtype=np.float32),
                             np.ascontiguousarray(template, dtype=np.float32),
                             cv2.TM_CCOEFF_NORMED)


def refine_bicubic(search: np.ndarray, template: np.ndarray,
                   cx: float, cy: float,
                   radius: int = REFINE_RADIUS,
                   up_factor: int = UP_FACTOR,
                   half: int = 3) -> tuple[float, float, float]:
    """Refine (cx, cy) by bicubic upsampling of the correlation surface.

    Matches the small +/- ``radius`` px window with ``TM_CCOEFF_NORMED``,
    then upsamples a (2*half+1)^2 neighbourhood around the integer peak by
    ``up_factor`` with a cubic kernel and returns the continuous argmax.
    Deterministic; ~0.2 ms on a 100 px template.
    """
    cropped = _crop_window(search, template, cx, cy, radius)
    if cropped is None:
        return cx, cy, 0.0
    window, x0c, y0c, bx, by = cropped

    res = _zncc_surface(window, template)
    _, score, _, loc = cv2.minMaxLoc(res)
    pj, pi = loc
    th, tw = template.shape

    # Neighbourhood around the peak, clamped at the surface border.
    rh, rw = res.shape
    j0, j1 = max(pj - half, 0), min(pj + half + 1, rw)
    i0, i1 = max(pi - half, 0), min(pi + half + 1, rh)
    patch = res[i0:i1, j0:j1]
    if patch.shape[0] < 3 or patch.shape[1] < 3:
        return (x0c + pj) + template.shape[1] / 2.0, \
               (y0c + pi) + template.shape[0] / 2.0, float(score)

    up = cv2.resize(patch, (patch.shape[1] * up_factor, patch.shape[0] * up_factor),
                    interpolation=cv2.INTER_CUBIC)
    k = int(np.argmax(up))
    # cv2.resize maps output pixel i to input coordinate (i + 0.5)/f - 0.5
    # (pixel-centre convention), so invert that mapping for the argmax.
    ui, uj = divmod(k, up.shape[1])
    xi = (uj + 0.5) / up_factor - 0.5
    yi = (ui + 0.5) / up_factor - 0.5

    x = (j0 + xi) + tw / 2.0          # absolute surface coord -> window
    y = (i0 + yi) + th / 2.0
    return x0c + x, y0c + y, float(score)


def refine_upsampled_dft(search: np.ndarray, template: np.ndarray,
                         cx: float, cy: float,
                         radius: int = REFINE_RADIUS,
                         up_factor: int = UP_FACTOR) -> tuple[float, float, float]:
    """Refine (cx, cy) with Guizar-Sicairos upsampled-DFT cross-correlation.

    The ZNCC peak of a window/template pair sits where the (mean-removed)
    cross-correlation is maximal; instead of interpolating that surface, this
    evaluates the exact DFT of the correlation on a 1/``up_factor`` px grid
    around the integer peak via a matrix-multiply DFT (Guizar-Sicairos et al.
    2008, Eq. 4-6). No interpolation kernel is involved, so the argmax is
    limited only by the grid step (1/20 px here) and the data.

    Equal-size patches (the GS/NoRMCorre configuration): the refinement
    compares the template against the template-sized patch of the window
    centred on the integer peak. On equal-size arrays the (circular)
    cross-correlation peak is the ZNCC peak -- with a LARGER window the raw
    correlation's argmax drifts off the normalized optimum because the
    numerator grows with overlapping fragment energy while the denominator
    does not, which is exactly the artefact observed before this choice.

    For patch ``a`` and template ``b`` (both (th, tw)), the cross-correlation
    ``r[m, n] = sum a[x, y] b[(x-m) mod th, (y-n) mod tw]`` is
    ``ifft2(fft2(a) * conj(fft2(b)))`` by the shift theorem. Around the
    integer offset ``(m0, n0)``::

        r[m0+s, n0+t] = 1/(PQ) * sum_{k,l} G[k, l]
                          * exp(+2i pi (k (m0+s)/P + l (n0+t)/Q))

    with ``G = fft2(a) * conj(fft2(b))``, which factors into two small
    matmuls: the fixed phase ``exp(2i pi (k m0/P + l n0/Q))`` recentres the
    expansion on the peak, and ``Ks[k, i] = exp(2i pi k s_i/P)`` /
    ``Kl[j, l] = exp(2i pi l t_j/Q)`` carry the fine offsets. At P=Q=100 and
    a 25-point grid this is ~2 * 25 * 10^4 complex MACs: ~1 ms.

    The fine grid spans +/-0.6 px around the integer peak (the ZNCC optimum
    lies within +/-0.5 px of the integer argmax by construction). Wrapping is
    harmless: a shift in [0, 0.6] px reads rows that only wrap where the
    patch is already outside the window at integer offset, which cannot be
    the argmax. Deterministic; returns the peak ZNCC value from the same
    surface refine_zncc uses as ``score``.
    """
    cropped = _crop_window(search, template, cx, cy, radius)
    if cropped is None:
        return cx, cy, 0.0
    window, x0c, y0c, bx, by = cropped
    th, tw = template.shape

    # Integer peak implied by the coarse centre (see docstring).
    # bx/by are (x, y) = (col, row) coordinates: m0 is the ROW offset,
    # n0 the COLUMN offset of the template top-left inside the window.
    m0 = int(round(by)) - y0c
    n0 = int(round(bx)) - x0c

    # Refine the RESIDUAL around the integer peak: extract the template-sized
    # patch at the integer top-left (equal-size GS/NoRMCorre configuration --
    # on a larger window the raw correlation's argmax drifts off the
    # normalized optimum because the numerator grows with fragment energy).
    a = window[m0:m0 + th, n0:n0 + tw].astype(np.float64)
    b = template.astype(np.float64)
    a -= a.mean()
    b -= b.mean()

    # Fractional alignment as Fourier shift of the template: bf(d) has its
    # content displaced by d = (dm, dn) px. Because the shift is circular,
    # Parseval guarantees ||bf(d)|| == ||b||, so the normalizer is CONSTANT
    # and argmax of the normalized correlation == argmax of
    #     c(d) = (1/(PQ)) * sum_k,l conj(A)[k,l] * B[k,l]
    #                        * exp(-2i pi (kf dm + lf dn))
    # (Plancherel: sum_x a(x) bf(x) = (1/N) sum_k conj(A_k) C_k; the sum is
    # real because the k <-> N-k terms pair as complex conjugates).
    # c(0,0) is exactly the numerator of ZNCC(patch, template), so the fine
    # surface is the *normalized* correlation surface -- the same quantity
    # matchTemplate's TM_CCOEFF_NORMED maximizes -- evaluated on the fine
    # grid without any interpolation kernel. Offsets are found by two small
    # matmuls (Guizar-Sicairos' matrix-multiply DFT); at 96x96 and a 25-point
    # grid that is ~0.5 M complex MACs: ~1 ms.
    #
    # Native (unpadded) FFT size: a phase ramp on a zero-padded array
    # interpolates with a periodized sinc whose first alias sits only
    # (pad - size) px away and contaminates the fine surface; unpadded, the
    # first wrap is a full period (~96 px) away and negligible.
    P, Q = a.shape
    Fa = np.fft.fft2(a)
    Fb = np.fft.fft2(b)
    G = np.conj(Fa) * Fb

    # Fine offsets: 1/up_factor px steps spanning +/-0.6 px around the
    # integer peak (the continuous optimum lies within +/-0.5 px of the
    # integer argmax by construction). With up_factor=20: a 25x25 grid.
    half_px = 0.6
    n_grid = int(2 * round(half_px * up_factor) + 1)
    s = (np.arange(n_grid, dtype=np.float64) - (n_grid - 1) / 2.0) / up_factor
    t = s.copy()                        # same grid on both axes

    kf = np.fft.fftfreq(P)[None, :]     # frequency in cycles/sample
    lf = np.fft.fftfreq(Q)[None, :]
    # Ks[i, k] = exp(-2i pi kf_k s_i) -> (n_grid, P)
    # Kl[j, l] = exp(-2i pi lf_l t_j) -> (n_grid, Q)
    # R[i, j] = (Ks @ G @ Kl.T)[i, j] / (P*Q) = c(s_i, t_j)
    Ks = np.exp(-2j * np.pi * s[:, None] * kf)
    Kl = np.exp(-2j * np.pi * t[:, None] * lf)
    R = (Ks @ G @ Kl.T / (P * Q)).real

    im, jm = np.unravel_index(int(np.argmax(R)), R.shape)
    # R's rows index dm (rows -> y), cols index dn (cols -> x). The residual
    # adds to the integer top-left the same way: content shifted by +d means
    # the template's top-left moves by +d.
    x = (n0 + t[jm]) + tw / 2.0
    y = (m0 + s[im]) + th / 2.0

    # Score: peak ZNCC on the same surface refine_zncc scores with.
    res = _zncc_surface(window, template)
    _, score, _, _ = cv2.minMaxLoc(res)
    return x0c + x, y0c + y, float(score)

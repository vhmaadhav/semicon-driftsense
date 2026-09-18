"""Raster-shear estimation and correction for Phase 3 (issue #101).

The defect
----------

The organizer's Search capture shears every scan row
(``src/sem_imaging.apply_raster_drift``)::

    row_shift = shear_amplitude_px * row / (h - 1) + jitter
    map_x     = x + row_shift          # SAMPLES at x + s  =>  content moves -s

while ``gt_x``/``gt_y`` are computed on the **pre-drift** geometry
(``src/cad_pipeline.render_cad_sample``). Imaged content therefore sits
``-A*row/(h-1)`` from where the answer says it is. This is not a matcher
defect: the matcher correctly reports where the content *is*.

Measured on the paired A-sweep (``scripts/gen_phase3_rotated.py --paired``, 60
pairs, rotation U(-10, +10)), the decode's signed x residual against ``y/999``
fits::

    slope = -1.0394 * A - 0.271        dy stays flat at +0.03 px

so **104% of the amplitude survives the decode as an x-only bias** -- the pose
search does not absorb it. The cost is the whole localisation block:

    A      0      1      2      3      4
    median 0.32   0.79   1.50   2.17   2.90  px
    <=1px  95%    88%    36%    31%    19%

Undoing it means adding ``A*y/(h-1)`` back to the reported x, which needs
``A`` -- and ``A`` is a generator parameter, not a constant. The 20 curated
cases published with the generator use ``{0.0, 2.5, 3.0, 3.5, 4.0}`` against a
1.5 default, so a fixed correction is a coin flip (issue #101 has the payoff
matrix).

What is identifiable, and what is not
-------------------------------------

A **global** shear maps a lattice to a lattice, so nothing in the Search frame
alone separates "this layout was sheared" from "this layout's own lattice is
slightly oblique". Two frame-wide estimators were built and measured against
the paired sweep before this one, and both failed the same way -- unit gain in
``A``, so the physics is right, sitting on a per-pair additive offset of sd ~3
px that survives with zero noise, zero jitter and zero drift, i.e. is a
property of the design canvas:

* reciprocal-basis non-orthogonality of the frame's two Manhattan grating
  families -- the estimator Sang & LeBeau (*Ultramicroscopy* 138 (2014) 28) use
  on a crystal, where the orthogonality is exact and this layout's is not;
* de-shearing the full-frame alias-peak columns until they are sharpest, which
  additionally collapses under rotation (the columns tilt by theta, 176 px
  across the frame at 10 deg).

That degeneracy is why the scan-distortion literature estimates affine drift
from **two** acquisitions -- orthogonal scan directions (Ophus, Ciston &
Nelson, *Ultramicroscopy* 171 (2016) 104, arXiv:1507.00320) or a rotated series
(Sang & LeBeau) -- rather than from one frame. See
``docs/PHASE3_RASTER_SHEAR.md``.

The one quantity a single frame *does* identify is the displacement field
between the frame and a **known undistorted model of its content**, and Phase 3
supplies exactly that: the reference is the exact design. ``row_offsets``
aligns the posed template's rows against the search's, so any obliquity the
design carries is present on both sides and cancels, and the slope of those
offsets against row index is ``-A/(h-1)`` with no geometric bias. This is the
single-frame, reference-driven member of the non-rigid scan-registration family
(Jones et al., *Adv. Struct. Chem. Imaging* 1:8 (2015)).

Two terms live in that slope and only one is wanted::

    row fit:     dx/dy = -A/rows - tan(dtheta)
    column fit:  dy/dx =          +tan(dtheta)

``dtheta`` is the decode's own rotation error. It is not a nuisance that
averages away: over a 100-row window rotation and shear are first-order
degenerate, so ``polish_pose`` absorbs part of the shear into the angle and the
row fit alone reads only 0.32 of the amplitude. Raster drift is x-only
(``map_y`` is the identity), so the transposed fit sees ``dtheta`` and not the
shear, and combining the two restores the gain.

Why this is pooled over the batch
---------------------------------

One site spans the template's ~100 rows, where the ramp is only ``0.1*A`` px.
The per-row 1-D offset read carries ~1.3 px of noise, so one site is worth ~4.5
px of amplitude and a whole frame, over every site it offers, is worth **4.47
px** (measured, ten sets). Against a signal that only ranges over 0-4 px,
**per-pair estimation does not work, and this module does not pretend
otherwise.** Note where that noise comes from: the 0.5 px per-row jitter alone
would allow ~1.7 px, so this is reader-limited, not jitter-limited, and a
better per-row estimator is the lever if anyone wants to revisit it.

Pooled over the ~45 measured pairs of one batch the same measurement is worth
**0.79 px**, and that is enough to be worth several localisation points. :func:`measure` is
therefore per-pair and :func:`pool` is per-batch, and the caller applies one
amplitude to the whole batch. That is a real property of the output, not a
detail: **with the correction enabled, a pair's answer depends on the other
pairs in the same run.** It is off by default for that reason.

The pooled estimate is gated (:data:`SHEAR_GATE`) rather than applied at
whatever it happens to read, because the amplitudes that pay are the large ones
and a small reading is not distinguishable from zero at this precision. The
gate is what makes the ``A = 0`` null control exactly flat -- the control this
repo has twice been burned for skipping (``docs/PHASE3_MEASUREMENT.md``).

Discipline
----------

Reads only ``search_path`` and ``reference_gds_path`` -- never a withheld
column, so nothing here can work on a training split and fail on a blind one.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from driftsense.matching import parabolic, row_offsets, row_pitch

# --------------------------------------------------------------------------
# Site selection.
#
# The periodic layout, which is what makes Phase 3 localisation hard, hands the
# sites over for free: the template correlates strongly at dozens of
# lattice-translated positions. Only the WITHIN-site slope is used from each --
# the between-site offsets are exactly the part the design's own lattice
# corrupts.

# Correlation floor for a site to be measured, as a fraction of the frame's own
# best peak and in absolute ZNCC. 0.70 relative was chosen on the dev sweep
# against 0.35 and 0.55: the looser floors admit sites in neighbouring mats,
# whose content differs by fabrication distortion and whose slopes scatter at
# sd 12 against 3.5 for sites in the reference's own mat.
SITE_REL_THRESHOLD = 0.70
SITE_ABS_THRESHOLD = 0.30

# Non-maximum-suppression radius in the correlation surface, in px. Smaller
# than the ~10 px lattice pitch so neighbouring repeats stay separate sites.
SITE_NMS_RADIUS = 4

# Row bands the frame is split into, and how many sites are taken per band.
# The cap is what buys the row span: without it the strongest peaks cluster
# inside one mat.
SITE_BAND_PX = 100
SITE_PER_BAND = 4
MAX_SITES = 40

# Half-width of the box around the reported centre a site must fall in.
# Measured on the dev sweep: unrestricted sites lower the pooled gain
# (0.80 -> 0.74) and widen the per-pair spread, because a distant site is a
# different mat with its own fabrication distortion.
SITE_RADIUS = 300.0

# --------------------------------------------------------------------------
# Per-site fit.

# Correlation floor for one row's offset to be trusted; mirrors
# DRIFT_ROW_MIN_CORR's role in drift_row_refine, slightly looser because a
# slope tolerates a bad row better than a single centre-row read does.
ROW_MIN_CORR = 0.35
ROW_MIN_USABLE = 40
ROW_LAG = 6

# Rows averaged into one offset sample before the slope is fitted. A single
# 100 px template row is a weak 1-D signal, and at a rotated pose it is weaker
# still, so the raw per-row offsets carry far more measurement noise than the
# 0.5 px jitter that sets the floor. Averaging k rows costs nothing in signal
# -- the ramp moves 0.004*A px per row, so a 10-row block is flat to 0.04*A px.
ROW_BLOCK = 10

# Theil-Sen pairs are restricted to rows at least this far apart: close pairs
# see 0.004 px of ramp against the noise and contribute only variance.
TS_MIN_SPAN = 20

# --------------------------------------------------------------------------
# Combination and acceptance.

# Weight on the column (rotation-error) term. The row term alone reads 0.32 of
# the amplitude because polish_pose absorbs the rest into the angle; 1.0
# restores the gain to 0.97 but triples the per-pair spread, and 0.5 was the
# dev-sweep optimum on pooled accuracy (gain 0.796, max sweep residual 0.54
# px). docs/PHASE3_RASTER_SHEAR.md has the full grid.
COLUMN_WEIGHT = 0.5

# A site slope outside these bounds has mis-measured rather than found a large
# drift. The rotation bound is generous against PR #102's measured 0.104 deg
# median pose error, which is ~1.8 px in these units.
A_SITE_BOUNDS = (-6.0, 12.0)
ROT_SITE_BOUNDS = (-20.0, 20.0)

MIN_SITES = 2

# --------------------------------------------------------------------------
# Pooling calibration.
#
# Two independent paired sweeps, each 5 amplitudes x 60 pairs at rotation
# U(-10, +10): dev (seed 20260918) and held-out (seed 20261124). Their own fits
# are 0.8252*A - 0.0011 and 0.6892*A - 0.1840, and each transfers to the other
# at a mean +3.9/40 localisation gain -- so the ~16% gain spread between two
# 60-pair sweeps is the honest transfer error, and it is reported rather than
# tuned away (docs/PHASE3_RASTER_SHEAR.md has both directions). The shipped
# constants are the fit over all ten sets, which is what leave-one-sweep-out
# validation licenses:
#
#     batch median of `measure` = 0.7572 * A - 0.0926     (max residual 0.51)
#
# so the batch amplitude is the inverse of that line. The gain is below 1
# because `polish_pose` absorbs part of the within-patch ramp into the angle
# and COLUMN_WEIGHT returns only half of it. It is a property of the decode,
# which is why it is calibrated rather than derived -- and erring low is the
# safe direction, since an under-correction still removes most of the bias
# while an over-correction adds error of its own.
#
# Reproduce with:
#     experiments/phase3/shear_sweep.py table --root <set> ... --calibrate
SHEAR_GAIN = 0.7572
SHEAR_OFFSET = -0.0926

# Below this the reading is not worth acting on, and correcting by it costs
# more than it returns. The amplitudes that pay -- the curated
# {2.5, 3.0, 3.5, 4.0} -- clear it comfortably. This is what makes the null
# control exact: on the held-out sweep A = 0 reads 0.19 and is gated off for a
# localisation delta of exactly +0.00/40.
SHEAR_GATE = 1.5

# A pooled estimate needs a batch. Below this many measured pairs the median is
# not worth the risk, and the correction declines.
MIN_POOL_PAIRS = 12

# Hard bound on the applied amplitude, well outside anything the generator
# draws, so a pathological batch cannot produce a large correction.
A_BOUNDS = (-2.0, 8.0)


@dataclass(frozen=True)
class ShearMeasurement:
    """One pair's contribution to the batch estimate.

    ``value`` is in the raw units of the measurement, **not** an amplitude --
    :func:`pool` applies the calibration, so the calibration lives in exactly
    one place.
    """

    value: float
    n_row: int
    n_col: int
    row_span: float
    n_bands: int
    ok: bool
    reason: str = ""


NOT_MEASURED = ShearMeasurement(float("nan"), 0, 0, 0.0, 0, False,
                                "not attempted")


@dataclass(frozen=True)
class ShearCorrection:
    """The one amplitude applied to a whole batch."""

    amplitude: float          # what `correct_x` applies; 0.0 when gated off
    estimate: float           # the calibrated amplitude before the gate
    batch_median: float       # the pooled raw measurement
    n_pairs: int
    applied: bool
    reason: str = ""


NO_CORRECTION = ShearCorrection(0.0, float("nan"), float("nan"), 0, False,
                                "not attempted")


def _theil_sen(idx: np.ndarray, val: np.ndarray, min_span: int = TS_MIN_SPAN,
               min_points: int = 5, min_pairs: int = 8):
    """Median pairwise slope, plus the residual scatter about it.

    Theil-Sen rather than least squares because a row that locked onto the
    neighbouring lattice repeat is off by a whole pitch (~10 px) against a
    signal of ~0.4 px: one such row destroys a least-squares slope and moves a
    median not at all.
    """
    n = len(idx)
    if n < min_points:
        return None
    i1, i2 = np.triu_indices(n, 1)
    span = idx[i2] - idx[i1]
    far = span >= min_span
    if far.sum() < min_pairs:
        return None
    slope = float(np.median((val[i2][far] - val[i1][far]) / span[far]))
    resid = val - slope * idx
    scatter = float(1.4826 * np.median(np.abs(resid - np.median(resid))))
    return slope, scatter


def _axis_slope(search: np.ndarray, template: np.ndarray, cx: float, cy: float,
                lag: int, min_corr: float, block: int = ROW_BLOCK):
    """Shared body of :func:`site_slope` and :func:`site_slope_T`.

    ``row_offsets`` is asked for its correlation surface rather than its
    per-row argmax, so rows can be pooled into blocks before the peak is taken
    (see ``ROW_BLOCK``). A block that locked onto a neighbouring lattice repeat
    is pulled to the repeat nearest the site's own consensus -- the same unwrap
    ``drift_row_refine`` performs, applied one level up.
    """
    got = row_offsets(search, template, cx, cy, lag=lag, return_corr=True)
    if got[0] is None:
        return None
    off, peak, corr = got
    ok = np.isfinite(off) & (peak > min_corr)
    if ok.sum() < ROW_MIN_USABLE:
        return None

    th, nlag = corr.shape
    centre = float(np.median(off[ok]))
    pitch = row_pitch(template)
    unwrap = pitch is not None and 4.0 <= pitch <= 30.0

    idx, val = [], []
    for b in range(th // block):
        lo = b * block
        c = corr[lo:lo + block].mean(axis=0)
        k = int(np.argmax(c))
        if k == 0 or k == nlag - 1 or not (c[k] >= min_corr):
            continue
        d = (k - lag) + parabolic(c[k - 1], c[k], c[k + 1])
        if unwrap:
            for n in (-2, -1, 1, 2):
                cand = d + n * pitch
                j = int(round(cand + lag))
                if (abs(cand - centre) < abs(d - centre) - 1e-9
                        and 0 < j < nlag - 1 and c[j] > 0.7 * c[k]):
                    d = cand
        idx.append(lo + (block - 1) / 2.0)
        val.append(d)
    if len(idx) < 5:
        return None
    return _theil_sen(np.asarray(idx, float), np.asarray(val, float))


def site_slope(search: np.ndarray, template: np.ndarray, cx: float, cy: float,
               lag: int = ROW_LAG, min_corr: float = ROW_MIN_CORR):
    """Per-row x-offset slope at one site, px per row, or ``None``.

    ``dx/dy = -A/(h-1) - tan(dtheta)`` -- the shear plus the decode's rotation
    error. :func:`site_slope_T` is what separates them.
    """
    return _axis_slope(search, template, cx, cy, lag, min_corr)


def site_slope_T(search: np.ndarray, template: np.ndarray, cx: float, cy: float,
                 lag: int = ROW_LAG, min_corr: float = ROW_MIN_CORR):
    """The same fit on the transposed frame: per-column y-offset slope.

    Raster drift is x-only -- ``map_y`` in ``apply_raster_drift`` is the
    identity -- so it contributes nothing here, and ``dy/dx = +tan(dtheta)`` is
    the rotation error alone.
    """
    return _axis_slope(np.ascontiguousarray(search.T),
                       np.ascontiguousarray(template.T), cy, cx, lag, min_corr)


def alias_sites(search: np.ndarray, template: np.ndarray,
                centre: tuple[float, float] | None = None,
                radius: float = SITE_RADIUS,
                rel_threshold: float = SITE_REL_THRESHOLD,
                abs_threshold: float = SITE_ABS_THRESHOLD,
                nms_radius: int = SITE_NMS_RADIUS,
                band_px: int = SITE_BAND_PX, per_band: int = SITE_PER_BAND,
                max_sites: int = MAX_SITES) -> list[tuple[float, float, float]]:
    """Strong template-correlation sites, spread over the frame's rows.

    Returns ``(cx, cy, zncc)`` template centres. ``centre`` (the decode's
    reported position) and ``radius`` keep the sites inside the reference's own
    mat, where the content really is the same design.
    """
    th, tw = template.shape
    if search.shape[0] < th + 2 or search.shape[1] < tw + 2:
        return []
    corr = cv2.matchTemplate(np.ascontiguousarray(search, dtype=np.float32),
                             np.ascontiguousarray(template, dtype=np.float32),
                             cv2.TM_CCOEFF_NORMED)
    thr = max(float(abs_threshold), float(rel_threshold) * float(corr.max()))
    work = corr.copy()
    _H, W = work.shape
    taken: dict[int, int] = {}
    out: list[tuple[float, float, float]] = []
    # Bounded: every pass zeroes a (2r+1)^2 block, so this cannot spin.
    for _ in range(max_sites * 40):
        i = int(np.argmax(work))
        py, px = divmod(i, W)
        v = float(work[py, px])
        if v < thr:
            break
        work[max(py - nms_radius, 0):py + nms_radius + 1,
             max(px - nms_radius, 0):px + nms_radius + 1] = -9.0
        cx, cy = px + tw / 2.0, py + th / 2.0
        if centre is not None and (abs(cx - centre[0]) > radius
                                   or abs(cy - centre[1]) > radius):
            continue
        band = int(cy // band_px)
        if taken.get(band, 0) >= per_band:
            continue
        taken[band] = taken.get(band, 0) + 1
        out.append((cx, cy, v))
        if len(out) >= max_sites:
            break
    return out


def measure(search: np.ndarray, template: np.ndarray,
            centre: tuple[float, float] | None = None,
            rows: float | None = None,
            column_weight: float = COLUMN_WEIGHT,
            min_sites: int = MIN_SITES) -> ShearMeasurement:
    """One pair's raw shear reading. Feed a batch of these to :func:`pool`.

    ``template`` must already be posed into the search frame
    (``matching.make_template(reference, scale, theta)``): raster drift acts on
    scan rows, so the row correspondence this relies on only exists once
    rotation and scale are undone.

    The two directions are pooled **separately**. ``dtheta`` is one number for
    the whole frame, so every site that yields a column fit should inform it,
    including sites whose row fit failed -- which is common, because the
    per-row x correlation carries the jitter and the per-column y correlation
    does not.
    """
    if rows is None:
        rows = float(search.shape[0] - 1)
    sites = alias_sites(search, template, centre=centre)
    if not sites:
        return ShearMeasurement(float("nan"), 0, 0, 0.0, 0, False, "no sites")

    a_row, a_col, ys = [], [], []
    for cx, cy, _v in sites:
        fit = site_slope(search, template, cx, cy)
        if fit is not None:
            v = -rows * fit[0]
            if A_SITE_BOUNDS[0] <= v <= A_SITE_BOUNDS[1]:
                a_row.append(v)
                ys.append(cy)
        fit_t = site_slope_T(search, template, cx, cy)
        if fit_t is not None:
            v = -rows * fit_t[0]
            if ROT_SITE_BOUNDS[0] <= v <= ROT_SITE_BOUNDS[1]:
                a_col.append(v)

    if len(a_row) < min_sites or len(a_col) < min_sites:
        return ShearMeasurement(
            float("nan"), len(a_row), len(a_col), 0.0, 0, False,
            f"{len(a_row)} row / {len(a_col)} column site(s)")
    yv = np.asarray(ys, float)
    value = float(np.median(a_row) + column_weight * np.median(a_col))
    return ShearMeasurement(value, len(a_row), len(a_col),
                            float(yv.max() - yv.min()),
                            len({int(v // SITE_BAND_PX) for v in yv}), True, "")


def pool(measurements, gain: float = SHEAR_GAIN, offset: float = SHEAR_OFFSET,
         gate: float = SHEAR_GATE,
         min_pairs: int = MIN_POOL_PAIRS) -> ShearCorrection:
    """Turn a batch of measurements into the one amplitude to apply.

    The median rather than the mean: a pair whose decode landed in the wrong
    lattice basin reads an arbitrary slope, and such pairs exist by
    construction in this task.
    """
    vals = np.array([m.value for m in measurements
                     if m.ok and np.isfinite(m.value)], float)
    if len(vals) < min_pairs:
        return ShearCorrection(0.0, float("nan"), float("nan"), len(vals),
                               False,
                               f"{len(vals)} measured pair(s) < {min_pairs}")
    if gain <= 0:
        return ShearCorrection(0.0, float("nan"), float("nan"), len(vals),
                               False, "non-positive calibration gain")
    med = float(np.median(vals))
    est = float(np.clip((med - offset) / gain, *A_BOUNDS))
    if abs(est) < gate:
        return ShearCorrection(0.0, est, med, len(vals), False,
                               f"|estimate| {abs(est):.3f} < gate {gate}")
    return ShearCorrection(est, est, med, len(vals), True, "")


def correct_x(x: float, y: float, amplitude: float, rows: float) -> float:
    """Report the pre-drift x the grader compares against.

    Content at row ``y`` was imaged ``amplitude*y/rows`` to the left of where
    the design puts it, so adding it back is the whole correction. There is no
    separate gain term here: the measured residual slope is ``-1.0394*A``, i.e.
    the decode passes the bias through essentially untouched.
    """
    if not np.isfinite(amplitude) or amplitude == 0.0 or rows <= 0:
        return float(x)
    return float(x) + float(amplitude) * float(y) / float(rows)

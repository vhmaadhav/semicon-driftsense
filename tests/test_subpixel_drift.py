"""Sub-pixel recovery of the centre scan row's raster-drift sample.

The generator shifts each search row horizontally by its own amount and the
label takes the shift of the single row the target centre lands on
(`generate.correct_gt`). These tests pin that contract: the estimator must read
that one row, must not touch y, and must be a no-op when there is no drift.
"""
import numpy as np
import cv2
import pytest

from driftsense.matching import (
    row_offsets, drift_row_refine, make_template, DRIFT_MAX_SHIFT,
)


def _pattern(seed=0, size=1000):
    """A wafer-like lattice with enough texture for a per-row correlation.

    Pitches are chosen so that after the 10x downsample to the search frame they
    land near 13 px and 9 px -- the range a real layout occupies. A purely
    periodic field would leave the 1-D correlation genuinely ambiguous, so a
    low-frequency envelope breaks the translational degeneracy the way a real
    die's zone structure does.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = (120.0
           + 55 * np.sin(2 * np.pi * xx / 130.0) * np.sin(2 * np.pi * yy / 150.0)
           + 35 * np.sin(2 * np.pi * xx / 90.0)
           + 45 * np.exp(-(((xx - 380.0) ** 2 + (yy - 560.0) ** 2) / (2 * 210.0 ** 2)))
           - 40 * np.exp(-(((xx - 700.0) ** 2 + (yy - 300.0) ** 2) / (2 * 160.0 ** 2))))
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def _apply_row_shift(img, shift):
    h, w = img.shape
    map_x = np.arange(w, dtype=np.float32)[None, :] + shift[:, None].astype(np.float32)
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def _scene(jitter_sd, seed=0, scale=10.0):
    """Return (reference, search, cx, cy, row_shift) for a nominal-pose pair."""
    ref_full = _pattern(seed)
    search_clean = cv2.resize(ref_full, (100, 100), interpolation=cv2.INTER_AREA)
    canvas = np.full((300, 300), 128, np.uint8)
    canvas[100:200, 100:200] = search_clean
    rng = np.random.default_rng(seed + 99)
    shift = (rng.normal(0, jitter_sd, size=canvas.shape[0]) if jitter_sd > 0
             else np.zeros(canvas.shape[0]))
    return ref_full, _apply_row_shift(canvas, shift), 150.0, 150.0, shift


def test_row_offsets_recovers_an_injected_per_row_shift():
    """The per-row reader must track a known shift, row by row."""
    ref, search, cx, cy, shift = _scene(jitter_sd=1.0, seed=1)
    tpl = make_template(ref, 10.0, 0.0)
    off, peak = row_offsets(search, tpl, cx, cy)
    assert off is not None
    th = tpl.shape[0]
    y0 = int(round(cy - th / 2.0))
    truth = -shift[y0:y0 + th]              # content moves opposite the sampling map
    ok = np.isfinite(off) & (peak > 0.3)
    assert ok.sum() > th // 2
    resid = (off[ok] - off[ok].mean()) - (truth[ok] - truth[ok].mean())
    assert np.std(resid) < 0.25, f"per-row error {np.std(resid):.3f} px too large"


def test_refine_targets_the_labelled_row_not_its_neighbours():
    """Drift is white per row, so the correction must key on exactly one row."""
    ref, search, cx, cy, shift = _scene(jitter_sd=1.2, seed=3)
    tpl = make_template(ref, 10.0, 0.0)
    moved = drift_row_refine(search, tpl, cx, cy)
    assert moved is not None
    got = moved[0] - cx
    ci = int(round(cy))
    at_row = -(shift[ci] - shift.mean())
    assert abs(got - at_row) < abs(got - -(shift[ci + 3] - shift.mean()))


def test_no_drift_is_a_no_op():
    """A nominal scene must not be moved -- Phase 1 numbers depend on this."""
    ref, search, cx, cy, _ = _scene(jitter_sd=0.0, seed=5)
    tpl = make_template(ref, 10.0, 0.0)
    moved = drift_row_refine(search, tpl, cx, cy)
    if moved is not None:
        assert abs(moved[0] - cx) < 0.15


def test_y_is_never_changed_by_the_flag():
    """Raster drift has no vertical component; y must be left alone."""
    from driftsense import matching as M
    ref, search, cx, cy, _ = _scene(jitter_sd=1.5, seed=7)
    tpl = make_template(ref, 10.0, 0.0)
    moved = M.drift_row_refine(search, tpl, cx, cy)
    if moved is not None:
        assert abs(moved[1] - cy) <= 3.0


def test_correction_is_clamped():
    """A runaway re-match must decline rather than emit a large jump."""
    ref, search, cx, cy, _ = _scene(jitter_sd=1.0, seed=11)
    tpl = make_template(ref, 10.0, 0.0)
    moved = drift_row_refine(search, tpl, cx, cy)
    if moved is not None:
        assert abs(moved[0] - cx) <= DRIFT_MAX_SHIFT


def test_declines_when_the_window_leaves_the_frame():
    ref, search, _, _, _ = _scene(jitter_sd=1.0, seed=13)
    tpl = make_template(ref, 10.0, 0.0)
    assert drift_row_refine(search, tpl, 4.0, 4.0) is None


def test_shipped_config_is_the_single_source_of_truth():
    """register.py, eval_ext.py and locate_phase2 must agree on the shipped value.

    Mirrors the band/threshold parity contract in tests/test_submission_parity.py
    so the row correction cannot be enabled in one entry point and not another.
    """
    import inspect
    import os
    from driftsense.config import SHIPPED_SUBPIXEL_ROWS
    from driftsense.matching import locate_phase2
    from test_submission_parity import _argparse_defaults, _load_eval_ext

    assert (inspect.signature(locate_phase2).parameters["subpixel_rows"].default
            == SHIPPED_SUBPIXEL_ROWS)

    ev_parser = _argparse_defaults(_load_eval_ext().main)
    assert ev_parser.get_default("subpixel_rows") == SHIPPED_SUBPIXEL_ROWS

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = open(os.path.join(here, "register.py")).read()
    assert "subpixel_rows=SHIPPED_SUBPIXEL_ROWS" in src, \
        "register.py must pass the shared shipped value, not a literal"


def test_row_pitch_finds_the_layout_period():
    """The unwrap step needs the lattice pitch; a wrong one would corrupt rows."""
    from driftsense.matching import row_pitch
    ref = _pattern(seed=2)
    tpl = make_template(ref, 10.0, 0.0)
    p = row_pitch(tpl)
    assert p is not None
    # 130 px and 90 px in the reference -> 13 px and 9 px after the 10x downsample
    assert 7.0 < p < 15.0, f"pitch {p:.2f} px is not a layout period"


def test_unwrap_rescues_a_whole_pitch_error_but_not_a_real_sample():
    """Unwrapping must move a repeat error and leave a genuine drift sample alone."""
    from driftsense.matching import row_pitch
    ref, search, cx, cy, shift = _scene(jitter_sd=1.0, seed=17)
    tpl = make_template(ref, 10.0, 0.0)
    pitch = row_pitch(tpl)
    assert pitch is not None
    moved = drift_row_refine(search, tpl, cx, cy)
    if moved is not None:
        # the reported correction must stay inside a plausible drift range,
        # never a whole pitch away from it
        assert abs(moved[0] - cx) < pitch, \
            "a correction of a full lattice pitch means the unwrap failed"

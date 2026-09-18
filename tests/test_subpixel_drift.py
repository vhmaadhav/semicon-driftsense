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


# ------------------------------------------------- shrinkage (issue #89)

def test_correction_is_scaled_by_how_much_the_frame_actually_drifts():
    """A quiet frame gets a small correction, a drifting one a large correction.

    The measured row offset is the row's own drift sample plus measurement
    noise; with almost no drift to recover, nearly all of it is noise, and
    applying it whole was measured to make x worse than not correcting at all.
    """
    moved = {}
    for sd in (0.05, 1.2):
        ref, search, cx, cy, _ = _scene(jitter_sd=sd, seed=7)
        tpl = make_template(ref, 10.0, 0.0)
        got = drift_row_refine(search, tpl, cx, cy)
        moved[sd] = abs(got[0] - cx) if got is not None else 0.0
    assert moved[0.05] < moved[1.2], (
        f"quiet frame moved {moved[0.05]:.3f} px, drifting frame {moved[1.2]:.3f} px")


def test_shrinkage_never_flips_or_amplifies_the_correction():
    """It scales the row's offset into [0, 1] of itself -- never past it, never
    against it, so the stage can only ever move x part of the way it would."""
    ref, search, cx, cy, _ = _scene(jitter_sd=1.2, seed=11)
    tpl = make_template(ref, 10.0, 0.0)
    full = drift_row_refine(search, tpl, cx, cy, shrink_sigma=None)
    shrunk = drift_row_refine(search, tpl, cx, cy)
    assert full is not None and shrunk is not None
    a, b = full[0] - cx, shrunk[0] - cx
    assert a * b >= 0, "shrinkage must not reverse the correction"
    assert abs(b) <= abs(a) + 1e-9, "shrinkage must not amplify the correction"


def test_shrink_sigma_none_reproduces_the_unshrunk_correction():
    """The switch is a true off switch: the pre-#89 answer stays reachable."""
    ref, search, cx, cy, _ = _scene(jitter_sd=1.0, seed=13)
    tpl = make_template(ref, 10.0, 0.0)
    got = drift_row_refine(search, tpl, cx, cy, shrink_sigma=None, align_rows=False)
    assert got is not None
    off, peak = row_offsets(search, tpl, cx, cy)
    ci = int(round(cy)) - int(round(cy - tpl.shape[0] / 2.0))
    assert got[0] - cx == pytest.approx(off[ci], abs=0.5)


def test_row_alignment_is_a_no_op_when_the_template_is_already_on_the_grid():
    """`align_rows` only resamples by the sub-pixel part of the row offset, so
    at a whole-pixel cy there is nothing to resample and the measurement is
    bit-identical. That is what makes it safe to leave on."""
    ref, search, cx, _, _ = _scene(jitter_sd=1.0, seed=17)
    tpl = make_template(ref, 10.0, 0.0)
    cy = 150.0 + tpl.shape[0] / 2.0 - round(tpl.shape[0] / 2.0)   # dy == 0
    plain = row_offsets(search, tpl, cx, cy, align_rows=False)
    aligned = row_offsets(search, tpl, cx, cy, align_rows=True)
    for a, b in zip(plain, aligned):
        np.testing.assert_array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))


# ------------------------------------ lattice repeats across rows (issue #104)

def _fin_pattern(seed=0, size=1000):
    """A FinFET-like field: parallel vertical fins, and no zone envelope.

    `_pattern` deliberately breaks the translational degeneracy with a
    low-frequency envelope. This one does not, which is the point: a row cut
    across parallel fins is near-periodic, so its 1-D correlation curve carries
    several near-equal rivals one fin pitch apart and the argmax can land on
    the wrong one.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = (120.0 + 60 * np.sin(2 * np.pi * xx / 100.0)
           + 12 * np.sin(2 * np.pi * yy / 70.0)
           + rng.normal(0, 2.0, (size, size)))
    return np.clip(img, 0, 255).astype(np.uint8)


def _fin_scene(seed=0, row_noise=45.0):
    """A fin scene with enough per-pixel noise to make some rows mis-lock."""
    ref = _fin_pattern(seed)
    canvas = np.full((300, 300), 128, np.uint8)
    canvas[100:200, 100:200] = cv2.resize(ref, (100, 100), interpolation=cv2.INTER_AREA)
    rng = np.random.default_rng(seed + 5)
    noisy = canvas.astype(np.float32) + rng.normal(0, row_noise, canvas.shape)
    return ref, np.clip(noisy, 0, 255).astype(np.uint8), 150.0, 150.0


@pytest.mark.parametrize("seed", [0, 2, 3])
def test_band_resolves_rows_that_locked_onto_the_neighbouring_fin(seed):
    """Neighbouring rows disagree about where the peak is only when one of them
    is on the wrong repeat: drift is white, the layout's periodicity is not."""
    ref, search, cx, cy = _fin_scene(seed)
    tpl = make_template(ref, 10.0, 0.0)
    alone, _ = row_offsets(search, tpl, cx, cy, band_sigma=0.0)
    banded, _ = row_offsets(search, tpl, cx, cy, band_sigma=2.0)
    # There is no drift in this scene, so any offset near a whole fin pitch is
    # a repeat error rather than a sample.
    assert np.nansum(np.abs(alone) > 5.0) >= 1, "scene did not produce a repeat error"
    assert np.nansum(np.abs(banded) > 5.0) == 0


def test_band_changes_which_peak_is_measured_never_the_row_s_own_value():
    """The sub-pixel offset stays fitted to the row's own correlation curve, so
    a row the band agrees with is bit-identical and a row it moves is moved by
    a whole repeat -- never by a fraction borrowed from its neighbours."""
    ref, search, cx, cy = _fin_scene(seed=3)
    tpl = make_template(ref, 10.0, 0.0)
    alone, _ = row_offsets(search, tpl, cx, cy, band_sigma=0.0)
    banded, _ = row_offsets(search, tpl, cx, cy, band_sigma=2.0)
    changed = np.isfinite(alone) & np.isfinite(banded) & (alone != banded)
    assert changed.any(), "the band changed nothing; the test proves nothing"
    assert np.abs(alone - banded)[changed].min() > 5.0
    np.testing.assert_array_equal(alone[~changed & np.isfinite(alone)],
                                  banded[~changed & np.isfinite(banded)])


def test_band_off_is_the_default_and_a_true_no_op():
    from driftsense.matching import DRIFT_BAND_SIGMA
    ref, search, cx, cy, _ = _scene(jitter_sd=1.0, seed=19)
    tpl = make_template(ref, 10.0, 0.0)
    plain = row_offsets(search, tpl, cx, cy)
    explicit = row_offsets(search, tpl, cx, cy, band_sigma=DRIFT_BAND_SIGMA)
    for a, b in zip(plain, explicit):
        np.testing.assert_array_equal(np.nan_to_num(a, nan=-9), np.nan_to_num(b, nan=-9))


# ------------------------------------------- the quiet-frame cap (issue #104)

def test_a_confident_frame_caps_the_correction():
    """Past the gate the correction is capped, and the cap is on the correction
    -- the rest of the re-match is left exactly where it was."""
    ref, search, cx, cy, _ = _scene(jitter_sd=1.5, seed=3)
    tpl = make_template(ref, 10.0, 0.0)
    uncapped = drift_row_refine(search, tpl, cx, cy, quiet_gate=None)
    capped = drift_row_refine(search, tpl, cx, cy, quiet_gate=0.5, frame_conf=0.9,
                              quiet_max_shift=0.1)
    assert uncapped is not None and capped is not None
    assert abs(uncapped[0] - cx) > abs(capped[0] - cx)
    assert capped[1] == uncapped[1], "the cap must not touch y"


def test_the_gate_is_on_the_frame_and_nothing_else():
    """Below the gate -- an ordinary or degraded frame -- the stage is untouched."""
    ref, search, cx, cy, _ = _scene(jitter_sd=1.5, seed=3)
    tpl = make_template(ref, 10.0, 0.0)
    plain = drift_row_refine(search, tpl, cx, cy, quiet_gate=None)
    below = drift_row_refine(search, tpl, cx, cy, quiet_gate=0.95, frame_conf=0.2,
                             quiet_max_shift=0.1)
    unknown = drift_row_refine(search, tpl, cx, cy, quiet_gate=0.5, frame_conf=None,
                               quiet_max_shift=0.1)
    assert plain == below == unknown

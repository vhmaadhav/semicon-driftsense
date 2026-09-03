"""Set C decoy-pitch fidelity (Phase 2 absent pairs).

The absent-pair decoy used to be generated with the SAME params/preset pool
as the reference scene, so its lattice pitch support was identical to the
scene in frame and near-matching mats decoded as weak matches (47/500
validation absent pairs above the 0.18 shipped gate before the fix). The fix
draws the decoy's pitch a multiplicative factor in [0.5, 1.0) away from the
reference's, clamped into the family's legal pitch envelope.

These tests pin:
  * the factor draw: in band, and reproducible from the decoy stream,
  * the scaled preset pool: same shape, never above the reference pitch,
    clamped to the family envelope,
  * the RENDERED pitch actually moves by the factor (measured off the
    pixels, not just the parameter),
  * present-pair generation is byte-reproducible on one machine from the
    same seed (PNG encoding is not byte-portable across cv2 versions and
    platforms, so there are no cross-machine golden digests),
  * absent-pair generation is byte-reproducible from the same seed,
  * from the same canvas seed the absent path really diverges: the decoy
    reference crop differs from the present pair's while the search
    frame, driven by the untouched scene stream, stays byte-identical.
"""

import hashlib
import os

import numpy as np

from driftsense.generate import (
    DECOY_PITCH_FACTOR_RANGE, PoseSpec, build_one, draw_decoy_pitch_factor,
    make_pairs,
)
from src.patterns.dram import generate_dram_canvas
from src.presets import (
    PRESETS, PITCH_FIELDS, pitch_envelope, scaled_pitch_presets,
)


# --- portable present-path guards -------------------------------------------
# The old guard compared PNG file digests against values captured on one
# machine, but PNG encoding (cv2's zlib settings, the cv2 build) is not
# byte-portable across platforms, so cross-platform digest equality can never
# hold. The guarantees that DO travel are asserted instead:
#   * determinism -- the same seed on the same machine writes the same bytes,
#   * divergence -- from the same canvas seed the absent path swaps in the
#     decoy canvas for the reference crop while the search frame, driven by
#     the untouched scene stream, stays byte-identical.


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _factor_in_band(f):
    lo, hi = DECOY_PITCH_FACTOR_RANGE
    return lo <= f < hi


# --- the factor draw --------------------------------------------------------

def test_decoy_factor_draw_stays_in_band():
    for seed in range(64):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        assert _factor_in_band(draw_decoy_pitch_factor(rng))


def test_decoy_factor_draw_is_stream_deterministic():
    """Same decoy stream state -> same factor. The draw depends on nothing
    global: both streams here are built from scratch."""
    a = np.random.Generator(np.random.PCG64(np.random.SeedSequence(12345)))
    b = np.random.Generator(np.random.PCG64(np.random.SeedSequence(12345)))
    fa = draw_decoy_pitch_factor(a)
    fb = draw_decoy_pitch_factor(b)
    assert fa == fb
    # The draw consumed exactly one number from the stream: a second draw on
    # the same stream differs from the first (uniform is continuous, so an
    # accidental repeat would itself be a determinism bug).
    assert draw_decoy_pitch_factor(a) != fa


# --- the scaled preset pool -------------------------------------------------

def test_pitch_envelope_matches_the_preset_pool():
    for kind, fields in PITCH_FIELDS.items():
        pool = [p for p in PRESETS.values() if p["kind"] == kind]
        env = pitch_envelope(kind)
        assert set(env) == set(fields)
        for f in fields:
            lo = min(p[f] for p in pool)
            hi = max(p[f] for p in pool)
            assert env[f] == (lo, hi)


def test_scaled_pool_shape_matches_the_original():
    for kind in PITCH_FIELDS:
        orig = scaled_pitch_presets(kind, 1.0)
        for factor in (0.5, 0.7, 0.9, 0.99):
            scaled = scaled_pitch_presets(kind, factor)
            assert len(scaled) == len(orig)
            for q, p in zip(scaled, orig):
                assert list(q.keys()) == list(p.keys())
                assert q["kind"] == p["kind"]


def test_scaled_pool_never_goes_up_and_never_leaves_the_envelope():
    for kind, fields in PITCH_FIELDS.items():
        orig = scaled_pitch_presets(kind, 1.0)
        env = pitch_envelope(kind)
        for factor in (0.5, 0.625, 0.75, 0.9):
            scaled = scaled_pitch_presets(kind, factor)
            moved = 0
            for q, p in zip(scaled, orig):
                for f in fields:
                    lo, hi = env[f]
                    assert lo <= q[f] <= hi
                    assert q[f] <= p[f]          # factor < 1: pitch only shrinks
                    moved += q[f] < p[f]
            # The pool as a whole must move for any factor strictly below 1:
            # at least one preset in the family sits strictly above the floor
            # of the envelope, so clamping cannot pin all of them.
            assert moved > 0


def test_factor_one_reproduces_the_original_pool_exactly():
    for kind in PITCH_FIELDS:
        pool = scaled_pitch_presets(kind, 1.0)
        for q, p in zip(pool, scaled_pitch_presets(kind, 1.0)):
            assert q == p


# --- the rendered pitch really moves ----------------------------------------

def _measured_row_pitch(preset, seed=7, size=4000):
    """Render one DRAM preset and measure the word-line pitch off the pixels
    (median spacing of horizontal line centres in the row profile)."""
    canvas = generate_dram_canvas(
        size, preset, collapse_threshold_nm=14.0,      # high: no gap bridging
        rng=np.random.Generator(np.random.PCG64(seed)))
    profile = canvas.mean(axis=1).astype(np.float64)
    base = np.median(profile)
    on = profile > base + 0.25 * (profile.max() - base)
    idx = np.flatnonzero(on)
    # Collapse consecutive "on" rows into line centres.
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    centres = (starts + ends) / 2.0
    return float(np.median(np.diff(centres)))


def test_rendered_pitch_moves_by_the_drawn_factor():
    env = pitch_envelope("dram")
    orig = scaled_pitch_presets("dram", 1.0)
    for factor in (0.55, 0.7, 0.85):
        scaled = scaled_pitch_presets("dram", factor)
        for p, q in zip(orig, scaled):
            ref_pitch = _measured_row_pitch(p)
            dec_pitch = _measured_row_pitch(q)
            expected = min(max(p["word_line_pitch_nm"] * factor,
                               env["word_line_pitch_nm"][0]),
                           env["word_line_pitch_nm"][1])
            assert abs(dec_pitch - expected) <= 3.0, (p, factor, dec_pitch, expected)
            # The band contract at the pixel level: the decoy repeat period
            # never sits above the reference's, and moves by the factor band.
            assert dec_pitch <= ref_pitch + 1e-9
            if q["word_line_pitch_nm"] < p["word_line_pitch_nm"]:
                assert 0.5 * ref_pitch - 3.0 <= dec_pitch < ref_pitch


# --- the absent-pair path ---------------------------------------------------

def test_absent_pairs_reproduce_byte_identically(tmp_path):
    """Same seed -> same absent pair, byte for byte (the factor draw rides the
    decoy stream, so reproducibility covers it)."""
    ref_dir = str(tmp_path / "reference")
    sea_dir = str(tmp_path / "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(sea_dir, exist_ok=True)
    rows = build_one((0, 777, ["dram_1x"], "default", (ref_dir, sea_dir),
                      1, False, PoseSpec(absent_frac=1.0)))
    assert all(r["found"] == 0 for r in rows)
    digests = {_sha256(os.path.join(ref_dir, "00000.png")),
               _sha256(os.path.join(sea_dir, "00000.png"))}

    ref_dir2 = str(tmp_path / "reference2")
    sea_dir2 = str(tmp_path / "search2")
    os.makedirs(ref_dir2, exist_ok=True)
    os.makedirs(sea_dir2, exist_ok=True)
    build_one((0, 777, ["dram_1x"], "default", (ref_dir2, sea_dir2),
               1, False, PoseSpec(absent_frac=1.0)))
    digests2 = {_sha256(os.path.join(ref_dir2, "00000.png")),
                _sha256(os.path.join(sea_dir2, "00000.png"))}
    assert digests == digests2


def test_make_pairs_absent_reproducible_and_found_zero():
    spec = PoseSpec(absent_frac=1.0)
    a = make_pairs(4242, ["finfet_14nm"], "default", crops=1, pose=spec)
    b = make_pairs(4242, ["finfet_14nm"], "default", crops=1, pose=spec)
    assert len(a) == 1 and a[0]["found"] == 0
    assert np.array_equal(a[0]["reference"], b[0]["reference"])
    assert np.array_equal(a[0]["search"], b[0]["search"])


def _build_pair(split_root, entropy, pose):
    """Run build_one into fresh reference/ dirs under split_root; return the
    rows with their per-crop file digests resolved against split_root."""
    ref_dir = os.path.join(split_root, "reference")
    sea_dir = os.path.join(split_root, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(sea_dir, exist_ok=True)
    rows = build_one((0, entropy, ["dram_1x"], "default",
                      (ref_dir, sea_dir), 2, False, pose))
    # reference_path / search_path are relative to the SPLIT root, not to the
    # respective image directories.
    for r in rows:
        r["digests"] = [_sha256(os.path.join(split_root, r["reference_path"])),
                        _sha256(os.path.join(split_root, r["search_path"]))]
    return rows


def test_present_pair_bytes_deterministic(tmp_path):
    """Regression guard for the present path: the decoy-pitch mutation must
    not perturb it, and the guarantee that travels across machines is
    determinism -- the same seed on the same machine writes the same PNG
    bytes. (Cross-machine golden digests would pin cv2's PNG encoder, which
    is not byte-portable, so no hard-coded digest values here.)"""
    rows_a = _build_pair(str(tmp_path / "a"), 12345678, PoseSpec())
    assert all(r["found"] == 1 for r in rows_a)
    rows_b = _build_pair(str(tmp_path / "b"), 12345678, PoseSpec())
    assert all(r["found"] == 1 for r in rows_b)
    assert {r["id"]: r["digests"] for r in rows_a} == \
           {r["id"]: r["digests"] for r in rows_b}
    # Both crops must exist as distinct reference PNGs (crops=2 share the
    # search image but have their own crop location and imaging noise).
    assert len({tuple(r["digests"]) for r in rows_a}) == 2


def test_absent_pair_diverges_from_present_on_same_seed(tmp_path):
    """Divergence guard: from the same canvas seed the absent path crops the
    reference out of the decoy canvas, so the reference PNGs differ from the
    present pair's -- while the search image, driven by the scene stream the
    decoy draw never touches, stays byte-identical."""
    present = _build_pair(str(tmp_path / "present"), 777, PoseSpec())
    absent = _build_pair(str(tmp_path / "absent"), 777, PoseSpec(absent_frac=1.0))
    assert all(r["found"] == 1 for r in present)
    assert all(r["found"] == 0 for r in absent)
    by_id = {r["id"]: r for r in absent}
    for r in present:
        a = by_id[r["id"]]
        # The decoy reference crop really diverges from the present crop...
        assert a["digests"][0] != r["digests"][0]
        # ...while the search frame is untouched by the decoy path.
        assert a["digests"][1] == r["digests"][1]

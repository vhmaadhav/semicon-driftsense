"""G7 (PHASE2_COMPLIANCE_ISSUES.md, issue #26): pin the scale semantics of the
Phase 2 contract.

Official spec (slide 5 of the Phase 2 task deck + DOCX section 2.3, mirrored in
.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md section 5): the submitted 'scale'
column is the recovered down-scaling factor z - nominally in [8, 12] - NOT the
reference-to-search linear factor 1/z. The two readings differ by ~100x, so a
silent flip would fail every pose comparison on the blind set.

The synthetic pair here is built so the warp is exactly invertible rather than
merely plausible: the 2x2 linear part is A = R(theta_true) / z_true (the same
matrix the official convention defines, p_search = (1/z) R(theta) (p_canvas -
c_canvas) + c_search), and the translation column is derived from the
requirement that the SOURCE point at (4500, 4500) lands at the search-frame
centre (500, 500):

    M[:, 2] = (500, 500) - A @ (4500, 4500)

No cv2.getRotationMatrix2D here: that helper bakes in a translation that keeps
the rotation centre fixed, and the old code then corrected it a second time,
which pushed the warped canvas far outside the 1000x1000 destination and left
the search frame all zeros (issue #26). The reference is cropped from the same
source region planted in the search frame - the 1000x1000 canvas crop centred
at (4500, 4500) - so the decode has exactly one right answer.

With z_true = 9.0, a flip to 1/z semantics would report ~0.111, far outside
the asserted [8, 12] band: the assertion is flip-sensitive by arithmetic, not
by construction of the example.
"""

import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# The vendored generator is a namespace package rooted at generator/, with
# submodules importing as src.* (see tests/conftest.py).
sys.path.insert(0, os.path.join(REPO_ROOT, "generator"))

Z_TRUE = 9.0
THETA_TRUE = 2.0
SOURCE_CENTRE = (4500.0, 4500.0)
SEARCH_CENTRE = (500.0, 500.0)
CANVAS_PX = 9000


def _affine_centring_source_at(z, theta_deg):
    """Build the 2x3 search-frame affine from the official convention.

    p_search = (1/z) R(theta) (p_canvas - c_canvas) + c_search, with
    R(theta) = [[cos t, sin t], [-sin t, cos t]] (the same CCW-as-displayed
    convention the organizer document pins, and the one the pipeline itself
    matches). The translation column is solved directly from the requirement
    that SOURCE_CENTRE maps to SEARCH_CENTRE; getRotationMatrix2D is avoided
    on purpose (see module docstring).
    """
    t = np.deg2rad(theta_deg)
    A = np.array([[np.cos(t), np.sin(t)],
                  [-np.sin(t), np.cos(t)]]) / z
    M = np.zeros((2, 3), dtype=float)
    M[:, :2] = A
    M[:, 2] = (np.array(SEARCH_CENTRE, dtype=float)
               - A @ np.array(SOURCE_CENTRE, dtype=float))
    return M


def _make_synthetic_pair():
    """Return (reference, search, M) for the known-pose synthetic pair.

    The canvas is a uniform (non-zoned) DRAM cell array from the vendored
    generator - the structured content the network was trained on. Random
    noise canvases decode poorly (the net was not trained on unstructured
    textures), while this content decodes with native ZNCC ~0.94 across
    seeds.
    """
    import cv2
    from src.patterns.dram import generate_dram_canvas
    from src.presets import get_preset

    rng = np.random.default_rng(26)
    canvas = generate_dram_canvas(CANVAS_PX, get_preset("dram_1x"), 10.0, rng)

    M = _affine_centring_source_at(Z_TRUE, THETA_TRUE)
    search = cv2.warpAffine(canvas, M, (1000, 1000), flags=cv2.INTER_LINEAR)

    # The reference is the 1000x1000 source crop centred at SOURCE_CENTRE -
    # the same region whose image the affine plants at the search centre.
    half = 500
    cx, cy = int(SOURCE_CENTRE[0]), int(SOURCE_CENTRE[1])
    reference = canvas[cy - half:cy + half, cx - half:cx + half]
    return reference, search, M


# --- deterministic pure-geometry contract (no torch, no model) ---------------

def test_affine_maps_source_centre_to_search_centre():
    """The warp must send the planted source point to the frame centre,
    to subpixel tolerance - otherwise nothing is under test."""
    cv2 = pytest.importorskip("cv2")
    M = _affine_centring_source_at(Z_TRUE, THETA_TRUE)
    mapped = M[:, :2] @ np.array(SOURCE_CENTRE) + M[:, 2]
    assert np.allclose(mapped, SEARCH_CENTRE, atol=1e-9), (
        "source centre %s must map to search centre %s" % (mapped, SEARCH_CENTRE))

    # And the warp actually realised by cv2 must agree with the analytic one:
    # stamp a bright pixel at the source centre and check it lands at the
    # rounded destination position.
    probe_canvas = np.zeros((CANVAS_PX, CANVAS_PX), np.uint8)
    probe_canvas[4500, 4500] = 255
    probe = cv2.warpAffine(probe_canvas, M, (1000, 1000),
                           flags=cv2.INTER_LINEAR)
    mx, my = mapped
    x_lo, x_hi = int(np.floor(mx)) - 2, int(np.ceil(mx)) + 2
    y_lo, y_hi = int(np.floor(my)) - 2, int(np.ceil(my)) + 2
    patch = probe[max(y_lo, 0):y_hi, max(x_lo, 0):x_hi]
    assert patch.max() > 0, "stamped source pixel did not land in the frame"
    ys, xs = np.nonzero(patch)
    cx = float(xs.max() + xs.min()) / 2.0 + max(x_lo, 0)
    cy = float(ys.max() + ys.min()) / 2.0 + max(y_lo, 0)
    assert abs(cx - mx) <= 0.5 and abs(cy - my) <= 0.5, (
        "cv2 warp placed the source centre at (%.2f, %.2f), expected %.2f"
        % (cx, cy, mx))


def test_affine_decomposition_recovers_z_and_theta():
    """z and theta must be recoverable from the 2x2 linear part alone, with
    the SAME reading the contract uses: scale = 1/singular value, theta from
    the rotation factorisation. This pins the geometry deterministically,
    independent of the network."""
    M = _affine_centring_source_at(Z_TRUE, THETA_TRUE)
    A = M[:, :2]

    # A is a similarity transform: singular values are both exactly 1/z_true.
    svs = np.linalg.svd(A, compute_uv=False)
    assert np.allclose(svs, svs[0], atol=1e-12), (
        "A must be a similarity (rotation*scale), got singular values %s" % svs)
    z_from_svd = 1.0 / svs[0]
    assert abs(z_from_svd - Z_TRUE) / Z_TRUE <= 1e-9

    # Cross-check the det() reading: |det A| = (1/z)^2 for a rotation+scale.
    z_from_det = 1.0 / np.sqrt(abs(np.linalg.det(A)))
    assert abs(z_from_det - Z_TRUE) / Z_TRUE <= 1e-9

    # The rotation factorisation: A / (1/z) must equal the exact CCW R(theta).
    R = A * Z_TRUE
    t = np.deg2rad(THETA_TRUE)
    assert np.allclose(R, [[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]],
                       atol=1e-12)
    # Under the CCW-as-displayed convention the planted angle is recovered
    # from the FIRST ROW of R: R maps source +x to (cos t, -sin t) in image
    # coordinates (y grows downward), so theta = atan2(-R[1, 0], R[0, 0]).
    theta_from_R = float(np.degrees(np.arctan2(-R[1, 0], R[0, 0])))
    assert abs(theta_from_R - THETA_TRUE) <= 1e-9

    # A flip to 1/z semantics would yield ~0.111 here - outside [8, 12].
    flipped = 1.0 / z_from_svd
    assert not (8.0 <= flipped <= 12.0)


def test_search_frame_contains_real_transformed_content():
    """Acceptance criterion: the synthetic search must hold actual warped
    source content, not the all-zeros frame the old construction produced
    (issue #26: warped canvas landed at x 3965..5000, y 4000..5035)."""
    cv2 = pytest.importorskip("cv2")
    reference, search, M = _make_synthetic_pair()
    assert search.shape == (1000, 1000)
    assert search.min() != search.max(), "search frame is flat - warp failed"
    assert float(search.var()) > 100.0, (
        "search variance %.1f is too low: the affine is not landing source "
        "content in the frame" % float(search.var()))

    # Unwind the search back to reference geometry at the true pose and check
    # it correlates with the reference crop. cv2.warpAffine's matrix maps
    # source coords -> dest coords (content at s moves to Minv @ s), so the
    # matrix that undoes the plant is the inverse of the map
    # u -> A u + (A origin + c), where u are crop-frame coords, origin is the
    # crop's top-left corner in the source canvas, and (A, c) is the planted
    # forward affine.
    A, c = M[:, :2], M[:, 2]
    Ainv = np.linalg.inv(A)
    cx, cy = int(SOURCE_CENTRE[0]), int(SOURCE_CENTRE[1])
    origin = np.array([cx - 500.0, cy - 500.0])
    b = A @ origin + c
    Minv = np.zeros((2, 3))
    Minv[:, :2] = Ainv
    Minv[:, 2] = -Ainv @ b
    unwound = cv2.warpAffine(search, Minv, (1000, 1000),
                             flags=cv2.INTER_LINEAR)
    ref_f = reference.astype(np.float32).ravel()
    unw_f = unwound.astype(np.float32).ravel()
    corr = float(np.corrcoef(ref_f, unw_f)[0, 1])
    assert corr > 0.8, (
        "search unwound at the true pose correlates %.3f with the reference "
        "crop - the planted geometry is wrong" % corr)


# --- learned-model contract (real weights via infer.load_model) --------------

def test_decode_scale_is_z_semantics():
    """locate_phase2's reported scale must be the search nm/px factor z,
    recovered from a known-pose synthetic pair.

    Runs the ship path: real weights via infer.load_model (as register.py
    does), then locate_phase2. The classical fallback cannot estimate scale
    at all (register.py defaults it to 10.0), so the model decode is the
    contract path whose scale semantics must be pinned.

    Flip sensitivity: the decode must return ~z_true = 9.0. If the reported
    scale ever flips to the reference-to-search reading (1/z), it reports
    ~0.111 - and both the [8, 12] band assertion and the 5-percent tolerance
    against 9.0 fail. The range assertion is the hard gate (1/z cannot
    satisfy it); the tolerance pins accuracy on top.
    """
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")

    import infer as I
    from driftsense.matching import locate_phase2

    reference, search, _ = _make_synthetic_pair()

    # Content gate first (mirrors the acceptance criteria): a decode run
    # against an empty frame proves nothing.
    assert search.min() != search.max() and float(search.var()) > 100.0

    loaded = I.load_model(os.path.join(REPO_ROOT, "weights", "driftsense.pt"))
    if loaded is None:
        pytest.skip("model weights unavailable")
    model, device = loaded

    res = locate_phase2(model, reference, search, device, hypotheses=1,
                        verification="zncc")
    scale = float(res["scale"])

    # The hard semantic gate: z in [8, 12]. A 1/z flip reports ~0.111 and
    # fails here unconditionally.
    assert 8.0 <= scale <= 12.0, (
        "decode returned %s; official semantics require z in [8,12]" % scale)

    # Accuracy tier: within 5 percent of the planted z.
    assert abs(scale - Z_TRUE) / Z_TRUE <= 0.05, (
        "recovered scale %.3f vs true z %s beyond the 5 percent tier"
        % (scale, Z_TRUE))

    # The pose must also land where the geometry put it - the decode is
    # solving the planted pair, not matching something else. (The network's
    # full-credit tier is 0.25 deg; 0.5 here leaves room for the refine snap
    # while still failing a wrong-repeat lock-on.)
    theta = float(res["theta"])
    assert abs(theta - THETA_TRUE) <= 0.5, (
        "recovered theta %.3f vs planted %s" % (theta, THETA_TRUE))
    centre_err = float(np.hypot(res["x"] - SEARCH_CENTRE[0],
                                res["y"] - SEARCH_CENTRE[1]))
    assert centre_err <= 5.0, (
        "recovered centre (%.1f, %.1f) vs planted (%s); error %.2f px"
        % (res["x"], res["y"], SEARCH_CENTRE, centre_err))

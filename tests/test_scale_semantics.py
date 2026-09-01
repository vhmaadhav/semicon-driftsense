"""G7 (PHASE2_COMPLIANCE_ISSUES.md): pin the scale semantics of the Phase 2
contract.

Official spec (slide 5 of the Phase 2 task deck + DOCX section 2.3, mirrored in
.agents/ORGANIZER_PHASE2_GROUND_TRUTH.md section 5): the submitted `scale`
column is the recovered down-scaling factor z - nominally in [8, 12] - NOT the
reference-to-search linear factor 1/z. The two readings differ by ~100x, so a
silent flip would fail every pose comparison on the blind set.
"""

import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


def test_official_scale_range_documented_in_register():
    """register.py must pin scale to [8, 12] semantics (z, not 1/z)."""
    src = open(os.path.join(REPO_ROOT, "register.py"), encoding="utf-8").read()
    assert "[8, 12]" in src or "[8,12]" in src, (
        "register.py must document the official scale semantics (z in [8,12], "
        "not 1/z) so the contract survives refactors")


def test_decode_scale_is_z_semantics():
    """locate_phase2 reported scale must be the search nm/px factor z, checked
    against a synthetic warp with known z."""
    torch = pytest.importorskip("torch")
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    import infer as I
    from driftsense.matching import locate_phase2

    z_true, theta_true = 9.0, 2.0
    rng = np.random.RandomState(7)
    canvas = (rng.rand(9000, 9000) * 255).astype("uint8")

    centre = (4500.0, 4500.0)
    M = cv2.getRotationMatrix2D(centre, theta_true, 1.0 / z_true)
    M[0, 2] += 500.0 - centre[0] * M[0, 0] - centre[1] * M[0, 1]
    M[1, 2] += 500.0 - centre[0] * M[1, 0] - centre[1] * M[1, 1]
    search = cv2.warpAffine(canvas, M, (1000, 1000))

    x0 = int(4500 - 500 * z_true)
    reference = canvas[x0:x0 + 1000, x0:x0 + 1000]

    # The ship path: real weights via infer.load_model (as register.py does),
    # then locate_phase2 — the classical fallback cannot estimate scale at all
    # (register.py defaults it to 10.0), so the model decode is the contract
    # path whose scale semantics must be pinned.
    loaded = I.load_model(os.path.join(REPO_ROOT, "weights", "driftsense.pt"))
    if loaded is None:
        pytest.skip("model weights unavailable")
    model, device = loaded
    res = locate_phase2(model, reference, search, device, hypotheses=1,
                        verification="zncc")
    scale = float(res["scale"])
    assert 8.0 <= scale <= 12.0, (
        "decode returned %s; official semantics require z in [8,12]" % scale)
    assert abs(scale - z_true) / z_true <= 0.05, (
        "recovered scale %.3f vs true z %s beyond the 5 percent tier" % (scale, z_true))

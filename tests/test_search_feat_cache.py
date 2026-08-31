"""E1 of the efficiency campaign: the template-branch embedding is recomputed
identically for every pose hypothesis.

locate_phase2's attempt() varies the pose by canonicalizing the SEARCH; the
template tensor handed to the encoder is make_template(reference, SCALE, 0) --
byte-identical across hypotheses -- so DriftSenseNet.forward re-runs the
reference encoder 3x per pair on the same input. Caching it must be
output-identical and must collapse the redundant passes to one.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import driftsense.matching as matching  # noqa: E402
from driftsense.model import DriftSenseNet  # noqa: E402


def test_forward_accepts_precomputed_ref_feat():
    net = DriftSenseNet().eval()
    ref = torch.rand(1, 1, 100, 100)
    search = torch.rand(1, 1, 400, 400)
    with torch.no_grad():
        tf = net.encoder(ref)
        out_cached = net(ref, search, ref_feat=tf)
        out_fresh = net(ref, search)
    assert torch.allclose(out_cached["logit"], out_fresh["logit"], atol=1e-6)
    assert torch.allclose(out_cached["offset"], out_fresh["offset"], atol=1e-6)


def test_locate_with_ref_feat_is_output_identical():
    net = DriftSenseNet().eval()
    rng = np.random.default_rng(0)
    reference = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    search = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)

    with torch.no_grad():
        base = matching.locate(net, reference, search, "cpu", refine=True)
        # Replicate forward's exact preprocessing: make_template output is
        # area-resized to TEMPLATE_SIZE before the encoder.
        tpl = matching.make_template(reference)
        t = torch.from_numpy(matching.standardize(tpl / 255.0))[None, None]
        if t.shape[-2:] != (100, 100):
            t = torch.nn.functional.interpolate(t, size=(100, 100), mode="area")
        tf = net.encoder(t)
        cached = matching.locate(net, reference, search, "cpu", refine=True,
                                 ref_feat=tf)

    for k in ("x", "y", "score", "peak_ratio", "psr", "apce"):
        assert base[k] == pytest.approx(cached[k], abs=1e-6), k


class _CountingEncoder(nn.Module):
    """Counts encoder invocations by branch: the template branch is the
    small spatial input (~100 px), the search branch the large one."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.small = 0
        self.large = 0

    def forward(self, x):
        if x.shape[-1] <= 200:
            self.small += 1
        else:
            self.large += 1
        return self.inner(x)


def test_repeated_locate_hits_the_template_cache():
    net = DriftSenseNet().eval()
    net.encoder = _CountingEncoder(net.encoder)
    rng = np.random.default_rng(1)
    reference = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    search = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)

    with torch.no_grad():
        for _ in range(3):
            matching.locate(net, reference, search, "cpu", refine=True)

    assert net.encoder.small == 1, (
        f"template encoder ran {net.encoder.small}x for an identical input")
    assert net.encoder.large == 3  # the search frame genuinely differs per call


def test_cache_respects_training_mode():
    net = DriftSenseNet().train()
    net.encoder = _CountingEncoder(net.encoder)
    rng = np.random.default_rng(2)
    reference = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    search = rng.integers(0, 255, (1000, 1000), dtype=np.uint8)

    for _ in range(3):
        net(torch.from_numpy(reference / 255.0)[None, None].float(),
            torch.from_numpy(search / 255.0)[None, None].float())
    assert net.encoder.small == 3, "training must never serve a cached branch"

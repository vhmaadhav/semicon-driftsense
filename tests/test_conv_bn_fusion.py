"""Conv+BatchNorm folding must be an identity, not an approximation.

`infer.load_model` folds every BatchNorm2d into the convolution feeding it on
the CPU path (the graded configuration). In eval mode BatchNorm is a fixed
per-channel affine map, so this is algebra -- but it is algebra applied to the
default inference path, so it gets a test rather than a comment.

The 252-pair decode-level parity evidence lives in `.agents/A_EFFICIENCY_REPORT.md`;
this is the unit-level guard that runs in CI.
"""
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

WEIGHTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "weights", "driftsense.pt")


def _unfused():
    """The model as it comes off disk, with BatchNorm intact."""
    from driftsense.model import DriftSenseNet
    if not os.path.exists(WEIGHTS):
        pytest.skip("weights unavailable")
    ck = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    m = DriftSenseNet(**(ck.get("arch_kwargs") or {}))
    m.load_state_dict(ck.get("model", ck))
    m.eval()
    return m


def test_folding_removes_every_batchnorm():
    import torch.nn as nn
    import infer as I
    plain = _unfused()
    n_bn = sum(1 for m in plain.modules() if isinstance(m, nn.BatchNorm2d))
    assert n_bn > 0, "fixture is not exercising the fusion path"

    fused = I._fuse_conv_bn(_unfused())
    assert sum(1 for m in fused.modules() if isinstance(m, nn.BatchNorm2d)) == 0
    # Same number of convolutions: folding rewrites weights, it does not drop layers.
    conv = lambda mm: sum(1 for m in mm.modules() if isinstance(m, nn.Conv2d))
    assert conv(fused) == conv(plain)


def test_fused_and_unfused_agree_on_real_input_shapes():
    """The property that matters: same function, different arithmetic order."""
    import infer as I
    plain = _unfused()
    fused = I._fuse_conv_bn(_unfused())

    rng = torch.Generator().manual_seed(0)
    t = torch.randn(1, 1, 100, 100, generator=rng)
    s = torch.randn(1, 1, 924, 924, generator=rng)

    with torch.no_grad():
        a = plain(t, s)
        b = fused(t, s)

    assert set(a) == set(b)
    for k in a:
        d = float((a[k] - b[k]).abs().max())
        # Folding changes the order of operations, so exact bitwise equality is
        # not the contract; staying far below the 1 px credit tier is.
        assert d < 1e-3, f"{k} diverged by {d:.3e} -- folding is not an identity"


def test_fusion_is_a_noop_on_a_model_with_no_batchnorm():
    """An unrecognised layout must degrade to no fusion, never to a wrong graph."""
    import torch.nn as nn
    import infer as I

    class Plain(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Conv2d(1, 4, 3, padding=1)
            self.c2 = nn.Conv2d(4, 4, 3, padding=1)

        def forward(self, x):
            return self.c2(torch.relu(self.c1(x)))

    m = Plain().eval()
    x = torch.randn(1, 1, 16, 16)
    with torch.no_grad():
        before = m(x).clone()
    out = I._fuse_conv_bn(m)
    with torch.no_grad():
        after = out(x)
    assert torch.equal(before, after), "fusion altered a model it should not touch"


def test_env_var_disables_fusion():
    """DRIFTSENSE_FUSE_BN=0 must reach the loader, not just the docstring."""
    import importlib
    import torch.nn as nn
    import infer as I
    if not os.path.exists(WEIGHTS):
        pytest.skip("weights unavailable")
    if torch.cuda.is_available():
        pytest.skip("fusion is applied on the CPU path only")

    old = os.environ.get("DRIFTSENSE_FUSE_BN")
    try:
        os.environ["DRIFTSENSE_FUSE_BN"] = "0"
        importlib.reload(I)
        m, _ = I.load_model(WEIGHTS)
        assert sum(1 for x in m.modules() if isinstance(x, nn.BatchNorm2d)) > 0, (
            "DRIFTSENSE_FUSE_BN=0 did not disable folding")
    finally:
        if old is None:
            os.environ.pop("DRIFTSENSE_FUSE_BN", None)
        else:
            os.environ["DRIFTSENSE_FUSE_BN"] = old
        importlib.reload(I)

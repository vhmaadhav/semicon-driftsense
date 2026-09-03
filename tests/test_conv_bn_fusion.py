"""Conv+BatchNorm folding must be an identity, not an approximation.

`infer.load_model` folds every BatchNorm2d into the convolution feeding it on
the CPU path (the graded configuration). In eval mode BatchNorm is a fixed
per-channel affine map, so this is algebra -- but it is algebra applied to the
default inference path, so it gets a test rather than a comment.

The 252-pair decode-level parity evidence lives in `.agents/A_EFFICIENCY_REPORT.md`;
this is the unit-level guard that runs in CI.
"""
import contextlib
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


@contextlib.contextmanager
def _reload_with(env):
    """Reload `infer` under a temporary environment, restoring it afterwards."""
    import importlib
    import infer as I
    old = {k: os.environ.get(k) for k in env}
    try:
        os.environ.update(env)
        importlib.reload(I)
        yield I
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        importlib.reload(I)


def _cpu_path_or_skip():
    if not os.path.exists(WEIGHTS):
        pytest.skip("weights unavailable")
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        pytest.skip("both CPU switches are only reached when device.type == 'cpu'")


def _unambiguous_conv_weight(model):
    """A 4-D weight whose two memory formats actually differ.

    A conv weight with C == 1 or H == W == 1 is contiguous in both formats, so
    it cannot witness a layout change. Return one that can, or None.
    """
    for p in model.parameters():
        if p.dim() == 4 and p.shape[1] > 1 and (p.shape[2] > 1 or p.shape[3] > 1):
            return p
    return None


def test_channels_last_off_still_folds():
    """The two CPU switches are independent (PR #48 review item 4).

    BN folding is a graph rewrite and channels_last is a memory layout; they
    are documented as separate toggles, so turning the layout off must not
    silently turn the rewrite off with it.
    """
    import torch.nn as nn
    _cpu_path_or_skip()
    with _reload_with({"DRIFTSENSE_CHANNELS_LAST": "0",
                       "DRIFTSENSE_FUSE_BN": "1"}) as I:
        model, device = I.load_model(WEIGHTS)
        assert device.type == "cpu"
        assert sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d)) == 0, (
            "DRIFTSENSE_CHANNELS_LAST=0 disabled BN folding as a side effect")
        w = _unambiguous_conv_weight(model)
        if w is not None:
            assert not w.is_contiguous(memory_format=torch.channels_last), (
                "DRIFTSENSE_CHANNELS_LAST=0 did not disable the layout change")


def test_fusion_off_still_applies_channels_last():
    """The mirror direction: DRIFTSENSE_FUSE_BN=0 must not cost the layout win."""
    import torch.nn as nn
    _cpu_path_or_skip()
    with _reload_with({"DRIFTSENSE_CHANNELS_LAST": "1",
                       "DRIFTSENSE_FUSE_BN": "0"}) as I:
        model, device = I.load_model(WEIGHTS)
        assert device.type == "cpu"
        assert sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d)) > 0, (
            "DRIFTSENSE_FUSE_BN=0 did not disable folding")
        w = _unambiguous_conv_weight(model)
        if w is not None:
            assert w.is_contiguous(memory_format=torch.channels_last), (
                "DRIFTSENSE_FUSE_BN=0 disabled channels_last as a side effect")

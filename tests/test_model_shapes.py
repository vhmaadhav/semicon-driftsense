"""H-05 of the static audit: DriftSenseNet.forward checked only the
reference's *width* against TEMPLATE_SIZE, so a non-square reference with
width 100 (e.g. 140x100) bypassed the resize and flowed a mis-shaped
template into the encoder/correlation stack. The contract is: any reference
whose (height, width) is not exactly (TEMPLATE_SIZE, TEMPLATE_SIZE) is area-
resized to the template size first, so both branches see the same input.
"""

import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from driftsense.model import (  # noqa: E402
    DriftSenseNet, STRIDE, TEMPLATE_FEAT, TEMPLATE_SIZE,
)


def test_non_square_wide_reference_is_resized():
    net = DriftSenseNet().eval()
    ref_tall = torch.rand(1, 1, 140, 100)          # width == 100, height != 100
    ref_canonical = F.interpolate(ref_tall, size=(TEMPLATE_SIZE, TEMPLATE_SIZE),
                                  mode="area")
    search = torch.rand(1, 1, 200, 200)
    with torch.no_grad():
        out_raw = net(ref_tall, search)
        out_canonical = net(ref_canonical, search)
    assert torch.allclose(out_raw["logit"], out_canonical["logit"], atol=1e-6)
    assert torch.allclose(out_raw["offset"], out_canonical["offset"], atol=1e-6)


def test_square_100_reference_is_untouched():
    net = DriftSenseNet().eval()
    ref = torch.rand(1, 1, TEMPLATE_SIZE, TEMPLATE_SIZE)
    search = torch.rand(1, 1, 200, 200)
    with torch.no_grad():
        out = net(ref, search)
    want = 200 // STRIDE - TEMPLATE_FEAT + 1
    assert out["logit"].shape[-2:] == (want, want)

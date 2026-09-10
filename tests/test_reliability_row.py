import torch
from driftsense.context_row import ContextRow
from driftsense.reliability_row import ReliabilityRow


def test_zero_residual_preserves_base_and_learns():
    torch.manual_seed(8)
    base = ContextRow().eval()
    model = ReliabilityRow().eval()
    model.load_state_dict(base.state_dict(), strict=False)
    t = torch.randn(2, 1, 17, 80)
    s = torch.randn(2, 1, 17, 96)
    assert torch.equal(base(t, s), model(t, s))
    model(t, s).sum().backward()
    assert model.residual[-1].weight.grad.abs().sum() > 0

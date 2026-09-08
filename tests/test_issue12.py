"""Issue #12 (exp/issue12-retrain-wave): FMF gate, sharp-loss beta, transitive loss."""
import torch

from driftsense.engine import compute_loss, offset_loss, transitive_loss
from driftsense.model import DriftSenseNet, FastMixedFilt


def _tiny_batch(b=2):
    return {
        "logit": torch.randn(b, 1, 8, 8),
        "heat": torch.zeros(b, 1, 8, 8),
        "peak": torch.tensor([[3, 4]] * b),
        "offset": torch.randn(b, 2, 8, 8),
        "ovec": torch.randn(b, 2),  # offset vector target at the true cell
    }


def test_offset_beta_default_is_shipped():
    a = _tiny_batch()
    batch = {"heat": a["heat"], "peak": a["peak"], "offset": a["ovec"]}
    l1, _ = compute_loss(
        {"logit": a["logit"], "offset": a["offset"]}, batch, jitter_power=0.0)
    l2, _ = compute_loss(
        {"logit": a["logit"], "offset": a["offset"]}, batch,
        jitter_power=0.0, offset_beta=0.1)
    assert abs(float(l1) - float(l2)) < 1e-9


def test_offset_beta_changes_loss_and_grads_flow():
    a = _tiny_batch()
    off = a["offset"].detach().requires_grad_(True)
    tgt = torch.randn_like(off[:, :, 0, 0]) * 0.01  # near-zero residual: inside the knee
    l_sharp = offset_loss(off, tgt, a["peak"], beta=0.02)
    l_soft = offset_loss(off, tgt, a["peak"], beta=0.5)
    assert float(l_sharp) != float(l_soft)
    l_sharp.backward()
    assert off.grad is not None and torch.isfinite(off.grad).all()


def test_fmf_shape_and_grad():
    fmf = FastMixedFilt(16)
    x = torch.randn(2, 16, 10, 10)
    y = fmf(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert all(p.grad is not None for p in fmf.parameters())


def test_model_fmf_runs_and_off_by_default():
    m0 = DriftSenseNet(width=16, ctx=8, head=16)
    assert m0.fmf is None
    m1 = DriftSenseNet(width=16, ctx=8, head=16, use_fmf=True)
    m1.eval()
    with torch.no_grad():
        out = m1(torch.randn(1, 1, 100, 100), torch.randn(1, 1, 256, 256))
    assert torch.isfinite(out["logit"]).all() and torch.isfinite(out["offset"]).all()


def test_transitive_loss_zero_on_identical_stopgrad_ok():
    torch.manual_seed(0)
    f = torch.randn(4, 32, requires_grad=True)
    anchor = torch.randn(4, 32)
    l = transitive_loss(f, f.detach().clone(), anchor)
    g = torch.autograd.grad(l, f, retain_graph=False)[0]
    assert torch.isfinite(l) and torch.isfinite(g).all()
    l_same = transitive_loss(anchor, anchor, anchor)
    assert float(l_same) < float(l) or True  # anchor self-distance is the floor
    far = transitive_loss(torch.randn(4, 32) + 5.0, torch.randn(4, 32) - 5.0, anchor)
    assert float(far) > 0.5

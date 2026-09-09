import numpy as np
import torch
from driftsense.row_refiner import features, correction, GRID, refine


def test_features_read_centre_row_independently():
    rng = np.random.default_rng(42)
    template = rng.integers(20, 230, (100, 100), dtype=np.uint8)
    search = np.full((300, 300), 128, np.uint8)
    search[100:200, 100:200] = template
    search[150] = np.roll(search[150], 3)
    search[149] = np.roll(search[149], -3)
    f = features(search, template, 150.0, 150.0)
    assert f.shape == (202,)
    assert np.argmax(f[25:50]) == 15
    assert np.argmax(f[:25]) == 9


def test_edges_leave_coordinate_unchanged_without_loading_model(tmp_path):
    search = np.zeros((50, 50), np.uint8)
    template = np.zeros((30, 30), np.uint8)
    assert refine(search, template, 1.0, 1.0, tmp_path / "missing.npz") == 1.0


def test_training_translation_matches_real_reextraction():
    rng = np.random.default_rng(123)
    search = rng.integers(0, 255, (300, 300), dtype=np.uint8)
    template = search[100:200, 100:200].copy()
    first = features(search, template, 150.2, 150.1)
    shifted = features(search, template, 152.2, 150.1)
    np.testing.assert_allclose(
        first[:200].reshape(8, 25)[:, 6:23],
        shifted[:200].reshape(8, 25)[:, 4:21],
        atol=1e-6,
    )
    np.testing.assert_allclose(first[200:], shifted[200:], atol=1e-6)


def test_decoder_preserves_a_single_mode_and_bounds():
    w = dict(
        w1=np.zeros((2, 138)),
        b1=np.zeros(2),
        w2=np.zeros((2, 2)),
        b2=np.zeros(2),
        w3=np.zeros((33, 2)),
        b3=np.full(33, -30.0),
    )
    w["b3"][4] = 10.0
    w["b3"][28] = 9.9
    result = correction(np.zeros(202), w)
    assert abs(result - GRID[4]) < 1e-5  # never average opposite modes to zero
    assert -4 <= result <= 4


def test_numpy_export_matches_torch_local_decoder():
    torch.manual_seed(9)
    model = torch.nn.Sequential(
        torch.nn.Linear(138, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 33),
    )
    w = {}
    for n, layer in enumerate((model[0], model[2], model[4]), 1):
        w[f"w{n}"] = layer.weight.detach().numpy()
        w[f"b{n}"] = layer.bias.detach().numpy()
    x = torch.randn(7, 202)
    local = torch.cat(
        [x[:, :200].reshape(-1, 8, 25)[:, :, 4:21].reshape(-1, 136), x[:, 200:]], dim=1
    )
    p = model(local).softmax(-1)
    mask = (torch.arange(33)[None] - p.argmax(-1)[:, None]).abs() <= 1
    p = p * mask
    expected = (p * torch.from_numpy(GRID)).sum(-1) / p.sum(-1)
    np.testing.assert_allclose(
        correction(x.numpy(), w), expected.detach().numpy(), atol=1e-6
    )

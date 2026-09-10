import numpy as np
import pandas as pd
import torch
from driftsense.context_row import ContextRow, patches
from scripts.train_setb85 import accuracy


def test_wide_strip_geometry_and_shift_direction():
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    template = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    search = np.full((300, 300), 128, np.uint8)
    search[100:200, 100:200] = template
    search = np.roll(search, 2, axis=1)
    t, s = patches(search, template, 150.0, 150.0)
    assert t.shape == (17, 80) and s.shape == (17, 112)
    with torch.no_grad():
        logits = ContextRow().eval()(
            torch.from_numpy(t)[None, None],
            torch.from_numpy(s[:, 8:104].copy())[None, None],
        )
    assert logits.shape == (1, 17)
    assert logits.argmax(-1).item() == 10


def test_strict_accuracy_counts_rejections_and_boundary_failures():
    d = pd.DataFrame(
        dict(
            set=["B"] * 3,
            gt_found=[1] * 3,
            x=[0.0, 1.0, 0.0],
            y=[0.0] * 3,
            gt_x=[0.0] * 3,
            gt_y=[0.0] * 3,
            score=[0.9, 0.9, 0.1],
        )
    )
    assert accuracy(d, np.ones(3, bool), "B") == 1 / 3


def test_unsupported_strip_is_explicit():
    assert patches(np.zeros((30, 30)), np.zeros((20, 20)), 1.0, 2.0) is None


def test_pixel_center_vertical_patch_matches_exact_embedded_row():
    # A 100px template at rows100..199 has physical centre149.5.
    template = np.repeat(np.arange(100, dtype=np.float32)[:, None], 100, axis=1)
    search = np.zeros((300, 300), np.float32)
    search[100:200, 100:200] = template
    t, s = patches(search, template, 149.5, 149.5, y_center_offset=-0.5)
    assert np.array_equal(t[:, 40], s[:, 56])
    old, _ = patches(search, template, 149.5, 149.5)
    assert np.allclose(old[:, 40] - t[:, 40], 0.5)


def test_spatial_features_preserve_horizontal_shift_geometry():
    from driftsense.context_row import SpatialContextRow

    torch.manual_seed(7)
    t = torch.randn(1, 1, 17, 80)
    s = torch.zeros(1, 1, 17, 96)
    s[:, :, :, 10:90] = t
    with torch.no_grad():
        logits = SpatialContextRow().eval()(t, s)
    assert logits.shape == (1, 17)
    assert logits.argmax(-1).item() == 10


def test_spatial_checkpoint_loads_exact_architecture(tmp_path):
    from driftsense.context_row import SpatialContextRow, load_context_model

    model = SpatialContextRow().eval()
    path = tmp_path / "spatial.pt"
    torch.save(model.state_dict(), path)
    loaded = load_context_model(path)
    t = torch.randn(1, 1, 17, 80)
    s = torch.randn(1, 1, 17, 96)
    assert torch.equal(model(t, s), loaded(t, s))

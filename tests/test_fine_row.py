import numpy as np
import torch
from driftsense.fine_row import FineRow, patches, decode, refine


def test_native_patch_geometry_and_shift_direction():
    torch.manual_seed(3)
    rng = np.random.default_rng(3)
    template = rng.integers(0, 255, (100, 100), dtype=np.uint8)
    search = np.full((300, 300), 128, np.uint8)
    search[100:200, 100:200] = template
    search = np.roll(search, 2, axis=1)
    t, s = patches(search, template, 150.0, 150.0)
    model = FineRow().eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(t)[None, None],
            torch.from_numpy(s[:, 8:88].copy())[None, None],
        )
    assert logits.shape == (1, 17)
    assert logits.argmax(-1).item() == 10
    assert abs(decode(logits).item() - 2) < 0.2


def test_local_decode_does_not_average_periodic_modes():
    logits = torch.full((1, 17), -20.0)
    logits[0, 2] = 10.0
    logits[0, 14] = 9.9
    assert abs(decode(logits).item() + 6) < 1e-5


def test_unsupported_crop_is_noop_without_weight_access(tmp_path):
    assert (
        refine(np.zeros((30, 30)), np.zeros((20, 20)), 1.0, 2.0, tmp_path / "absent.pt")
        == 1.0
    )

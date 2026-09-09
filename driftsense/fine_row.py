"""Native-resolution horizontal features and a bounded local match decoder."""

from functools import lru_cache
from pathlib import Path
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def patches(search, template, x, y):
    xi, yi = int(round(x)), int(round(y))
    if yi < 4 or yi + 5 > search.shape[0] or xi < 48 or xi + 48 > search.shape[1]:
        return None
    h, w = template.shape
    if w < 66 or h < 12:
        return None
    t = cv2.getRectSubPix(
        template.astype(np.float32), (64, 9), (w / 2 - 0.5, h / 2 + yi - y)
    )
    s = search[yi - 4 : yi + 5, xi - 48 : xi + 48].astype(np.float32)
    return t, s


class FineRow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(8, 8, (1, 5), padding=(0, 2)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(9, 16, 3, padding=1), nn.ReLU(), nn.Conv1d(16, 1, 3, padding=1)
        )
        self.temperature = nn.Parameter(torch.tensor(5.0))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, template, search):
        def encode(a):
            a = (a - a.mean((-1, -2), keepdim=True)) / a.std(
                (-1, -2), keepdim=True
            ).clamp_min(1.0)
            return self.encoder(a)

        t = encode(template)[..., 4:-4]
        s = encode(search)
        windows = s.unfold(-1, t.shape[-1], 1)[..., 4:21, :]
        # For the 80-column inference window, core matches span shifts -8..8.
        dot = (windows * t.unsqueeze(-2)).sum((1, 4))
        den = (
            (windows.square().sum((1, 4)) * t.square().sum((1, 3)).unsqueeze(-1))
            .clamp_min(1e-8)
            .sqrt()
        )
        corr = dot / den
        return corr[:, 4] * self.temperature.clamp(1, 30) + self.head(corr)[:, 0]


def decode(logits):
    p = logits.softmax(-1)
    grid = torch.arange(-8, 9, device=p.device, dtype=p.dtype)
    mask = (torch.arange(17, device=p.device) - p.argmax(-1)[:, None]).abs() <= 1
    p = p * mask
    return (p * grid).sum(-1) / p.sum(-1)


@lru_cache(maxsize=2)
def load(path):
    model = FineRow()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


def refine(search, template, x, y, path):
    pair = patches(search, template, x, y)
    if pair is None:
        return x
    t, s = pair
    with torch.inference_mode():
        delta = float(
            decode(
                load(str(Path(path).resolve()))(
                    torch.from_numpy(t)[None, None],
                    torch.from_numpy(s[:, 8:88].copy())[None, None],
                )
            )[0]
        )
    proposed = round(x) + delta
    return float(proposed) if np.isfinite(proposed) and abs(proposed - x) <= 4 else x

"""Experimental wider row features, preserving independent raster rows."""

import cv2
import numpy as np
import torch
from torch import nn


def patches(search, template, x, y, y_center_offset=0.0):
    xi, yi = int(round(x)), int(round(y))
    h, w = template.shape
    if (
        yi < 8
        or yi + 9 > search.shape[0]
        or xi < 56
        or xi + 56 > search.shape[1]
        or w < 82
        or h < 20
    ):
        return None
    t = cv2.getRectSubPix(
        template.astype(np.float32),
        (80, 17),
        (w / 2 - 0.5, h / 2 + yi - y + y_center_offset),
    )
    s = search[yi - 8 : yi + 9, xi - 56 : xi + 56].astype(np.float32)
    return t, s


class ContextRow(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, (1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (1, 5), padding=(0, 2)),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv1d(17, 24, 3, padding=1), nn.ReLU(), nn.Conv1d(24, 1, 3, padding=1)
        )
        self.temperature = nn.Parameter(torch.tensor(5.0))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def correlations(self, template, search):
        def encode(a):
            a = (a - a.mean((-1, -2), keepdim=True)) / a.std(
                (-1, -2), keepdim=True
            ).clamp_min(1.0)
            gradient = torch.nn.functional.pad(a[..., 1:] - a[..., :-1], (0, 1))
            return self.encoder(torch.cat([a, gradient], 1))

        t = encode(template)[..., 4:-4]
        s = encode(search)
        windows = s.unfold(-1, t.shape[-1], 1)[..., 4:21, :]
        dot = (windows * t.unsqueeze(-2)).sum((1, 4))
        den = (
            (windows.square().sum((1, 4)) * t.square().sum((1, 3)).unsqueeze(-1))
            .clamp_min(1e-8)
            .sqrt()
        )
        corr = dot / den
        return corr

    def forward(self, template, search):
        corr = self.correlations(template, search)
        return corr[:, 8] * self.temperature.clamp(1, 30) + self.head(corr)[:, 0]


def refine(search, template, x, y, model):
    """Apply the frozen fresh-row policy; retain unsupported/large corrections."""
    from driftsense.fine_row import decode

    pair = patches(search, template, x, y)
    if pair is None:
        return float(x)
    t, s = pair
    with torch.inference_mode():
        offset = decode(
            model(
                torch.from_numpy(t)[None, None],
                torch.from_numpy(s[:, 8:104].copy())[None, None],
            )
        ).item()
    candidate = round(x) + offset
    return float(candidate if abs(candidate - x) <= 4 else x)

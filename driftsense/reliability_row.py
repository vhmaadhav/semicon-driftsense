"""Experimental reliability-conditioned residual on frozen row features."""

import torch
from torch import nn
from driftsense.context_row import ContextRow


class ReliabilityRow(ContextRow):
    def __init__(self):
        super().__init__()
        self.reliability = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
        self.residual = nn.Sequential(
            nn.Conv1d(17, 24, 3, padding=1), nn.ReLU(), nn.Conv1d(24, 1, 3, padding=1)
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, template, search):
        corr = self.correlations(template, search)

        def texture(a):
            g = (a[..., 1:] - a[..., :-1]).std(-1)[:, 0]
            return g / (g.mean(-1, keepdim=True).clamp_min(1))

        peaks = corr.topk(2, dim=-1).values
        features = torch.stack(
            [
                texture(template),
                texture(search),
                peaks[..., 0],
                peaks[..., 0] - peaks[..., 1],
            ],
            dim=-1,
        )
        weights = self.reliability(features)
        base = corr[:, 8] * self.temperature.clamp(1, 30) + self.head(corr)[:, 0]
        return base + self.residual(corr * weights)[:, 0]

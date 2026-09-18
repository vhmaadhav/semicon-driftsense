"""Shared CAD polygon shape helpers."""

from __future__ import annotations

import gdstk
import numpy as np


def tablet(
    cx: float, cy: float, short_r: float, aspect_ratio: float,
    thickness_factor: float, angle_deg: float, layer: int,
) -> gdstk.Polygon:
    """A capsule/stadium ("tablet") shaped polygon -- built with its long
    axis along the word-line (X) axis, filleted by exactly half its
    (post-thickness) short dimension so the short edges round fully into
    semicircular caps, then rotated by `angle_deg` around its own center.
    Real contacts/vias are routinely drawn this way rather than as pure
    circles or squares.

    `short_r` is the preset-derived baseline half-thickness for this
    instance (already includes any per-instance jitter/outlier scaling);
    `thickness_factor` scales it uniformly (how "fat" the capsule is);
    `aspect_ratio` (>= 1.0) stretches the long dimension relative to that
    thickness (1.0 = a plain circle, no elongation). `angle_deg` is measured
    from the word-line axis -- 0 = elongated along word-lines, 90 =
    elongated along bit-lines, anything in between is a diagonal tablet.
    """
    half_short = short_r * thickness_factor
    half_long = half_short * max(aspect_ratio, 1.0)
    rect = gdstk.rectangle((cx - half_long, cy - half_short), (cx + half_long, cy + half_short), layer=layer)
    rect.fillet(min(half_long, half_short), tolerance=0.05)
    if angle_deg % 180 != 0:
        rect.rotate(np.radians(angle_deg), center=(cx, cy))
    return rect

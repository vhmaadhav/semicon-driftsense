"""Classical, non-learned representations for Phase-2 hypothesis verification.

These helpers score an already-localised hypothesis.  They deliberately return
only a similarity value: native ZNCC remains the sole owner of the reported
sub-pixel coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np


def _gray_float(image: np.ndarray) -> np.ndarray:
    """Return a contiguous 2-D float32 image accepted by OpenCV."""
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {image.shape}")
    return np.ascontiguousarray(image, dtype=np.float32)


def rank_transform(image: np.ndarray, radius: int = 2) -> np.ndarray:
    """Count neighbours darker than each centre pixel in a reflected window."""
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    src = np.asarray(image)
    if src.ndim != 2:
        raise ValueError(f"expected a 2-D grayscale image, got shape {src.shape}")
    if radius == 0:
        return np.zeros(src.shape, dtype=np.float32)

    padded = np.pad(src, radius, mode="reflect")
    h, w = src.shape
    centre = padded[radius:radius + h, radius:radius + w]
    rank = np.zeros((h, w), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            neighbour = padded[radius + dy:radius + dy + h,
                               radius + dx:radius + dx + w]
            rank += neighbour < centre
    return rank


def common_band(image: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Low-pass both acquisitions to a common, conservative bandwidth."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return cv2.GaussianBlur(_gray_float(image), (0, 0),
                            sigmaX=float(sigma), sigmaY=float(sigma))


def dog_feature(image: np.ndarray, sigma_small: float = 0.8,
                sigma_large: float = 3.0) -> np.ndarray:
    """Difference-of-Gaussians structural representation."""
    if sigma_small <= 0 or sigma_large <= 0 or sigma_small >= sigma_large:
        raise ValueError("require 0 < sigma_small < sigma_large")
    src = _gray_float(image)
    small = cv2.GaussianBlur(src, (0, 0), sigmaX=float(sigma_small),
                             sigmaY=float(sigma_small))
    large = cv2.GaussianBlur(src, (0, 0), sigmaX=float(sigma_large),
                             sigmaY=float(sigma_large))
    return small - large


def local_match_score(search_feature: np.ndarray, template_feature: np.ndarray,
                      cx: float, cy: float, radius: int = 4) -> float:
    """Maximum local ZNCC near a fixed centre, without returning a new centre."""
    search = _gray_float(search_feature)
    template = _gray_float(template_feature)
    radius = int(radius)
    if radius < 0:
        raise ValueError("radius must be non-negative")

    h, w = search.shape
    th, tw = template.shape
    bx, by = float(cx) - tw / 2.0, float(cy) - th / 2.0
    x0, y0 = int(round(bx)) - radius, int(round(by)) - radius
    x1, y1 = x0 + tw + 2 * radius, y0 + th + 2 * radius
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x1, w), min(y1, h)
    if x1c - x0c < tw or y1c - y0c < th:
        return 0.0

    window = search[y0c:y1c, x0c:x1c]
    result = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    score = float(cv2.minMaxLoc(result)[1])
    return score if np.isfinite(score) else 0.0

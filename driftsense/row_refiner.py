"""Small, local horizontal refiner; preserves y, pose and confidence."""

from pathlib import Path
from functools import lru_cache
import cv2
import numpy as np

GRID = np.linspace(-4.0, 4.0, 33, dtype=np.float32)


def features(search, template, x, y):
    from .matching import row_offsets

    h, w = template.shape
    y0 = int(round(y - h / 2))
    x0 = int(round(x - w / 2))
    ci = int(round(y)) - y0
    if not 1 <= ci < h - 1:
        return None
    delta = y0 - (y - h / 2)
    ty = np.tile((np.arange(h, dtype=np.float32) + delta)[:, None], (1, w))
    tx = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    aligned = cv2.remap(
        template.astype(np.float32),
        tx,
        ty,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    curves = []
    for tpl in (template, aligned):
        _, _, corr = row_offsets(search, tpl, x, y, return_corr=True)
        if corr is None or not np.isfinite(corr).all():
            return None
        curves.extend([corr[ci - 1], corr[ci], corr[ci + 1], np.mean(corr, axis=0)])
    out = np.concatenate([np.asarray(curves).ravel(), [x0 + w / 2 - x, delta]])
    return out.astype(np.float32)


@lru_cache(maxsize=4)
def load_weights(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def correction(feature, weights):
    if weights["w1"].shape[1] == 138:
        feature = np.concatenate(
            [
                feature[..., :200]
                .reshape(*feature.shape[:-1], 8, 25)[..., 4:21]
                .reshape(*feature.shape[:-1], 136),
                feature[..., 200:],
            ],
            axis=-1,
        )
    a = np.maximum(feature @ weights["w1"].T + weights["b1"], 0)
    a = np.maximum(a @ weights["w2"].T + weights["b2"], 0)
    logits = a @ weights["w3"].T + weights["b3"]
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(axis=-1, keepdims=True)
    # Choose one mode, then interpolate locally, avoiding a global mean of
    # incompatible periodic matches.
    peak = np.argmax(p, axis=-1)
    mask = np.abs(np.arange(len(GRID)) - np.expand_dims(peak, -1)) <= 1
    p = p * mask
    return np.sum(p * GRID, axis=-1) / np.sum(p, axis=-1)


def refine(search, template, x, y, path):
    feature = features(search, template, x, y)
    if feature is None:
        return x
    dx = float(correction(feature, load_weights(str(Path(path).resolve()))))
    return float(x + dx) if np.isfinite(dx) and abs(dx) <= 4 else x

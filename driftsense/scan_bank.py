"""Image-only radial-distortion hypotheses for experimental horizontal refinement."""

import cv2
import numpy as np
from .matching import make_template, row_offsets, drift_row_refine

K_VALUES = np.linspace(-0.02, 0.02, 9)


def radial(x, y, k, shape, inverse=False):
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    nx = (np.asarray(x) - cx) / cx
    ny = (np.asarray(y) - cy) / cy
    r = np.hypot(nx, ny)
    if inverse:
        rd = r.copy()
        for _ in range(10):
            rd -= (rd * (1 + k * rd * rd) - r) / (1 + 3 * k * rd * rd)
        scale = np.divide(rd, r, out=np.ones_like(rd), where=r > 1e-12)
    else:
        scale = 1 + k * r * r
    return nx * scale * cx + cx, ny * scale * cy + cy


def candidates(reference, search, x, y, scale, theta):
    """Return coordinates, observable features and validity, with identity first."""
    xs = [float(x)]
    feat = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    valid = [True]
    cy, cx = (search.shape[0] - 1) / 2, (search.shape[1] - 1) / 2
    r2 = ((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2
    for k in K_VALUES:
        sx, sy = radial(x, y, k, search.shape)
        factor = float(np.clip(scale / (1 + 2 * k * r2), 8, 12))
        tpl = make_template(reference, factor, theta)
        radius = max(tpl.shape) // 2 + 20
        x0 = int(np.floor(sx)) - radius
        y0 = int(np.floor(sy)) - radius
        yy, xx = np.indices((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        mx, my = radial(xx + x0, yy + y0, k, search.shape, inverse=True)
        available = (
            mx.min() >= 0
            and my.min() >= 0
            and mx.max() < search.shape[1] - 1
            and my.max() < search.shape[0] - 1
        )
        proposal = x
        f = [0.0, k / 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if available:
            patch = cv2.remap(
                search, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR
            )
            lx, ly = float(sx - x0), float(sy - y0)
            off, peaks, corr = row_offsets(patch, tpl, lx, ly, return_corr=True)
            moved = drift_row_refine(patch, tpl, lx, ly)
            available = moved is not None and off is not None
            if available:
                proposed, _ = radial(moved[0] + x0, sy, k, search.shape, inverse=True)
                proposal = float(proposed)
                available = np.isfinite(proposal) and abs(proposal - x) <= 4
                row = int(round(ly)) - int(round(ly - len(tpl) / 2))
                finite = np.isfinite(peaks)
                if not 0 <= row < len(tpl) or not finite.any():
                    available = False
                else:
                    curve = corr[row]
                    order = np.sort(curve)
                    good = off[np.isfinite(off)]
                    f = [
                        0.0,
                        k / 0.02,
                        (proposal - x) / 4,
                        float(curve.max()),
                        float(np.mean(peaks[finite])),
                        float(np.std(peaks[finite])),
                        float(order[-1] - order[-2]),
                        float(np.median(good)) / 12,
                        float(np.std(good)) / 12,
                        float(finite.mean()),
                        r2,
                        float(factor - scale),
                    ]
        xs.append(proposal if available else x)
        feat.append(f)
        valid.append(bool(available))
    return (
        np.array(xs),
        np.nan_to_num(np.array(feat, dtype=np.float32)),
        np.array(valid),
    )

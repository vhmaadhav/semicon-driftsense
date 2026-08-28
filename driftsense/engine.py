"""Losses, training step and evaluation for Drift-Sense."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from driftsense.matching import locate, locate_phase2, response_to_center, select_peak
from driftsense.model import TEMPLATE_SIZE


def focal_loss(logit: torch.Tensor, target: torch.Tensor, peak: torch.Tensor,
               found: torch.Tensor | None = None) -> torch.Tensor:
    """CenterNet-style penalty-reduced focal loss.

    There is exactly one positive cell against ~10^4 negatives, so plain BCE
    is swamped. The (1-target)^4 term also softens the penalty on cells right
    next to the true one, which matters here because the neighbours are not
    really wrong -- they are half a stride away.
    """
    p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
    b = logit.shape[0]

    pos_mask = torch.zeros_like(p)
    idx = torch.arange(b, device=p.device)
    pos_mask[idx, 0, peak[:, 0], peak[:, 1]] = 1.0
    if found is not None:
        # An absent pair has no true cell. Clearing its positive leaves the
        # whole map supervised as negative, which is exactly the signal that
        # drives the peak logit down on absent pairs -- and so turns the
        # existing score into a calibrated rejection statistic.
        pos_mask = pos_mask * found.view(b, 1, 1, 1)

    pos_loss = -((1 - p) ** 2) * torch.log(p) * pos_mask
    neg_weights = (1 - target) ** 4
    neg_loss = -(p ** 2) * torch.log(1 - p) * neg_weights * (1 - pos_mask)

    return (pos_loss.sum() + neg_loss.sum()) / b


def offset_loss(offset: torch.Tensor, target: torch.Tensor, peak: torch.Tensor,
                found: torch.Tensor | None = None) -> torch.Tensor:
    """Smooth-L1 on the sub-cell offset, supervised only at the true cell.

    Absent pairs carry no true cell, so they are masked out entirely rather
    than regressed towards a placeholder.
    """
    b = offset.shape[0]
    idx = torch.arange(b, device=offset.device)
    pred = offset[idx, :, peak[:, 0], peak[:, 1]]
    if found is None:
        return F.smooth_l1_loss(pred, target, beta=0.1)
    per = F.smooth_l1_loss(pred, target, beta=0.1, reduction="none").mean(dim=1)
    return (per * found).sum() / found.sum().clamp(min=1.0)


def compute_loss(out: dict, batch: dict, offset_weight: float = 1.0) -> tuple:
    found = batch.get("found")
    lf = focal_loss(out["logit"], batch["heat"], batch["peak"], found)
    lo = offset_loss(out["offset"], batch["offset"], batch["peak"], found)
    return lf + offset_weight * lo, {"focal": float(lf.detach()), "offset": float(lo.detach())}


@torch.no_grad()
def decode_batch(out: dict, search_hw: tuple[int, int],
                 template_hw: tuple[int, int] = (TEMPLATE_SIZE, TEMPLATE_SIZE)) -> np.ndarray:
    """Decode a training batch's response maps to centres, for a quick
    in-the-loop accuracy read (no ZNCC refinement -- that is inference-only)."""
    prob = torch.sigmoid(out["logit"])[:, 0].float().cpu().numpy()
    offs = out["offset"].float().cpu().numpy()
    centers = []
    for b in range(prob.shape[0]):
        i, j, _ = select_peak(prob[b], search_hw, template_hw)
        centers.append(response_to_center(i, j, template_hw[0], template_hw[1],
                                          float(offs[b, 1, i, j]), float(offs[b, 0, i, j])))
    return np.array(centers, dtype=np.float32)


@torch.no_grad()
def evaluate(model, rows, device, tolerance=5.0, refine=True, limit=None, log_every=0,
             phase2=False):
    """Full-frame evaluation -- the real metric, run exactly like inference."""
    import os
    import cv2

    model.eval()
    if phase2:
        # Absent pairs carry a sentinel instead of a centre, so they cannot
        # contribute to a distance metric. Rejection quality is measured
        # separately by scripts/eval_phase2.py.
        rows = [r for r in rows if int(float(r.get("found", 1))) == 1]
    rows = rows[:limit] if limit else rows
    dists, scores = [], []

    for n, r in enumerate(rows):
        ref = cv2.imread(os.path.join(r["_split_dir"], r["reference_path"]), cv2.IMREAD_GRAYSCALE)
        sea = cv2.imread(os.path.join(r["_split_dir"], r["search_path"]), cv2.IMREAD_GRAYSCALE)
        res = (locate_phase2(model, ref, sea, device, refine=refine) if phase2
               else locate(model, ref, sea, device, refine=refine))
        gx, gy = float(r["gt_x_corr"]), float(r["gt_y_corr"])
        dists.append(float(np.hypot(res["x"] - gx, res["y"] - gy)))
        scores.append(res["score"])
        if log_every and (n + 1) % log_every == 0:
            print(f"    eval {n + 1}/{len(rows)}", flush=True)

    d = np.array(dists)
    return {
        "n": len(d),
        "median_px": float(np.median(d)),
        "mean_px": float(d.mean()),
        "acc@1px": float((d <= 1.0).mean()),
        "acc@2px": float((d <= 2.0).mean()),
        "acc@5px": float((d <= tolerance).mean()),
        "acc@10px": float((d <= 10.0).mean()),
        "p90_px": float(np.percentile(d, 90)),
        "dists": d,
        "scores": np.array(scores),
    }

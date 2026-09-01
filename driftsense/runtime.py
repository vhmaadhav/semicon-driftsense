"""Shared runtime helpers for Drift-Sense inference entry points.

Phase 2's canonical entry point is :mod:`register.py`.  This module owns the
small amount of reusable runtime plumbing that used to live in the historical
Phase 1 ``infer.py`` CLI: checkpoint loading, grayscale image loading, and the
classical fallback used when the learned model is unavailable.

Keeping these helpers here prevents the Phase 2 submission path from depending
on a legacy CLI while preserving that CLI as a compatibility wrapper.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(REPO_ROOT, "weights", "driftsense.pt")


def zncc_fallback(reference: np.ndarray, search: np.ndarray) -> dict:
    """Classical multi-scale ZNCC fallback used when the model cannot load."""
    from driftsense.matching import template_hypotheses

    best = None
    for scale in [
        f * m
        for f in template_hypotheses(reference)
        for m in (0.9, 0.95, 1.0, 1.05, 1.1)
    ]:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= search.shape[1] or th >= search.shape[0]:
            continue
        tmpl = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(search, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        if best is None or score > best["score"]:
            best = {
                "x": loc[0] + tw / 2.0,
                "y": loc[1] + th / 2.0,
                "score": float(score),
                "method": "zncc-fallback",
            }
    if best is None:
        return {
            "x": search.shape[1] / 2.0,
            "y": search.shape[0] / 2.0,
            "score": 0.0,
            "method": "center-fallback",
        }
    return best


def load_model(weights_path: str):
    """Return ``(model, device)`` or ``None`` if the learned path is unavailable."""
    try:
        import torch
        from driftsense.model import DriftSenseNet
    except Exception as exc:  # torch missing / broken install
        print(f"[warn] PyTorch unavailable ({exc}); using ZNCC fallback", file=sys.stderr)
        return None

    if not os.path.exists(weights_path):
        print(
            f"[warn] weights not found at {weights_path}; using ZNCC fallback",
            file=sys.stderr,
        )
        return None

    try:
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model", ckpt)
        # Scaled checkpoints record constructor kwargs.  Older checkpoints do
        # not, in which case the model's own defaults remain authoritative.
        kw = ckpt.get("arch_kwargs") or {}
        model = DriftSenseNet(**kw)
        model.load_state_dict(state)
        model.eval()
    except Exception as exc:
        print(f"[warn] could not load weights ({exc}); using ZNCC fallback", file=sys.stderr)
        return None

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return model.to(device), device


def read_gray(path: str) -> np.ndarray:
    """Read an image as grayscale, failing loudly on an unreadable path."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"error: could not read image '{path}'")
    return img

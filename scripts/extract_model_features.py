#!/usr/bin/env python3
"""Extract intermediate feature transformations through DriftSenseNet on real SEM data.

Extracts activation maps for 3 representative cases:
1. Nominal clean periodic array (Set A - test_A_00000000)
2. Degraded / high-noise / boundary case (Set B - test_B_00000001)
3. Hard absent / decoy rejection (Set C - test_C_00000000)
"""

from __future__ import annotations

import base64
import json
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE)
sys.path.insert(0, PROJECT_ROOT)

from driftsense.model import DriftSenseNet, grouped_xcorr, SCALE
from driftsense.matching import (
    make_template,
    canonicalize_search,
    standardize,
    pad_to_stride,
    uncanonicalize_point,
)

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "driftsense.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "docs", "model_visualizer")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_trained_model():
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
    state = ckpt.get("model", ckpt)
    kw = ckpt.get("arch_kwargs") or {}
    model = DriftSenseNet(**kw)
    model.load_state_dict(state)
    model.eval()
    return model, kw


def tensor_to_png_base64(tensor: torch.Tensor, colormap=cv2.COLORMAP_VIRIDIS, normalize=True) -> str:
    """Convert a 2D tensor (H, W) or (C, H, W) to a base64-encoded PNG string."""
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        # Mean across channels for multi-channel activations
        arr = np.mean(arr, axis=0)
    
    if normalize:
        vmin, vmax = np.percentile(arr, 1), np.percentile(arr, 99)
        if vmax > vmin:
            arr = np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)
        else:
            arr = np.zeros_like(arr)
        u8 = (arr * 255.0).astype(np.uint8)
    else:
        u8 = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        
    if colormap is not None:
        colored = cv2.applyColorMap(u8, colormap)
    else:
        colored = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
        
    _, buf = cv2.imencode(".png", colored)
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def image_file_to_base64(path: str, max_size: int = 500) -> str:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if max(img.shape) > max_size:
        scale = max_size / max(img.shape)
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def image_array_to_base64(img: np.ndarray, colormap=None) -> str:
    if colormap is not None:
        u8 = np.clip(img * 255.0 if img.dtype != np.uint8 else img, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(u8, colormap)
    else:
        colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    _, buf = cv2.imencode(".png", colored)
    return "data:image/png;base64," + base64.b64encode(buf).decode("ascii")


def trace_case(model: DriftSenseNet, case_meta: dict):
    ref_path = case_meta["ref_path"]
    search_path = case_meta["search_path"]
    scale = float(case_meta.get("scale", 10.0))
    theta = float(case_meta.get("theta", 0.0))
    
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    search_img = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)
    
    # 1. Pose Canonicalization & Area Downsampling (Standardized)
    # The reference is downsampled 10x to a nominal 100x100 template at 10 nm/px
    nominal_tpl = make_template(ref_img, float(SCALE), 0.0)
    tpl_n = standardize(nominal_tpl / 255.0)
    
    # The search frame is canonicalized by (scale, theta) to align scale and orientation
    canon_search, M = canonicalize_search(search_img, scale, theta)
    sea_padded = pad_to_stride(canon_search)
    sea_n = standardize(sea_padded / 255.0)
    
    template_t = torch.from_numpy(tpl_n)[None, None]
    search_t = torch.from_numpy(sea_n)[None, None]
    
    # 2. Extract Intermediate Activations Through the Network
    with torch.no_grad():
        # --- Shared Siamese Encoder Trunk ---
        # Template Branch
        tpl_stem0 = model.encoder.stem[0](template_t)  # Stride 2 (50x50)
        tpl_stem1 = model.encoder.stem[1](tpl_stem0)
        tpl_stem2 = model.encoder.stem[2](tpl_stem1)   # Stride 4 (25x25)
        tpl_body0 = model.encoder.body[0](tpl_stem2)   # ResBlock dilation=1
        tpl_body1 = model.encoder.body[1](tpl_body0)   # ResBlock dilation=2
        tf = model.encoder.out(tpl_body1)              # (1, 96, 25, 25)
        
        # Search Branch
        sea_stem0 = model.encoder.stem[0](search_t)    # Stride 2 (Hs/2, Ws/2)
        sea_stem1 = model.encoder.stem[1](sea_stem0)
        sea_stem2 = model.encoder.stem[2](sea_stem1)   # Stride 4 (Hs/4, Ws/4)
        sea_body0 = model.encoder.body[0](sea_stem2)   # ResBlock dilation=1
        sea_body1 = model.encoder.body[1](sea_body0)   # ResBlock dilation=2
        sf = model.encoder.out(sea_body1)              # (1, 96, Hs/4, Ws/4)
        
        # Channel L2-Normalization (makes matching invariant to SEM dose/gamma)
        tf_norm = F.normalize(tf, dim=1)
        sf_norm = F.normalize(sf, dim=1)
        
        # --- Grouped Cross-Correlation Branch ---
        raw_corr = grouped_xcorr(sf_norm, tf_norm, groups=8)  # (1, 8, Hc, Wc)
        corr_mix = model.corr_mix(raw_corr)                   # (1, 96, Hc, Wc)
        if model.fmf is not None:
            corr_mix = model.fmf(corr_mix)
            
        # --- Wide-Receptive-Field Context Branch ---
        ctx_d2 = model.context.body[0](sf_norm)   # dilation 2
        ctx_d4 = model.context.body[1](ctx_d2)    # dilation 4
        ctx_d8 = model.context.body[2](ctx_d4)    # dilation 8
        ctx_d16 = model.context.body[3](ctx_d8)   # dilation 16 (effective RF > 300px!)
        
        # Context Center Alignment (matches correlation response grid center)
        oi, oj = tf.shape[-2] // 2, tf.shape[-1] // 2
        ctx_aligned = ctx_d16[:, :, oi:oi + corr_mix.shape[-2], oj:oj + corr_mix.shape[-1]]
        
        # --- Concatenation & Deep Dilated Head ---
        fused = torch.cat([corr_mix, ctx_aligned], dim=1)  # (1, 144, Hc, Wc)
        head0 = model.head[0](fused)                       # dilation 1
        head1 = model.head[1](head0)                       # dilation 2
        head2 = model.head[2](head1)                       # dilation 4
        head3 = model.head[3](head2)                       # dilation 8
        
        # --- Dual Output Heads ---
        logits = model.logit(head3)                        # (1, 1, Hc, Wc)
        offsets = model.offset(head3)                      # (1, 2, Hc, Wc)
        
        prob_map = torch.sigmoid(logits)[0, 0].cpu().numpy()
        offset_map = offsets[0].cpu().numpy()              # (2, Hc, Wc)
        
    # Find peak in canonical frame
    pi, pj = np.unravel_index(np.argmax(prob_map), prob_map.shape)
    peak_score = float(prob_map[pi, pj])
    
    # Continuous sub-cell offset correction
    dx = float(offset_map[0, pi, pj])
    dy = float(offset_map[1, pi, pj])
    canon_cx = (pj + dx) * 4.0 + 50.0
    canon_cy = (pi + dy) * 4.0 + 50.0
    
    # Map back to native search frame coordinates
    nat_x, nat_y = uncanonicalize_point(M, canon_cx, canon_cy)
    
    # Top 5 peaks in response map to highlight periodic decoys vs true peak
    flat_indices = np.argsort(prob_map.ravel())[::-1][:6]
    top_peaks = []
    for idx in flat_indices:
        r, c = np.unravel_index(idx, prob_map.shape)
        p_dx = float(offset_map[0, r, c])
        p_dy = float(offset_map[1, r, c])
        c_x = (c + p_dx) * 4.0 + 50.0
        c_y = (r + p_dy) * 4.0 + 50.0
        n_x, n_y = uncanonicalize_point(M, c_x, c_y)
        top_peaks.append({
            "cell": [int(r), int(c)],
            "score": float(prob_map[r, c]),
            "raw_corr_mean": float(raw_corr[0, :, r, c].mean().item()),
            "canonical_xy": [float(c_x), float(c_y)],
            "native_xy": [float(n_x), float(n_y)],
            "subpixel_offset": [float(p_dx), float(p_dy)],
        })
        
    gt_found = case_meta["found"]
    gt_x = case_meta["gt_x"]
    gt_y = case_meta["gt_y"]
    error_px = float(np.hypot(nat_x - gt_x, nat_y - gt_y)) if gt_found == 1 else None

    # Individual channels for raw correlation to show periodic decoy symmetry
    corr_ch0 = tensor_to_png_base64(raw_corr[0, 0], colormap=cv2.COLORMAP_TURBO)
    corr_ch3 = tensor_to_png_base64(raw_corr[0, 3], colormap=cv2.COLORMAP_TURBO)
    corr_ch7 = tensor_to_png_base64(raw_corr[0, 7], colormap=cv2.COLORMAP_TURBO)
    
    return {
        "case_id": case_meta["case_id"],
        "title": case_meta["title"],
        "architecture": case_meta["architecture"],
        "description": case_meta["description"],
        "pose": {
            "scale": scale,
            "theta": theta,
        },
        "ground_truth": {
            "gt_x": gt_x,
            "gt_y": gt_y,
            "found": gt_found,
        },
        "prediction": {
            "pred_x": float(nat_x),
            "pred_y": float(nat_y),
            "canonical_x": float(canon_cx),
            "canonical_y": float(canon_cy),
            "confidence": peak_score,
            "found": int(peak_score >= 0.18),
            "peak_cell": [int(pi), int(pj)],
            "subpixel_offset": [dx, dy],
            "error_px": error_px,
        },
        "top_peaks": top_peaks,
        "images": {
            "ref_sem": image_file_to_base64(ref_path, max_size=400),
            "search_sem": image_file_to_base64(search_path, max_size=500),
            "canonical_search": image_array_to_base64(canon_search),
            "template_downsampled": image_array_to_base64(nominal_tpl),
            "encoder_stem_stride2": tensor_to_png_base64(sea_stem0[0], colormap=cv2.COLORMAP_INFERNO),
            "encoder_stem_stride4": tensor_to_png_base64(sea_stem2[0], colormap=cv2.COLORMAP_INFERNO),
            "encoder_dilated_body": tensor_to_png_base64(sea_body1[0], colormap=cv2.COLORMAP_PLASMA),
            "raw_correlation_volume": tensor_to_png_base64(raw_corr[0], colormap=cv2.COLORMAP_TURBO),
            "raw_corr_ch0": corr_ch0,
            "raw_corr_ch3": corr_ch3,
            "raw_corr_ch7": corr_ch7,
            "context_dilation_2": tensor_to_png_base64(ctx_d2[0], colormap=cv2.COLORMAP_VIRIDIS),
            "context_dilation_4": tensor_to_png_base64(ctx_d4[0], colormap=cv2.COLORMAP_VIRIDIS),
            "context_dilation_8": tensor_to_png_base64(ctx_d8[0], colormap=cv2.COLORMAP_VIRIDIS),
            "context_dilation_16": tensor_to_png_base64(ctx_d16[0], colormap=cv2.COLORMAP_VIRIDIS),
            "dilated_head_lattice": tensor_to_png_base64(head3[0], colormap=cv2.COLORMAP_MAGMA),
            "heatmap_probability": tensor_to_png_base64(torch.from_numpy(prob_map), colormap=cv2.COLORMAP_HOT),
        },
        "shapes": {
            "ref_shape": list(ref_img.shape),
            "search_shape": list(search_img.shape),
            "canonical_search_shape": list(canon_search.shape),
            "template_shape": list(nominal_tpl.shape),
            "encoder_template_feat": list(tf.shape[1:]),
            "encoder_search_feat": list(sf.shape[1:]),
            "correlation_volume": list(raw_corr.shape[1:]),
            "context_features": list(ctx_d16.shape[1:]),
            "dilated_head_features": list(head3.shape[1:]),
            "probability_map": list(prob_map.shape),
        }
    }


def main():
    print("Loading trained DriftSenseNet checkpoint...")
    model, kw = load_trained_model()
    print(f"Model loaded: width={kw.get('width', 64)}, ctx={kw.get('ctx', 32)}, head={kw.get('head', 64)}")

    # 3 Real cases from ext_p2
    cases = [
        {
            "case_id": "case_1_nominal",
            "title": "Case 1: Nominal Periodic Memory (Set A)",
            "architecture": "FinFET 14nm",
            "description": "Standard nominal acquisition with repeating cell lattice. Local appearance produces multiple high-scoring correlation decoy peaks; the Context Branch (d=2,4,8,16) and Dilated Head (d=1,2,4,8) cleanly resolve the true site with sub-pixel precision.",
            "ref_path": os.path.join(PROJECT_ROOT, "data", "ext_p2", "A_0000", "reference", "00000.png"),
            "search_path": os.path.join(PROJECT_ROOT, "data", "ext_p2", "A_0000", "search", "00000.png"),
            "scale": 9.76816,
            "theta": 1.75417,
            "gt_x": 310.290,
            "gt_y": 335.824,
            "found": 1,
        },
        {
            "case_id": "case_2_degraded",
            "title": "Case 2: Degraded / Severe Noise & Boundary (Set B)",
            "architecture": "FinFET 14nm Degraded",
            "description": "Severe noise profile with detector grain, contrast drop, and charging streaks. While raw template correlation is degraded with ambiguous periodic peaks, the wide-RF Context Branch anchors on macro routing strips to lock onto the target.",
            "ref_path": os.path.join(PROJECT_ROOT, "data", "ext_p2", "B_0000", "reference", "00001.png"),
            "search_path": os.path.join(PROJECT_ROOT, "data", "ext_p2", "B_0000", "search", "00001.png"),
            "scale": 8.65025,
            "theta": 2.95604,
            "gt_x": 555.975,
            "gt_y": 315.135,
            "found": 1,
        },
        {
            "case_id": "case_3_absent",
            "title": "Case 3: Hard Absent / Decoy Rejection (Set C)",
            "architecture": "FinFET 7nm Non-Matching Region",
            "description": "The reference patch comes from an independent region of the die with identical local repeating cells. Raw cross-correlation finds deceptive candidate peaks, but the macro context mismatch suppresses the response head below the 0.18 threshold, correctly declining as absent (found=0).",
            "ref_path": os.path.join(PROJECT_ROOT, "data", "ext_p2", "C_0000", "reference", "00000.png"),
            "search_path": os.path.join(PROJECT_ROOT, "data", "ext_p2", "C_0000", "search", "00000.png"),
            "scale": 8.67064,
            "theta": 1.71021,
            "gt_x": -1.0,
            "gt_y": -1.0,
            "found": 0,
        },
    ]

    all_results = []
    for c in cases:
        print(f"\nTracing {c['title']}...")
        result = trace_case(model, c)
        pred = result['prediction']
        print(f"  Confidence: {pred['confidence']:.4f} | Found: {pred['found']}")
        print(f"  Predicted Native: ({pred['pred_x']:.2f}, {pred['pred_y']:.2f})")
        if pred['error_px'] is not None:
            print(f"  Ground Truth: ({result['ground_truth']['gt_x']:.2f}, {result['ground_truth']['gt_y']:.2f}) | Error: {pred['error_px']:.3f} px")
        all_results.append(result)

    json_path = os.path.join(OUTPUT_DIR, "feature_traces.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFeature traces saved to {json_path} ({os.path.getsize(json_path):,} bytes)")


if __name__ == "__main__":
    main()

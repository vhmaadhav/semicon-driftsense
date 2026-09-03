#!/usr/bin/env python3
"""
Evaluate the ZNCC baseline matcher across noise levels and produce
Precision-Recall curves.

Methodology:
  - For each noise level, generate N samples (mixed DRAM/FinFET presets).
  - Run the ZNCC matcher on each; record (score, correct) where correct =
    (distance to ground truth <= --tolerance-px).
  - Sweep the ZNCC score as an acceptance threshold: at each threshold,
    predictions scoring >= threshold are "accepted" (a reported match).
      TP = accepted & correct
      FP = accepted & incorrect
      FN = rejected -- every sample has exactly one true match to find, so
           total positives = N regardless of threshold.
    precision = TP / (TP + FP); recall = TP / N
  - AP (average precision) = area under the PR curve (trapezoidal).

Usage:
    python baseline_solution/evaluate.py --samples-per-level 40 --tolerance-px 5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.pipeline import GenerationParams, generate_sample
from baseline_solution.zncc import zncc_match

NOISE_LEVELS = [
    {"label": "low", "dose_search": 800.0, "detector_noise_sigma_search": 2.0,
     "shear_amplitude_px": 0.5, "drift_jitter_px": 0.2},
    {"label": "medium", "dose_search": 200.0, "detector_noise_sigma_search": 5.0,
     "shear_amplitude_px": 1.5, "drift_jitter_px": 0.5},
    {"label": "high", "dose_search": 60.0, "detector_noise_sigma_search": 8.0,
     "shear_amplitude_px": 2.5, "drift_jitter_px": 1.0, "speckle_sigma": 0.15},
    {"label": "severe", "dose_search": 25.0, "detector_noise_sigma_search": 12.0,
     "shear_amplitude_px": 4.0, "drift_jitter_px": 1.8, "speckle_sigma": 0.3, "salt_pepper_prob": 0.01},
]

ARCHITECTURES = ["dram_1x", "dram_dense", "finfet_10nm", "finfet_7nm"]


def pr_curve(scores, corrects, n_total):
    order = np.argsort(-np.asarray(scores))
    corrects_sorted = np.asarray(corrects)[order]

    tp_cum = np.cumsum(corrects_sorted)
    fp_cum = np.cumsum(~corrects_sorted)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / max(n_total, 1)

    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return precision, recall


def average_precision(precision, recall):
    order = np.argsort(recall)
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(precision[order], recall[order]))


def evaluate_level(level, n_samples, tolerance_px, base_seed):
    rng = np.random.default_rng(base_seed)
    scores, corrects = [], []
    overrides = {k: v for k, v in level.items() if k != "label"}
    for i in range(n_samples):
        # Cycle architectures deterministically (not randomly) so every
        # noise level evaluates the exact same architecture distribution --
        # otherwise architecture-mix luck can swamp the noise-level signal
        # at moderate sample counts.
        arch = ARCHITECTURES[i % len(ARCHITECTURES)]
        params = GenerationParams(**overrides)
        sample = generate_sample(arch, rng, params)
        match = zncc_match(sample["reference_img"], sample["search_img"])
        dist = float(np.hypot(match["x"] - sample["gt_x"], match["y"] - sample["gt_y"]))
        scores.append(match["score"])
        corrects.append(dist <= tolerance_px)
    corrects = np.array(corrects, dtype=bool)
    precision, recall = pr_curve(scores, corrects, n_samples)
    ap = average_precision(precision, recall)
    accuracy = float(corrects.mean())
    return {
        "label": level["label"], "precision": precision, "recall": recall,
        "ap": ap, "accuracy": accuracy, "n": n_samples,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples-per-level", type=int, default=40)
    p.add_argument("--tolerance-px", type=float, default=5.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output-dir", default="./baseline_solution/eval_results")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for i, level in enumerate(NOISE_LEVELS):
        print(f"Evaluating noise level '{level['label']}' ({args.samples_per_level} samples)...")
        res = evaluate_level(level, args.samples_per_level, args.tolerance_px, args.seed + i * 9973)
        results.append(res)
        print(f"  AP={res['ap']:.3f}  accuracy@{args.tolerance_px}px={res['accuracy']:.3f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    for res in results:
        ax.plot(res["recall"], res["precision"], marker="o", markersize=3,
                 label=f"{res['label']} (AP={res['ap']:.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"ZNCC baseline: Precision-Recall by noise level (tol={args.tolerance_px}px)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(alpha=0.3)
    pr_path = os.path.join(args.output_dir, "pr_curves.png")
    fig.savefig(pr_path, dpi=150, bbox_inches="tight")
    print("wrote", pr_path)

    labels = [r["label"] for r in results]
    aps = [r["ap"] for r in results]
    accs = [r["accuracy"] for r in results]
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax2.plot(x, aps, marker="o", label="Average Precision")
    ax2.plot(x, accs, marker="s", label=f"Accuracy (<= {args.tolerance_px}px)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylim(0, 1.02)
    ax2.set_ylabel("Score")
    ax2.set_title("Baseline matcher quality vs noise level")
    ax2.legend()
    ax2.grid(alpha=0.3)
    trend_path = os.path.join(args.output_dir, "ap_vs_noise.png")
    fig2.savefig(trend_path, dpi=150, bbox_inches="tight")
    print("wrote", trend_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Single-process microbenchmark for Phase-2 verification overhead."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from driftsense.verification import (  # noqa: E402
    common_band, dog_feature, local_match_score, rank_transform,
)


def timed(fn, repeats: int) -> float:
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        values.append(time.perf_counter() - start)
    return float(np.median(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "results" / "verification_benchmark.json")
    args = parser.parse_args()
    cv2.setNumThreads(4)
    torch.set_num_threads(4)
    rng = np.random.default_rng(20260829)
    search = rng.integers(0, 256, (1000, 1000), dtype=np.uint8)
    templates = [rng.integers(0, 256, (n, n), dtype=np.uint8)
                 for n in (80, 100, 125)]
    centres = [(500.0, 500.0)] * 3

    rank_search = rank_transform(search)
    band_search = common_band(search)
    dog_search = dog_feature(search)

    def template_work(transform, search_feature):
        for template, (cx, cy) in zip(templates, centres):
            local_match_score(search_feature, transform(template), cx, cy, radius=4)

    results = {
        "rank_transform_full_s": timed(lambda: rank_transform(search), args.repeats),
        "band_transform_full_s": timed(lambda: common_band(search), args.repeats),
        "dog_transform_full_s": timed(lambda: dog_feature(search), args.repeats),
        "rank_3_templates_local_s": timed(
            lambda: template_work(rank_transform, rank_search), args.repeats),
        "band_3_templates_local_s": timed(
            lambda: template_work(common_band, band_search), args.repeats),
        "dog_3_templates_local_s": timed(
            lambda: template_work(dog_feature, dog_search), args.repeats),
    }
    results["total_verifier_overhead_s"] = sum(results.values())
    results["opencv_threads"] = cv2.getNumThreads()
    results["torch_threads"] = torch.get_num_threads()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

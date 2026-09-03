#!/usr/bin/env python3
"""Run the official-style coarse NCC baseline over an audit output."""

from __future__ import annotations

import argparse

from src.phase2_audit import OUTPUT_DEFAULT, run_baseline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args(argv)
    result = run_baseline(args.output_dir, args.threshold)
    print(f"calibrated {len(result['rows'])} pairs; threshold={result['threshold']:.2f}")
    print(f"wrote {args.output_dir}/baseline_calibration.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

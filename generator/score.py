#!/usr/bin/env python3
"""Run Issue 45 verification, geometry, resampling, and calibration checks."""

from __future__ import annotations

import argparse

from src.phase2_audit import OUTPUT_DEFAULT, run_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args(argv)
    metrics = run_score(args.output_dir, args.threshold)
    print(f"present verification: {metrics['all_present_verification_pass']}")
    print(f"overall present credit: {metrics['overall_present_credit']:.3f}")
    print(f"wrote {args.output_dir}/score.json and {args.output_dir}/REPORT.md")
    return 0 if metrics["all_present_verification_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

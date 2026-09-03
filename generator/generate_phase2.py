#!/usr/bin/env python3
"""Generate the deterministic Issue 45 A8/B6/C4/D2 audit package."""

from __future__ import annotations

import argparse

from src.phase2_audit import OUTPUT_DEFAULT, SEED_DEFAULT, generate_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--pairs", type=int, default=20,
                        help="audit pair count; Issue 45 requires exactly 20")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing non-empty output directory")
    args = parser.parse_args(argv)
    if args.pairs != 20:
        parser.error("Issue 45 audit generation requires --pairs 20")
    result = generate_audit(args.output_dir, args.seed, args.force)
    print(f"generated {result['meta']['pair_count']} pairs in {result['meta']['runtime_seconds']:.2f}s")
    print(f"output: {result['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Create the visual QA contact sheet for an Issue 45 audit output."""

from __future__ import annotations

import argparse

from src.phase2_audit import OUTPUT_DEFAULT, make_contact_sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    args = parser.parse_args(argv)
    path = make_contact_sheet(args.output_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

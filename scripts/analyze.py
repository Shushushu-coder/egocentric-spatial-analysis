#!/usr/bin/env python3
"""CLI for viewpoint-aware spatial analysis of reconstruction artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egocentric_spatial_analysis.analysis.pipeline import analyze_reconstruction
from egocentric_spatial_analysis.analysis.validate import validate_reconstruction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze reconstruction artifacts")
    parser.add_argument("--reconstruction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Explicit per-camera labels, in reconstruction order",
    )
    parser.add_argument(
        "--labels-json",
        default=None,
        help="JSON mapping of camera index or filename to observer label",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Optional external metric scale. Omit to keep reconstruction units.",
    )
    parser.add_argument("--n-rays", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.validate_only:
        print(json.dumps(validate_reconstruction(args.reconstruction_dir), indent=2))
        return 0
    report = analyze_reconstruction(
        args.reconstruction_dir,
        args.output_dir,
        labels=args.labels,
        labels_json=args.labels_json,
        scale=args.scale,
        overwrite=args.overwrite,
        n_rays=args.n_rays,
    )
    print(json.dumps({"output": args.output_dir, "labels": report["labels"], "units": report["units"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

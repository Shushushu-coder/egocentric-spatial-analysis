#!/usr/bin/env python3
"""CLI wrapper around the external VGGT reconstruction backend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egocentric_spatial_analysis.reconstruction.backend import (
    ReconstructionConfig,
    run_vggt_reconstruction,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run VGGT reconstruction (VGGT must be installed separately)"
    )
    parser.add_argument("--image-dir", required=True, help="Directory of input frames")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--weight-path",
        default=None,
        help="Local VGGT checkpoint. Not redistributed by this repository.",
    )
    parser.add_argument(
        "--from-pretrained",
        action="store_true",
        help="Load facebook/VGGT-1B via the official VGGT package (may download weights).",
    )
    parser.add_argument("--confidence-threshold", type=float, default=1.0)
    parser.add_argument("--skip-ply", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = ReconstructionConfig(
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        max_images=args.max_images,
        seed=args.seed,
        weight_path=Path(args.weight_path) if args.weight_path else None,
        from_pretrained=args.from_pretrained,
        confidence_threshold=args.confidence_threshold,
        export_ply=not args.skip_ply,
        overwrite=args.overwrite,
    )
    report = run_vggt_reconstruction(config)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

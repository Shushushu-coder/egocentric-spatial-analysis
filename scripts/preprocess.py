#!/usr/bin/env python3
"""CLI for video probing, frame-rate alignment, extraction, and resize."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from egocentric_spatial_analysis.preprocessing.extract import extract_synchronized_frames
from egocentric_spatial_analysis.preprocessing.framerate import align_framerate, probe_video
from egocentric_spatial_analysis.preprocessing.layout import assemble_image_folder
from egocentric_spatial_analysis.preprocessing.resize import resize_image_dir


def _parse_video_args(values: list[str]) -> dict[str, str]:
    videos: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(
                f"expected LABEL=PATH, got {item!r}"
            )
        label, path = item.split("=", 1)
        videos[label] = path
    return videos


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Egocentric video preprocessing")
    sub = parser.add_subparsers(dest="command", required=True)

    probe = sub.add_parser("probe", help="Print video metadata")
    probe.add_argument("video")

    align = sub.add_parser("align-fps", help="Re-encode a video to a target FPS")
    align.add_argument("--input", required=True)
    align.add_argument("--output", required=True)
    align.add_argument("--fps", type=float, required=True)
    align.add_argument("--overwrite", action="store_true")

    extract = sub.add_parser(
        "extract",
        help="Extract synchronized frames from labeled videos (LABEL=PATH)",
    )
    extract.add_argument("--video", action="append", required=True, dest="videos")
    extract.add_argument("--output-dir", required=True)
    extract.add_argument("--start-time", type=float, required=True)
    extract.add_argument("--end-time", type=float, required=True)
    extract.add_argument("--num-frames", type=int, default=10)
    extract.add_argument(
        "--strategy",
        choices=["uniform", "quality_weighted", "complementary"],
        default="quality_weighted",
    )
    extract.add_argument("--min-quality", type=float, default=0.0)
    extract.add_argument("--overwrite", action="store_true")

    resize = sub.add_parser("resize", help="Resize a folder of images to a square")
    resize.add_argument("--input-dir", required=True)
    resize.add_argument("--output-dir", required=True)
    resize.add_argument("--size", type=int, default=518)
    resize.add_argument(
        "--method", choices=["center_crop", "pad", "stretch"], default="center_crop"
    )
    resize.add_argument("--overwrite", action="store_true")

    assemble = sub.add_parser(
        "assemble",
        help="Copy labeled frame folders into one reconstruction image directory",
    )
    assemble.add_argument("--frames", action="append", required=True, dest="frames")
    assemble.add_argument("--output-dir", required=True)
    assemble.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "probe":
        info = probe_video(args.video)
        payload = {
            "path": str(info.path),
            "fps": info.fps,
            "fps_fraction": info.fps_fraction,
            "frame_count": info.frame_count,
            "duration_sec": info.duration_sec,
            "width": info.width,
            "height": info.height,
            "codec": info.codec,
        }
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "align-fps":
        out = align_framerate(args.input, args.output, target_fps=args.fps, overwrite=args.overwrite)
        print(out)
        return 0
    if args.command == "extract":
        videos = _parse_video_args(args.videos)
        result = extract_synchronized_frames(
            videos,
            args.output_dir,
            start_time=args.start_time,
            end_time=args.end_time,
            num_frames=args.num_frames,
            strategy=args.strategy,
            min_quality=args.min_quality,
            overwrite=args.overwrite,
        )
        print(json.dumps({"n_selected": len(result["selected"]), "output": args.output_dir}, indent=2))
        return 0
    if args.command == "resize":
        written = resize_image_dir(
            args.input_dir,
            args.output_dir,
            target_size=args.size,
            method=args.method,
            overwrite=args.overwrite,
        )
        print(json.dumps({"n_written": len(written), "output": args.output_dir}, indent=2))
        return 0
    if args.command == "assemble":
        labeled = _parse_video_args(args.frames)
        copied = assemble_image_folder(labeled, args.output_dir, overwrite=args.overwrite)
        print(json.dumps({"n_copied": len(copied), "output": args.output_dir}, indent=2))
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())

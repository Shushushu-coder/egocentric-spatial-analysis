"""Video probing and ffmpeg-based frame-rate alignment."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from egocentric_spatial_analysis.paths import refuse_if_exists, require_file


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    fps: float
    fps_fraction: str
    frame_count: int
    duration_sec: float
    width: int
    height: int
    codec: str


def _require_ffmpeg_binary(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise FileNotFoundError(
            f"{name} is not on PATH. Install ffmpeg and ensure it is available."
        )
    return found


def probe_video(path: str | Path) -> VideoInfo:
    """Return basic video metadata via ffprobe, falling back to OpenCV."""
    video_path = require_file(path, what="video")
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        return _probe_with_ffprobe(ffprobe, video_path)
    return _probe_with_opencv(video_path)


def _probe_with_ffprobe(ffprobe: str, video_path: Path) -> VideoInfo:
    cmd = [
        ffprobe,
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {video_path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), None)
    if stream is None:
        raise ValueError(f"no video stream in {video_path}")

    fps_str = stream.get("r_frame_rate", "0/1")
    num, den = (int(x) for x in fps_str.split("/"))
    fps = num / den if den else 0.0
    duration = float(data.get("format", {}).get("duration", 0.0) or 0.0)
    nb_frames = int(stream.get("nb_frames") or 0)
    if nb_frames == 0 and duration > 0 and fps > 0:
        nb_frames = int(round(duration * fps))
    return VideoInfo(
        path=video_path,
        fps=fps,
        fps_fraction=fps_str,
        frame_count=nb_frames,
        duration_sec=duration,
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        codec=str(stream.get("codec_name", "unknown")),
    )


def _probe_with_opencv(video_path: Path) -> VideoInfo:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps else 0.0
    finally:
        cap.release()
    return VideoInfo(
        path=video_path,
        fps=fps,
        fps_fraction=f"{fps:.6f}" if fps else "0/1",
        frame_count=frame_count,
        duration_sec=duration,
        width=width,
        height=height,
        codec="unknown",
    )


def align_framerate(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_fps: float,
    overwrite: bool = False,
    crf: int = 18,
    preset: str = "medium",
) -> Path:
    """Re-encode ``input_path`` to ``target_fps`` using ffmpeg.

    This does not assume any observer identity. Call it independently for
    each video that needs a shared sampling rate.
    """
    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    src = require_file(input_path, what="input video")
    dst = refuse_if_exists(output_path, overwrite=overwrite)
    ffmpeg = _require_ffmpeg_binary("ffmpeg")

    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-filter:v",
        f"fps=fps={target_fps}",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-c:a",
        "copy",
        str(dst),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg frame-rate alignment failed: {result.stderr.strip()}")
    return dst

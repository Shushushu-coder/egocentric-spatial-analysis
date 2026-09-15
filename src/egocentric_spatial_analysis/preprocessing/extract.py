"""Synchronized frame extraction from one or more labeled videos."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from egocentric_spatial_analysis.paths import ensure_output_dir, require_file
from egocentric_spatial_analysis.preprocessing.quality import FrameQualityEvaluator
from egocentric_spatial_analysis.preprocessing.sampling import (
    FrameCandidate,
    SamplingStrategy,
    select_frames,
)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV (cv2) is required for video extraction") from exc
    if not hasattr(cv2, "VideoCapture"):
        raise ImportError("cv2 is installed but missing VideoCapture; install opencv-python")
    return cv2


def _open_capture(path: Path):
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    return cap


def _validate_label(label: str) -> str:
    cleaned = label.strip()
    if not cleaned:
        raise ValueError("observer labels must be non-empty")
    if any(ch in cleaned for ch in r'\/:*?"<>|'):
        raise ValueError(f"invalid observer label: {label!r}")
    return cleaned


def extract_synchronized_frames(
    videos: dict[str, str | Path],
    output_dir: str | Path,
    *,
    start_time: float,
    end_time: float,
    num_frames: int = 10,
    strategy: SamplingStrategy | str = SamplingStrategy.QUALITY_WEIGHTED,
    min_quality: float = 0.0,
    blur_threshold: float = 100.0,
    exposure_threshold: float = 0.2,
    overwrite: bool = False,
) -> dict:
    """Extract time-aligned frames from labeled observer videos.

    ``videos`` maps an explicit observer label to a video path. Labels are
    never inferred from frame order.
    """
    if start_time < 0 or end_time <= start_time:
        raise ValueError("require 0 <= start_time < end_time")
    if not videos:
        raise ValueError("videos must contain at least one labeled path")

    labeled = {_validate_label(k): require_file(v, what=f"video for {k}") for k, v in videos.items()}
    out_root = ensure_output_dir(output_dir, overwrite=overwrite)
    evaluator = FrameQualityEvaluator(
        blur_threshold=blur_threshold, exposure_threshold=exposure_threshold
    )
    cv2 = _require_cv2()

    captures = {label: _open_capture(path) for label, path in labeled.items()}
    try:
        fps_values = {label: float(cap.get(cv2.CAP_PROP_FPS) or 0.0) for label, cap in captures.items()}
        if any(fps <= 0 for fps in fps_values.values()):
            raise ValueError("could not read a positive FPS from one or more videos")

        reference_label = next(iter(labeled))
        reference_fps = fps_values[reference_label]
        start_idx = int(start_time * reference_fps)
        end_idx = int(end_time * reference_fps)
        for cap in captures.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

        candidates: list[FrameCandidate] = []
        frame_count = 0
        while True:
            frames: dict[str, np.ndarray] = {}
            ok = True
            for label, cap in captures.items():
                ret, frame = cap.read()
                if not ret:
                    ok = False
                    break
                frames[label] = frame
            if not ok:
                break
            current_idx = start_idx + frame_count
            if current_idx > end_idx:
                break
            timestamp = start_time + frame_count / reference_fps

            qualities = [evaluator.evaluate(frame) for frame in frames.values()]
            avg_quality = float(np.mean([q.quality_score for q in qualities]))
            if avg_quality >= min_quality:
                similarity = 0.0
                matches = 0
                labels = list(frames)
                if len(labels) >= 2:
                    similarity = evaluator.histogram_similarity(frames[labels[0]], frames[labels[1]])
                    desc_a = evaluator.feature_descriptors(frames[labels[0]])
                    desc_b = evaluator.feature_descriptors(frames[labels[1]])
                    matches = evaluator.match_count(desc_a, desc_b)
                candidates.append(
                    FrameCandidate(
                        frame_index=current_idx,
                        timestamp_sec=timestamp,
                        quality_score=avg_quality,
                        sharpness=float(np.mean([q.sharpness for q in qualities])),
                        brightness=float(np.mean([q.brightness for q in qualities])),
                        contrast=float(np.mean([q.contrast for q in qualities])),
                        feature_count=float(np.mean([q.feature_count for q in qualities])),
                        similarity=similarity,
                        feature_matches=matches,
                    )
                )
            frame_count += 1
    finally:
        for cap in captures.values():
            cap.release()

    selected = select_frames(
        candidates,
        num_frames=num_frames,
        strategy=strategy,
        start_time=start_time,
        end_time=end_time,
    )
    if not selected:
        raise RuntimeError("no frames met the quality/time criteria")

    saved = _write_selected_frames(labeled, selected, out_root, overwrite=overwrite)
    metadata = {
        "videos": {label: str(path) for label, path in labeled.items()},
        "start_time": start_time,
        "end_time": end_time,
        "num_frames_requested": num_frames,
        "strategy": str(SamplingStrategy(strategy).value),
        "min_quality": min_quality,
        "selected": [asdict(item) for item in selected],
        "saved_files": {label: [str(p) for p in paths] for label, paths in saved.items()},
    }
    (out_root / "extraction_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    list_path = out_root / "reconstruction_input.txt"
    with list_path.open("w", encoding="utf-8") as handle:
        n = len(selected)
        for i in range(n):
            for label in labeled:
                handle.write(str(saved[label][i]) + "\n")
    return metadata


def _write_selected_frames(
    labeled: dict[str, Path],
    selected: list[FrameCandidate],
    out_root: Path,
    *,
    overwrite: bool,
) -> dict[str, list[Path]]:
    cv2 = _require_cv2()
    captures = {label: _open_capture(path) for label, path in labeled.items()}
    saved: dict[str, list[Path]] = {label: [] for label in labeled}
    try:
        for idx, candidate in enumerate(selected):
            for label, cap in captures.items():
                cap.set(cv2.CAP_PROP_POS_FRAMES, candidate.frame_index)
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError(
                        f"failed to seek frame {candidate.frame_index} in {labeled[label]}"
                    )
                label_dir = out_root / label
                label_dir.mkdir(parents=True, exist_ok=True)
                filename = f"frame_{idx:03d}_{label}_{candidate.timestamp_sec:.3f}s.png"
                out_path = label_dir / filename
                if out_path.exists() and not overwrite:
                    raise FileExistsError(f"refusing to overwrite {out_path}")
                if not cv2.imwrite(str(out_path), frame):
                    raise RuntimeError(f"failed to write {out_path}")
                saved[label].append(out_path)
    finally:
        for cap in captures.values():
            cap.release()
    return saved

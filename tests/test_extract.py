from pathlib import Path

import numpy as np
import pytest

from egocentric_spatial_analysis.preprocessing.extract import extract_synchronized_frames

cv2 = pytest.importorskip("cv2")
if not hasattr(cv2, "VideoWriter"):
    pytest.skip("working OpenCV video I/O is not available", allow_module_level=True)


def _write_video(path: Path, n_frames: int = 15, fps: float = 5.0, color=(0, 255, 0)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (32, 32))
    assert writer.isOpened(), "could not open VideoWriter"
    for i in range(n_frames):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[:] = color
        frame[i % 32, :, :] = 255
        writer.write(frame)
    writer.release()


def test_extract_named_observers(tmp_path: Path) -> None:
    video_a = tmp_path / "a.mp4"
    video_b = tmp_path / "b.mp4"
    _write_video(video_a, color=(0, 0, 255))
    _write_video(video_b, color=(0, 255, 0))
    out = tmp_path / "frames"
    result = extract_synchronized_frames(
        {"observer_a": video_a, "observer_b": video_b},
        out,
        start_time=0.0,
        end_time=2.0,
        num_frames=3,
        strategy="uniform",
        min_quality=0.0,
    )
    assert len(result["selected"]) == 3
    assert (out / "observer_a").is_dir()
    assert (out / "observer_b").is_dir()
    names_a = sorted(p.name for p in (out / "observer_a").glob("*.png"))
    assert names_a[0].startswith("frame_000_observer_a_")


def test_extract_requires_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        extract_synchronized_frames({}, tmp_path, start_time=0, end_time=1)

"""Load VGGT-style reconstruction artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from egocentric_spatial_analysis.paths import require_dir, require_file


@dataclass
class ReconstructionData:
    directory: Path
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    points_3d: np.ndarray
    depth_maps: np.ndarray | None = None
    depth_confidence: np.ndarray | None = None
    image_names: list[str] | None = None
    scale: float | None = None


def load_reconstruction(
    directory: str | Path,
    *,
    scale: float | None = None,
    require_depth: bool = False,
) -> ReconstructionData:
    root = require_dir(directory, what="reconstruction directory")
    extrinsic = np.load(require_file(root / "extrinsic.npy", what="extrinsic.npy"))
    intrinsic = np.load(require_file(root / "intrinsic.npy", what="intrinsic.npy"))
    points_3d = np.load(require_file(root / "points_3d.npy", what="points_3d.npy"))

    depth_maps = _optional_npy(root / "depth_maps.npy")
    depth_confidence = _optional_npy(root / "depth_confidence.npy")
    if require_depth and (depth_maps is None or depth_confidence is None):
        raise FileNotFoundError("depth_maps.npy and depth_confidence.npy are required")

    image_names = None
    report_path = root / "reconstruction_report.json"
    if report_path.is_file():
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))
        image_names = list(report.get("image_paths") or [])

    if scale is not None and scale <= 0:
        raise ValueError("scale must be positive when provided")
    return ReconstructionData(
        directory=root,
        extrinsic=np.asarray(extrinsic),
        intrinsic=np.asarray(intrinsic),
        points_3d=np.asarray(points_3d),
        depth_maps=None if depth_maps is None else np.asarray(depth_maps),
        depth_confidence=None if depth_confidence is None else np.asarray(depth_confidence),
        image_names=image_names,
        scale=scale,
    )


def flatten_valid_points(points_3d: np.ndarray) -> np.ndarray:
    if points_3d.size == 0:
        return np.empty((0, 3), dtype=float)
    points = points_3d.reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    nonzero = np.abs(points).sum(axis=1) > 1e-8
    return points[finite & nonzero]


def _optional_npy(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path)

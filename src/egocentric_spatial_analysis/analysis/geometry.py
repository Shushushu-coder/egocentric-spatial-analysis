"""Camera geometry helpers. Distances are in reconstruction units unless scaled."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_matrices(extrinsic: np.ndarray) -> np.ndarray:
    """Return (N, 3, 3) rotation matrices from OpenCV [R|t] extrinsics."""
    _require_extrinsic(extrinsic)
    return extrinsic[:, :, :3]


def translation_vectors(extrinsic: np.ndarray) -> np.ndarray:
    _require_extrinsic(extrinsic)
    return extrinsic[:, :, 3]


def camera_centers(extrinsic: np.ndarray) -> np.ndarray:
    """World-space camera centers C = -R^T t."""
    rotations = rotation_matrices(extrinsic)
    translations = translation_vectors(extrinsic)
    centers = np.empty((extrinsic.shape[0], 3), dtype=float)
    for i, (rotation, translation) in enumerate(zip(rotations, translations)):
        centers[i] = -rotation.T @ translation
    return centers


def viewing_directions(extrinsic: np.ndarray) -> np.ndarray:
    """World-space camera +Z axes."""
    rotations = rotation_matrices(extrinsic)
    camera_z = np.array([0.0, 0.0, 1.0])
    return np.array([rotation.T @ camera_z for rotation in rotations], dtype=float)


def euler_xyz_degrees(rotation: np.ndarray) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees from an xyz Euler decomposition.

    scipy ``xyz`` yields (roll-about-x, pitch-about-y, yaw-about-z) in some
    conventions; here we map the triple to (pitch, yaw, roll) as
    (x-rotation, y-rotation, z-rotation) to stay consistent with the prior
    analysis scripts.
    """
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be 3x3, got {rotation.shape}")
    if not np.isfinite(rotation).all():
        raise ValueError("rotation contains non-finite values")
    if np.linalg.det(rotation) == 0:
        raise ValueError("rotation matrix is degenerate")
    euler = Rotation.from_matrix(rotation).as_euler("xyz", degrees=True)
    roll, pitch, yaw = (float(x) for x in euler)
    return pitch, yaw, roll


def elevation_from_horizontal_degrees(
    view_dir: np.ndarray, *, up_axis: int = 1
) -> float:
    """Signed elevation of a viewing direction relative to the horizontal plane."""
    if view_dir.shape != (3,):
        raise ValueError(f"view_dir must have shape (3,), got {view_dir.shape}")
    if up_axis not in (0, 1, 2):
        raise ValueError("up_axis must be 0, 1, or 2")
    norm = np.linalg.norm(view_dir)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("view_dir is degenerate")
    unit = view_dir / norm
    return float(np.degrees(np.arcsin(np.clip(unit[up_axis], -1.0, 1.0))))


def apply_scale(points: np.ndarray, scale: float | None) -> np.ndarray:
    if scale is None:
        return points
    if scale <= 0:
        raise ValueError("scale must be positive when provided")
    return points * scale


def _require_extrinsic(extrinsic: np.ndarray) -> None:
    if extrinsic.ndim != 3 or extrinsic.shape[1:] != (3, 4):
        raise ValueError(f"extrinsic must have shape (N, 3, 4), got {extrinsic.shape}")
    if extrinsic.shape[0] == 0:
        raise ValueError("extrinsic is empty")
    if not np.isfinite(extrinsic).all():
        raise ValueError("extrinsic contains non-finite values")

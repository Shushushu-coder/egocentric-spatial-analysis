"""Viewpoint / field-of-view angle summaries."""

from __future__ import annotations

import numpy as np

from egocentric_spatial_analysis.analysis.geometry import (
    elevation_from_horizontal_degrees,
    euler_xyz_degrees,
    rotation_matrices,
    viewing_directions,
)
from egocentric_spatial_analysis.analysis.labels import group_indices


def viewpoint_angles(
    extrinsic: np.ndarray,
    *,
    labels: list[str],
    up_axis: int = 1,
) -> dict:
    rotations = rotation_matrices(extrinsic)
    directions = viewing_directions(extrinsic)
    pitch: list[float] = []
    yaw: list[float] = []
    roll: list[float] = []
    elevation: list[float] = []
    for rotation, direction in zip(rotations, directions):
        p, y, r = euler_xyz_degrees(rotation)
        pitch.append(p)
        yaw.append(y)
        roll.append(r)
        elevation.append(elevation_from_horizontal_degrees(direction, up_axis=up_axis))

    grouped = group_indices(labels)
    by_label: dict[str, dict] = {}
    for label, indices in grouped.items():
        by_label[label] = {
            "pitch_mean": _mean(pitch, indices),
            "yaw_mean": _mean(yaw, indices),
            "roll_mean": _mean(roll, indices),
            "elevation_mean": _mean(elevation, indices),
            "n": len(indices),
            "pitch": [pitch[i] for i in indices],
            "yaw": [yaw[i] for i in indices],
            "roll": [roll[i] for i in indices],
            "elevation": [elevation[i] for i in indices],
        }
    return {
        "units": "degrees",
        "up_axis": up_axis,
        "by_label": by_label,
    }


def _mean(values: list[float], indices: list[int]) -> float:
    if not indices:
        return float("nan")
    return float(np.mean([values[i] for i in indices]))

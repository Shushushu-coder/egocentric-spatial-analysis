"""Relative height-layer summaries in reconstruction units."""

from __future__ import annotations

import numpy as np

from egocentric_spatial_analysis.analysis.geometry import camera_centers
from egocentric_spatial_analysis.analysis.labels import group_indices


def height_layer_histogram(
    points: np.ndarray,
    *,
    layer_height: float = 0.2,
    up_axis: int = 1,
) -> dict:
    if layer_height <= 0:
        raise ValueError("layer_height must be positive")
    if points.size == 0:
        return {
            "edges": [],
            "counts": [],
            "min": None,
            "max": None,
            "total": 0,
            "units": "reconstruction",
        }
    heights = points[:, up_axis]
    min_h = float(heights.min())
    max_h = float(heights.max())
    if min_h == max_h:
        return {
            "edges": [min_h, min_h + layer_height],
            "counts": [int(len(heights))],
            "min": min_h,
            "max": max_h,
            "total": int(len(heights)),
            "units": "reconstruction",
        }
    edges = np.arange(min_h, max_h + layer_height, layer_height)
    counts, edges = np.histogram(heights, bins=edges)
    return {
        "edges": [float(x) for x in edges],
        "counts": [int(x) for x in counts],
        "min": min_h,
        "max": max_h,
        "total": int(len(heights)),
        "units": "reconstruction",
    }


def classify_relative_height(
    points: np.ndarray,
    camera_height: float,
    *,
    band: float = 0.1,
    up_axis: int = 1,
) -> dict[str, float]:
    if points.size == 0:
        return {"below": 0.0, "eye_level": 0.0, "above": 0.0}
    if band < 0:
        raise ValueError("band must be non-negative")
    heights = points[:, up_axis]
    total = max(len(heights), 1)
    below = np.sum(heights < camera_height - band)
    eye = np.sum(np.abs(heights - camera_height) <= band)
    above = np.sum(heights > camera_height + band)
    return {
        "below": float(below / total),
        "eye_level": float(eye / total),
        "above": float(above / total),
    }


def camera_height_by_label(
    extrinsic: np.ndarray,
    labels: list[str],
    *,
    up_axis: int = 1,
) -> dict[str, dict[str, float]]:
    centers = camera_centers(extrinsic)
    grouped = group_indices(labels)
    result: dict[str, dict[str, float]] = {}
    for label, indices in grouped.items():
        heights = centers[indices, up_axis]
        result[label] = {
            "mean": float(heights.mean()),
            "std": float(heights.std()),
            "n": float(len(indices)),
        }
    return result

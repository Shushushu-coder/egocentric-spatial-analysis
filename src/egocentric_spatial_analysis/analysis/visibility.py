"""Frustum visibility, ground coverage, and ray-hit helpers."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, KDTree

from egocentric_spatial_analysis.analysis.geometry import camera_centers, rotation_matrices


def frustum_visibility_mask(
    camera_center: np.ndarray,
    rotation: np.ndarray,
    points: np.ndarray,
    *,
    fov_h_deg: float = 60.0,
    fov_v_deg: float = 45.0,
) -> np.ndarray:
    """Return a boolean mask of points inside a simple angular frustum.

    ``rotation`` is the camera-from-world rotation R, so X_cam = R @ (X - C).
    """
    if points.size == 0:
        return np.zeros((0,), dtype=bool)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {points.shape}")
    if not np.isfinite(camera_center).all() or not np.isfinite(rotation).all():
        raise ValueError("camera pose contains non-finite values")

    points_cam = (rotation @ (points - camera_center).T).T
    front = points_cam[:, 2] > 0
    angles_h = np.degrees(np.arctan2(points_cam[:, 0], points_cam[:, 2]))
    angles_v = np.degrees(np.arctan2(points_cam[:, 1], points_cam[:, 2]))
    return front & (np.abs(angles_h) < fov_h_deg / 2) & (np.abs(angles_v) < fov_v_deg / 2)


def ground_points_by_percentile(
    points: np.ndarray, *, percentile: float = 20.0, up_axis: int = 1
) -> tuple[np.ndarray, float]:
    if not 0 < percentile < 100:
        raise ValueError("percentile must be between 0 and 100")
    if points.size == 0:
        return points.reshape(0, 3), float("nan")
    heights = points[:, up_axis]
    threshold = float(np.percentile(heights, percentile))
    return points[heights <= threshold], threshold


def planar_coverage_area(points: np.ndarray, *, up_axis: int = 1) -> float:
    """Convex-hull area on the plane orthogonal to ``up_axis``."""
    if len(points) < 3:
        return 0.0
    axes = [0, 1, 2]
    axes.remove(up_axis)
    planar = points[:, axes]
    if not np.isfinite(planar).all():
        return 0.0
    try:
        hull = ConvexHull(planar)
        return float(hull.volume)
    except Exception:
        spans = np.ptp(planar, axis=0)
        return float(spans[0] * spans[1])


def ground_visibility_stats(
    extrinsic: np.ndarray,
    points: np.ndarray,
    *,
    labels: list[str],
    ground_percentile: float = 20.0,
    up_axis: int = 1,
    fov_h_deg: float = 60.0,
    fov_v_deg: float = 45.0,
) -> dict:
    if len(labels) != len(extrinsic):
        raise ValueError("labels must match the number of cameras")
    centers = camera_centers(extrinsic)
    rotations = rotation_matrices(extrinsic)
    ground, threshold = ground_points_by_percentile(
        points, percentile=ground_percentile, up_axis=up_axis
    )
    grouped: dict[str, dict[str, list[float]]] = {}
    for label in labels:
        grouped.setdefault(
            label,
            {"ratio": [], "max_distance": [], "mean_distance": [], "coverage": [], "count": []},
        )

    for i, label in enumerate(labels):
        mask = frustum_visibility_mask(
            centers[i],
            rotations[i],
            ground,
            fov_h_deg=fov_h_deg,
            fov_v_deg=fov_v_deg,
        )
        visible = ground[mask]
        ratio = float(mask.mean()) if len(ground) else 0.0
        if len(visible):
            horiz_axes = [ax for ax in range(3) if ax != up_axis]
            delta = visible[:, horiz_axes] - centers[i][horiz_axes]
            distances = np.linalg.norm(delta, axis=1)
            max_d = float(distances.max())
            mean_d = float(distances.mean())
            coverage = planar_coverage_area(visible, up_axis=up_axis)
        else:
            max_d = mean_d = coverage = 0.0
        grouped[label]["ratio"].append(ratio)
        grouped[label]["max_distance"].append(max_d)
        grouped[label]["mean_distance"].append(mean_d)
        grouped[label]["coverage"].append(coverage)
        grouped[label]["count"].append(float(mask.sum()))

    return {
        "ground_threshold": threshold,
        "n_ground_points": int(len(ground)),
        "units": "reconstruction",
        "by_label": {
            label: {
                "mean_ratio": float(np.mean(values["ratio"])) if values["ratio"] else 0.0,
                "mean_max_distance": float(np.mean(values["max_distance"])) if values["max_distance"] else 0.0,
                "mean_distance": float(np.mean(values["mean_distance"])) if values["mean_distance"] else 0.0,
                "mean_coverage": float(np.mean(values["coverage"])) if values["coverage"] else 0.0,
                "samples": values,
            }
            for label, values in grouped.items()
        },
    }


def generate_frustum_rays(
    n_rays: int, *, fov_h_deg: float = 60.0, fov_v_deg: float = 45.0
) -> np.ndarray:
    if n_rays <= 0:
        raise ValueError("n_rays must be positive")
    grid = max(1, int(np.sqrt(n_rays)))
    h_angles = np.linspace(-fov_h_deg / 2, fov_h_deg / 2, grid)
    v_angles = np.linspace(-fov_v_deg / 2, fov_v_deg / 2, grid)
    rays = []
    for h_ang in h_angles:
        for v_ang in v_angles:
            h_rad = np.radians(h_ang)
            v_rad = np.radians(v_ang)
            direction = np.array(
                [np.sin(h_rad), np.sin(v_rad), np.cos(h_rad) * np.cos(v_rad)], dtype=float
            )
            norm = np.linalg.norm(direction)
            rays.append(direction / norm)
    return np.array(rays, dtype=float)


def cast_rays(
    origin: np.ndarray,
    directions_world: np.ndarray,
    points: np.ndarray,
    *,
    max_distance: float = 10.0,
    hit_radius: float = 0.05,
    n_samples: int = 100,
) -> tuple[list[np.ndarray], list[float]]:
    if points.size == 0 or len(directions_world) == 0:
        return [], []
    tree = KDTree(points)
    distances = np.linspace(0.1, max_distance, n_samples)
    hits: list[np.ndarray] = []
    hit_distances: list[float] = []
    for direction in directions_world:
        norm = np.linalg.norm(direction)
        if norm == 0 or not np.isfinite(norm):
            continue
        unit = direction / norm
        samples = origin[None, :] + distances[:, None] * unit[None, :]
        dists, indices = tree.query(samples, k=1)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] < hit_radius:
            hits.append(points[int(indices[min_idx])])
            hit_distances.append(float(distances[min_idx]))
    return hits, hit_distances


def ray_hit_stats(
    extrinsic: np.ndarray,
    points: np.ndarray,
    *,
    labels: list[str],
    n_rays: int = 100,
    max_distance: float = 10.0,
) -> dict:
    centers = camera_centers(extrinsic)
    rotations = rotation_matrices(extrinsic)
    local_rays = generate_frustum_rays(n_rays)
    grouped: dict[str, dict[str, list[float]]] = {}
    for i, label in enumerate(labels):
        world_rays = np.array([rotations[i].T @ ray for ray in local_rays])
        hits, distances = cast_rays(centers[i], world_rays, points, max_distance=max_distance)
        grouped.setdefault(label, {"distance": [], "hit_height": []})
        grouped[label]["distance"].extend(distances)
        grouped[label]["hit_height"].extend([float(hit[1]) for hit in hits])
    return {
        "units": "reconstruction",
        "by_label": {
            label: {
                "mean_hit_distance": float(np.mean(values["distance"])) if values["distance"] else 0.0,
                "n_hits": len(values["distance"]),
            }
            for label, values in grouped.items()
        },
    }

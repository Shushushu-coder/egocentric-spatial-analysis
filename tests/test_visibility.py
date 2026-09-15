import numpy as np
import pytest

from egocentric_spatial_analysis.analysis.visibility import (
    cast_rays,
    frustum_visibility_mask,
    ground_points_by_percentile,
    planar_coverage_area,
)


def test_frustum_keeps_forward_point() -> None:
    center = np.zeros(3)
    rotation = np.eye(3)
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -2.0],
            [10.0, 0.0, 1.0],
        ]
    )
    mask = frustum_visibility_mask(center, rotation, points, fov_h_deg=60, fov_v_deg=45)
    assert mask.tolist() == [True, False, False]


def test_empty_points() -> None:
    mask = frustum_visibility_mask(np.zeros(3), np.eye(3), np.zeros((0, 3)))
    assert mask.shape == (0,)


def test_ground_percentile() -> None:
    points = np.array([[0, 0, 0], [0, 1, 0], [0, 2, 0], [0, 3, 0]], dtype=float)
    ground, threshold = ground_points_by_percentile(points, percentile=25, up_axis=1)
    assert threshold <= 1.0
    assert len(ground) >= 1


def test_coverage_too_few_points() -> None:
    assert planar_coverage_area(np.zeros((2, 3))) == 0.0


def test_cast_ray_hits_nearby_point() -> None:
    origin = np.zeros(3)
    points = np.array([[0.0, 0.0, 1.0]])
    hits, distances = cast_rays(
        origin,
        np.array([[0.0, 0.0, 1.0]]),
        points,
        max_distance=2.0,
        hit_radius=0.2,
        n_samples=20,
    )
    assert len(hits) == 1
    assert distances[0] == pytest.approx(1.0, abs=0.2)


def test_cast_ray_misses_empty_cloud() -> None:
    hits, distances = cast_rays(np.zeros(3), np.array([[0.0, 0.0, 1.0]]), np.zeros((0, 3)))
    assert hits == []
    assert distances == []

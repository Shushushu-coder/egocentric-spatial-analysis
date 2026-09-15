import numpy as np
import pytest

from egocentric_spatial_analysis.analysis.geometry import (
    camera_centers,
    elevation_from_horizontal_degrees,
    euler_xyz_degrees,
    viewing_directions,
)


def _identity_extrinsic(n: int = 2) -> np.ndarray:
    ext = np.zeros((n, 3, 4), dtype=float)
    ext[:, :3, :3] = np.eye(3)
    ext[:, :, 3] = np.array([0.0, 0.0, 2.0])
    return ext


def test_camera_center_from_opencv_extrinsic() -> None:
    ext = _identity_extrinsic(1)
    centers = camera_centers(ext)
    np.testing.assert_allclose(centers[0], np.array([0.0, 0.0, -2.0]))


def test_viewing_direction_identity() -> None:
    dirs = viewing_directions(_identity_extrinsic(1))
    np.testing.assert_allclose(dirs[0], np.array([0.0, 0.0, 1.0]))


def test_euler_identity() -> None:
    pitch, yaw, roll = euler_xyz_degrees(np.eye(3))
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(0.0)
    assert roll == pytest.approx(0.0)


def test_degenerate_rotation_raises() -> None:
    with pytest.raises(ValueError):
        euler_xyz_degrees(np.zeros((3, 3)))


def test_elevation_level_view() -> None:
    angle = elevation_from_horizontal_degrees(np.array([0.0, 0.0, 1.0]), up_axis=1)
    assert angle == pytest.approx(0.0)


def test_empty_extrinsic_raises() -> None:
    with pytest.raises(ValueError):
        camera_centers(np.zeros((0, 3, 4)))

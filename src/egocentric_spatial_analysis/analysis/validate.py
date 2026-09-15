"""Lightweight reconstruction artifact checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from egocentric_spatial_analysis.analysis.io import load_reconstruction


REQUIRED = ("extrinsic.npy", "intrinsic.npy", "points_3d.npy")
OPTIONAL = ("depth_maps.npy", "depth_confidence.npy")


def validate_reconstruction(directory: str | Path) -> dict:
    data = load_reconstruction(directory)
    issues: list[str] = []
    n = data.extrinsic.shape[0]
    if data.extrinsic.shape != (n, 3, 4):
        issues.append(f"unexpected extrinsic shape {data.extrinsic.shape}")
    if data.intrinsic.shape != (n, 3, 3):
        issues.append(f"unexpected intrinsic shape {data.intrinsic.shape}")
    if data.points_3d.ndim != 4 or data.points_3d.shape[0] != n or data.points_3d.shape[-1] != 3:
        issues.append(f"unexpected points_3d shape {data.points_3d.shape}")
    if not np.isfinite(data.extrinsic).all():
        issues.append("extrinsic contains non-finite values")
    if not np.isfinite(data.points_3d).all():
        issues.append("points_3d contains non-finite values")
    missing_optional = [name for name in OPTIONAL if not (Path(directory) / name).is_file()]
    return {
        "ok": not issues,
        "n_cameras": int(n),
        "issues": issues,
        "missing_optional": missing_optional,
        "units": "reconstruction",
    }

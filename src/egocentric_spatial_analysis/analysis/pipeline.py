"""Run viewpoint-aware summaries from a reconstruction directory."""

from __future__ import annotations

import json
from pathlib import Path

from egocentric_spatial_analysis.analysis.height import (
    camera_height_by_label,
    classify_relative_height,
    height_layer_histogram,
)
from egocentric_spatial_analysis.analysis.io import flatten_valid_points, load_reconstruction
from egocentric_spatial_analysis.analysis.labels import assign_labels
from egocentric_spatial_analysis.analysis.validate import validate_reconstruction
from egocentric_spatial_analysis.analysis.viewpoints import viewpoint_angles
from egocentric_spatial_analysis.analysis.visibility import ground_visibility_stats, ray_hit_stats
from egocentric_spatial_analysis.paths import ensure_output_dir


def analyze_reconstruction(
    reconstruction_dir: str | Path,
    output_dir: str | Path,
    *,
    labels: list[str] | None = None,
    labels_json: str | Path | None = None,
    scale: float | None = None,
    overwrite: bool = False,
    n_rays: int = 64,
) -> dict:
    validation = validate_reconstruction(reconstruction_dir)
    data = load_reconstruction(reconstruction_dir, scale=scale)
    image_names = data.image_names
    camera_labels = assign_labels(
        len(data.extrinsic),
        labels=labels,
        image_names=image_names,
        mapping_path=labels_json,
    )
    points = flatten_valid_points(data.points_3d)
    if scale is not None:
        points = points * scale
        extrinsic = data.extrinsic.copy()
        extrinsic[:, :, 3] = extrinsic[:, :, 3] * scale
    else:
        extrinsic = data.extrinsic

    units = "metric_if_calibrated" if scale is not None else "reconstruction"
    heights = camera_height_by_label(extrinsic, camera_labels)
    relative_height = {
        label: classify_relative_height(points, stats["mean"])
        for label, stats in heights.items()
    }
    report = {
        "validation": validation,
        "labels": camera_labels,
        "units": units,
        "scale_applied": scale,
        "camera_height": heights,
        "relative_height": relative_height,
        "height_layers": height_layer_histogram(points),
        "viewpoints": viewpoint_angles(extrinsic, labels=camera_labels),
        "ground_visibility": ground_visibility_stats(
            extrinsic, points, labels=camera_labels
        ),
        "ray_hits": ray_hit_stats(extrinsic, points, labels=camera_labels, n_rays=n_rays),
        "notes": [
            "Distances are reconstruction units unless an external scale is supplied.",
            "Metric scale recovery is not treated as a reliable built-in capability.",
        ],
    }
    out = ensure_output_dir(output_dir, overwrite=overwrite)
    (out / "analysis_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

import numpy as np

from egocentric_spatial_analysis.analysis.height import (
    classify_relative_height,
    height_layer_histogram,
)


def test_empty_points() -> None:
    hist = height_layer_histogram(np.zeros((0, 3)))
    assert hist["total"] == 0
    rel = classify_relative_height(np.zeros((0, 3)), 0.0)
    assert rel["below"] == 0.0


def test_histogram_counts() -> None:
    points = np.array([[0, 0.0, 0], [0, 0.15, 0], [0, 0.55, 0]], dtype=float)
    hist = height_layer_histogram(points, layer_height=0.2)
    assert hist["total"] == 3
    assert sum(hist["counts"]) == 3


def test_relative_height_band() -> None:
    points = np.array([[0, -1.0, 0], [0, 0.0, 0], [0, 1.0, 0]], dtype=float)
    rel = classify_relative_height(points, 0.0, band=0.1)
    assert rel["below"] == rel["above"]
    assert rel["eye_level"] > 0

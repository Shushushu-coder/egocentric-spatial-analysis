import json
from pathlib import Path

import pytest

from egocentric_spatial_analysis.analysis.labels import assign_labels


def test_explicit_labels() -> None:
    assert assign_labels(2, labels=["a", "b"]) == ["a", "b"]


def test_filename_pattern() -> None:
    names = ["frame_000_observer_a_1.0s.png", "frame_000_observer_b_1.0s.png"]
    assert assign_labels(2, image_names=names) == ["observer_a", "observer_b"]


def test_mapping_json(tmp_path: Path) -> None:
    path = tmp_path / "labels.json"
    path.write_text(json.dumps({"0": "cam_a", "1": "cam_b"}), encoding="utf-8")
    assert assign_labels(2, mapping_path=path) == ["cam_a", "cam_b"]


def test_refuses_to_guess_from_order() -> None:
    with pytest.raises(ValueError, match="even/odd"):
        assign_labels(4)


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        assign_labels(2, labels=["only_one"])

"""Explicit observer labels. Frame index parity is never used."""

from __future__ import annotations

import json
import re
from pathlib import Path

_FRAME_PATTERN = re.compile(r"^frame_\d+_(?P<label>.+)_\d+(?:\.\d+)?s$")


def assign_labels(
    n_cameras: int,
    *,
    labels: list[str] | None = None,
    image_names: list[str] | None = None,
    mapping: dict[str, str] | None = None,
    mapping_path: str | Path | None = None,
) -> list[str]:
    """Return one label per camera.

    Priority:
    1. explicit ``labels`` list
    2. JSON mapping file / dict keyed by index or filename
    3. filename pattern ``frame_000_<label>_...``
    """
    if n_cameras <= 0:
        raise ValueError("n_cameras must be positive")

    if labels is not None:
        if len(labels) != n_cameras:
            raise ValueError(f"labels length {len(labels)} != n_cameras {n_cameras}")
        return [_require_label(item) for item in labels]

    loaded_mapping = dict(mapping or {})
    if mapping_path is not None:
        path = Path(mapping_path)
        loaded_mapping.update(json.loads(path.read_text(encoding="utf-8")))

    if loaded_mapping:
        return _labels_from_mapping(n_cameras, loaded_mapping, image_names)

    if image_names:
        inferred = [_label_from_filename(name) for name in image_names]
        if len(inferred) != n_cameras:
            raise ValueError(
                f"image_names length {len(inferred)} != n_cameras {n_cameras}"
            )
        if all(item is not None for item in inferred):
            return [item for item in inferred if item is not None]

    raise ValueError(
        "observer labels are required. Pass --labels / --labels-json, "
        "or use filenames of the form frame_000_<label>_.... "
        "This pipeline never infers identity from even/odd frame indices."
    )


def group_indices(labels: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        groups.setdefault(label, []).append(index)
    return groups


def _labels_from_mapping(
    n_cameras: int,
    mapping: dict[str, str],
    image_names: list[str] | None,
) -> list[str]:
    result: list[str] = []
    for index in range(n_cameras):
        key_candidates = [str(index)]
        if image_names is not None and index < len(image_names):
            key_candidates.append(image_names[index])
            key_candidates.append(Path(image_names[index]).name)
        found = None
        for key in key_candidates:
            if key in mapping:
                found = mapping[key]
                break
        if found is None:
            raise ValueError(f"no label mapping for camera index {index}")
        result.append(_require_label(found))
    return result


def _label_from_filename(name: str) -> str | None:
    stem = Path(name).stem
    match = _FRAME_PATTERN.match(stem)
    if not match:
        return None
    return _require_label(match.group("label"))


def _require_label(value: str) -> str:
    label = str(value).strip()
    if not label:
        raise ValueError("labels must be non-empty strings")
    return label

"""Assemble a flat image folder suitable for VGGT-style reconstruction."""

from __future__ import annotations

import shutil
from pathlib import Path

from egocentric_spatial_analysis.paths import ensure_output_dir, refuse_if_exists, require_dir


def assemble_image_folder(
    labeled_dirs: dict[str, str | Path],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    pattern: str = "*.png",
) -> list[Path]:
    """Copy labeled frame directories into one reconstruction input folder.

    Files keep their names. Observer identity stays in the filename (or in a
    separate labels file), never in even/odd index conventions.
    """
    if not labeled_dirs:
        raise ValueError("labeled_dirs must not be empty")
    dst = ensure_output_dir(output_dir, overwrite=overwrite)
    copied: list[Path] = []
    for label, directory in labeled_dirs.items():
        src = require_dir(directory, what=f"frames for {label}")
        for image_path in sorted(src.glob(pattern)):
            out_path = refuse_if_exists(dst / image_path.name, overwrite=overwrite)
            shutil.copy2(image_path, out_path)
            copied.append(out_path)
    return copied

"""Path validation helpers. All project I/O should go through these functions."""

from __future__ import annotations

from pathlib import Path


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser()


def require_file(path: str | Path, *, what: str = "file") -> Path:
    resolved = as_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"{what} not found: {resolved}")
    return resolved


def require_dir(path: str | Path, *, what: str = "directory") -> Path:
    resolved = as_path(path)
    if not resolved.is_dir():
        raise FileNotFoundError(f"{what} not found: {resolved}")
    return resolved


def ensure_output_dir(path: str | Path, *, overwrite: bool = False) -> Path:
    """Create an output directory.

    Existing non-empty directories are refused unless ``overwrite`` is True.
    """
    resolved = as_path(path)
    if resolved.exists():
        if not resolved.is_dir():
            raise NotADirectoryError(f"output path exists and is not a directory: {resolved}")
        nonempty = any(resolved.iterdir())
        if nonempty and not overwrite:
            raise FileExistsError(
                f"output directory is not empty: {resolved}. "
                "Pass overwrite=True to write into it."
            )
    else:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def refuse_if_exists(path: str | Path, *, overwrite: bool = False) -> Path:
    resolved = as_path(path)
    if resolved.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing file: {resolved}. "
            "Pass overwrite=True to replace it."
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved

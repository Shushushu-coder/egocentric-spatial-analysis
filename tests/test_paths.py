from pathlib import Path

import pytest

from egocentric_spatial_analysis.paths import ensure_output_dir, refuse_if_exists, require_file


def test_require_file_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        require_file(tmp_path / "nope.txt")


def test_ensure_output_dir_refuses_nonempty(tmp_path: Path) -> None:
    target = tmp_path / "out"
    target.mkdir()
    (target / "existing.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_output_dir(target, overwrite=False)
    ensure_output_dir(target, overwrite=True)


def test_refuse_if_exists(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("a", encoding="utf-8")
    with pytest.raises(FileExistsError):
        refuse_if_exists(path)
    refuse_if_exists(path, overwrite=True)

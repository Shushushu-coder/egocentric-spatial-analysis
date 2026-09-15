from pathlib import Path

import numpy as np
import pytest

from egocentric_spatial_analysis.analysis.pipeline import analyze_reconstruction
from egocentric_spatial_analysis.analysis.validate import validate_reconstruction
from egocentric_spatial_analysis.reconstruction.backend import ReconstructionConfig


def _identity_scene(tmp_path: Path) -> Path:
    recon = tmp_path / "recon"
    recon.mkdir()
    n = 2
    ext = np.zeros((n, 3, 4), dtype=float)
    ext[:, :3, :3] = np.eye(3)
    ext[0, :, 3] = [0, 0, 1]
    ext[1, :, 3] = [0, 0.2, 1]
    intrinsic = np.repeat(np.eye(3)[None, ...], n, axis=0)
    intrinsic[:, 0, 0] = 200
    intrinsic[:, 1, 1] = 200
    intrinsic[:, 0, 2] = 10
    intrinsic[:, 1, 2] = 10
    # Tiny point maps in front of the cameras.
    grid = np.zeros((n, 4, 4, 3), dtype=float)
    xs, ys = np.meshgrid(np.linspace(-0.2, 0.2, 4), np.linspace(-0.2, 0.2, 4))
    grid[..., 0] = xs
    grid[..., 1] = ys
    grid[..., 2] = 1.5
    np.save(recon / "extrinsic.npy", ext)
    np.save(recon / "intrinsic.npy", intrinsic)
    np.save(recon / "points_3d.npy", grid)
    return recon


def test_validate_ok(tmp_path: Path) -> None:
    recon = _identity_scene(tmp_path)
    report = validate_reconstruction(recon)
    assert report["ok"] is True
    assert report["n_cameras"] == 2


def test_analyze_with_explicit_labels(tmp_path: Path) -> None:
    recon = _identity_scene(tmp_path)
    out = tmp_path / "analysis"
    report = analyze_reconstruction(
        recon, out, labels=["observer_a", "observer_b"], n_rays=4
    )
    assert set(report["labels"]) == {"observer_a", "observer_b"}
    assert report["units"] == "reconstruction"
    assert (out / "analysis_report.json").is_file()


def test_reconstruction_config_requires_weights_or_pretrained() -> None:
    cfg = ReconstructionConfig(
        image_dir=Path("."),
        output_dir=Path("."),
        from_pretrained=False,
        weight_path=None,
    )
    assert cfg.weight_path is None
    assert cfg.from_pretrained is False


def test_missing_reconstruction_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        validate_reconstruction(tmp_path / "missing")


def test_require_dir_helper(tmp_path: Path) -> None:
    from egocentric_spatial_analysis.paths import require_dir

    require_dir(tmp_path)
    with pytest.raises(FileNotFoundError):
        require_dir(tmp_path / "nope")

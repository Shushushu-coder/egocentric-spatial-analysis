"""VGGT reconstruction wrapper. VGGT itself is an external dependency."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from egocentric_spatial_analysis.paths import ensure_output_dir, require_dir, require_file


@dataclass(frozen=True)
class ReconstructionConfig:
    image_dir: Path
    output_dir: Path
    device: str = "auto"
    max_images: int = 20
    seed: int = 42
    weight_path: Path | None = None
    from_pretrained: bool = False
    pretrained_id: str = "facebook/VGGT-1B"
    confidence_threshold: float = 1.0
    export_ply: bool = True
    overwrite: bool = False


def _select_device(requested: str):
    import torch

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"device must be auto, cpu, or cuda, got {requested!r}")


def _dtype_for_device(device) -> object:
    import torch

    if device.type == "cuda":
        major = torch.cuda.get_device_capability(device)[0]
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def _collect_images(image_dir: Path, max_images: int) -> list[Path]:
    paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"no images found in {image_dir}")
    if len(paths) > max_images:
        paths = paths[:max_images]
    return paths


def _load_vggt_model(config: ReconstructionConfig, device):
    import torch

    try:
        from vggt.models.vggt import VGGT
    except ImportError as exc:
        raise ImportError(
            "VGGT is an external dependency and is not installed. "
            "Install it from https://github.com/facebookresearch/vggt "
            "and obtain weights under the VGGT license."
        ) from exc

    if config.from_pretrained:
        model = VGGT.from_pretrained(config.pretrained_id)
    else:
        if config.weight_path is None:
            raise ValueError(
                "Provide weight_path or set from_pretrained=True. "
                "This project does not redistribute VGGT weights."
            )
        weight_path = require_file(config.weight_path, what="VGGT weight file")
        model = VGGT()
        state = torch.load(weight_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model.to(device)


def run_vggt_reconstruction(config: ReconstructionConfig) -> dict:
    """Run feed-forward VGGT reconstruction and write numpy artifacts.

    Does not download weights unless ``from_pretrained`` is explicitly True.
    Observer identity is not inferred; optional grouping is left to analysis.
    """
    image_dir = require_dir(config.image_dir, what="image directory")
    output_dir = ensure_output_dir(config.output_dir, overwrite=config.overwrite)
    image_paths = _collect_images(image_dir, config.max_images)

    import torch
    import torch.nn.functional as F
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images_square
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _select_device(config.device)
    dtype = _dtype_for_device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    images, _original_coords = load_and_preprocess_images_square(
        [str(p) for p in image_paths], 518
    )
    images = images.to(device)
    model = _load_vggt_model(config, device)

    try:
        with torch.no_grad():
            autocast_device = device.type if device.type == "cuda" else "cpu"
            with torch.autocast(device_type=autocast_device, dtype=dtype):
                vggt_resolution = 518
                images_resized = F.interpolate(
                    images,
                    size=(vggt_resolution, vggt_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
                images_batch = images_resized[None]
                aggregated_tokens_list, ps_idx = model.aggregator(images_batch)
                pose_enc = model.camera_head(aggregated_tokens_list)[-1]
                extrinsic, intrinsic = pose_encoding_to_extri_intri(
                    pose_enc, images_batch.shape[-2:]
                )
                depth_map, depth_conf = model.depth_head(
                    aggregated_tokens_list, images_batch, ps_idx
                )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                "GPU ran out of memory during VGGT inference. "
                "Reduce --max-images or use --device cpu."
            ) from exc
        raise

    extrinsic_np = extrinsic.squeeze(0).detach().cpu().numpy()
    intrinsic_np = intrinsic.squeeze(0).detach().cpu().numpy()
    depth_np = depth_map.squeeze(0).detach().cpu().numpy()
    conf_np = depth_conf.squeeze(0).detach().cpu().numpy()
    points_3d = unproject_depth_map_to_point_map(depth_np, extrinsic_np, intrinsic_np)

    colors = F.interpolate(images, size=(518, 518), mode="bilinear", align_corners=False)
    colors_np = (colors.detach().cpu().numpy() * 255).astype(np.uint8)
    colors_np = np.transpose(colors_np, (0, 2, 3, 1))

    if device.type == "cuda":
        torch.cuda.empty_cache()

    np.save(output_dir / "extrinsic.npy", extrinsic_np)
    np.save(output_dir / "intrinsic.npy", intrinsic_np)
    np.save(output_dir / "depth_maps.npy", depth_np)
    np.save(output_dir / "depth_confidence.npy", conf_np)
    np.save(output_dir / "points_3d.npy", points_3d)

    if config.export_ply:
        _maybe_export_ply(
            output_dir / "merged_point_cloud.ply",
            points_3d,
            colors_np,
            conf_np,
            config.confidence_threshold,
        )

    report = {
        "image_paths": [p.name for p in image_paths],
        "n_images": len(image_paths),
        "device": str(device),
        "output_dir": str(output_dir),
        "extrinsic_shape": list(extrinsic_np.shape),
        "points_3d_shape": list(points_3d.shape),
        "units": "reconstruction",
        "scale": "unknown",
        "config": {
            "max_images": config.max_images,
            "from_pretrained": config.from_pretrained,
            "weight_path": str(config.weight_path) if config.weight_path else None,
        },
    }
    (output_dir / "reconstruction_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def _maybe_export_ply(
    path: Path,
    points_3d: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    threshold: float,
) -> None:
    try:
        import trimesh
    except ImportError:
        return
    valid = confidence.reshape(-1) >= threshold
    vertices = points_3d.reshape(-1, 3)[valid]
    vertex_colors = colors.reshape(-1, 3)[valid]
    cloud = trimesh.PointCloud(vertices=vertices, colors=vertex_colors)
    cloud.export(path)


def config_from_args(args) -> ReconstructionConfig:
    return ReconstructionConfig(
        image_dir=Path(args.image_dir),
        output_dir=Path(args.output_dir),
        device=args.device,
        max_images=args.max_images,
        seed=args.seed,
        weight_path=Path(args.weight_path) if args.weight_path else None,
        from_pretrained=args.from_pretrained,
        confidence_threshold=args.confidence_threshold,
        export_ply=not args.skip_ply,
        overwrite=args.overwrite,
    )

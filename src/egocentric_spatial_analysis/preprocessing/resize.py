"""Resize images to a square VGGT-compatible resolution."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import numpy as np

from egocentric_spatial_analysis.paths import ensure_output_dir, refuse_if_exists, require_dir
from egocentric_spatial_analysis.preprocessing.image_ops import resize_hw

VGGT_DEFAULT_SIZE = 518


class ResizeMethod(str, Enum):
    CENTER_CROP = "center_crop"
    PAD = "pad"
    STRETCH = "stretch"


def center_crop_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = min(height, width)
    start_y = (height - size) // 2
    start_x = (width - size) // 2
    return image[start_y : start_y + size, start_x : start_x + size]


def pad_to_square(image: np.ndarray, pad_color: str = "black") -> np.ndarray:
    height, width = image.shape[:2]
    size = max(height, width)
    if pad_color == "black":
        color = (0, 0, 0)
    elif pad_color == "white":
        color = (255, 255, 255)
    elif pad_color == "mean":
        color = tuple(int(x) for x in image.mean(axis=(0, 1)))
    else:
        raise ValueError(f"unsupported pad_color: {pad_color}")

    if image.ndim == 3:
        padded = np.full((size, size, image.shape[2]), color, dtype=image.dtype)
    else:
        padded = np.full((size, size), color[0], dtype=image.dtype)
    start_y = (size - height) // 2
    start_x = (size - width) // 2
    padded[start_y : start_y + height, start_x : start_x + width] = image
    return padded


def resize_image(
    image: np.ndarray,
    *,
    target_size: int = VGGT_DEFAULT_SIZE,
    method: ResizeMethod | str = ResizeMethod.CENTER_CROP,
    pad_color: str = "black",
) -> np.ndarray:
    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")
    method = ResizeMethod(method)
    if method is ResizeMethod.CENTER_CROP:
        square = center_crop_square(image)
    elif method is ResizeMethod.PAD:
        square = pad_to_square(image, pad_color=pad_color)
    else:
        square = image
    return resize_hw(square, target_size, target_size)


def resize_image_dir(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    target_size: int = VGGT_DEFAULT_SIZE,
    method: ResizeMethod | str = ResizeMethod.CENTER_CROP,
    pad_color: str = "black",
    overwrite: bool = False,
    pattern: str = "*.png",
) -> list[Path]:
    src = require_dir(input_dir, what="input image directory")
    dst = ensure_output_dir(output_dir, overwrite=overwrite)
    written: list[Path] = []
    for image_path in sorted(src.glob(pattern)):
        image = _read_image(image_path)
        resized = resize_image(
            image, target_size=target_size, method=method, pad_color=pad_color
        )
        out_path = refuse_if_exists(dst / image_path.name, overwrite=overwrite)
        _write_image(out_path, resized)
        written.append(out_path)
    return written


def _read_image(path: Path) -> np.ndarray:
    try:
        import cv2

        if hasattr(cv2, "imread"):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"failed to read image: {path}")
            return image
    except Exception:
        pass
    from matplotlib import image as mpimg

    rgb = mpimg.imread(path)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    return rgb[..., ::-1]


def _write_image(path: Path, image: np.ndarray) -> None:
    try:
        import cv2

        if hasattr(cv2, "imwrite") and cv2.imwrite(str(path), image):
            return
    except Exception:
        pass
    from matplotlib import image as mpimg

    mpimg.imsave(path, image[..., ::-1])


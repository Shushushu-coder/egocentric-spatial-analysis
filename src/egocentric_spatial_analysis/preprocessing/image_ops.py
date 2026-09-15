"""NumPy image helpers so preprocessing tests do not require a working OpenCV build."""

from __future__ import annotations

import numpy as np


def as_uint8_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 image, got {image.shape}")
    return np.asarray(image)


def to_gray(image: np.ndarray) -> np.ndarray:
    bgr = as_uint8_bgr(image).astype(np.float64)
    return 0.114 * bgr[..., 0] + 0.587 * bgr[..., 1] + 0.299 * bgr[..., 2]


def laplacian_variance(gray: np.ndarray) -> float:
    kernel = np.array([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    padded = np.pad(gray, 1, mode="edge")
    acc = np.zeros_like(gray, dtype=np.float64)
    for dy in range(3):
        for dx in range(3):
            acc += kernel[dy, dx] * padded[dy : dy + gray.shape[0], dx : dx + gray.shape[1]]
    return float(acc.var())


def resize_hw(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbor resize; sufficient for VGGT input preparation tests."""
    if height <= 0 or width <= 0:
        raise ValueError("target height and width must be positive")
    src_h, src_w = image.shape[:2]
    ys = (np.linspace(0, src_h - 1, height)).astype(int)
    xs = (np.linspace(0, src_w - 1, width)).astype(int)
    return image[ys][:, xs]


def exposure_flags(gray: np.ndarray, threshold: float) -> tuple[bool, bool]:
    hist, _ = np.histogram(gray.clip(0, 255), bins=256, range=(0, 256))
    hist = hist / max(hist.sum(), 1)
    overexposed = bool(hist[240:].sum() > threshold)
    underexposed = bool(hist[:16].sum() > threshold)
    return overexposed, underexposed


def gradient_feature_count(gray: np.ndarray, limit: int = 2000) -> int:
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    thresh = float(np.percentile(mag, 90)) if mag.size else 0.0
    count = int((mag >= thresh).sum())
    return min(limit, count)

"""Frame quality metrics used for sampling reconstruction inputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from egocentric_spatial_analysis.preprocessing.image_ops import (
    exposure_flags,
    gradient_feature_count,
    laplacian_variance,
    to_gray,
)


@dataclass(frozen=True)
class FrameQuality:
    quality_score: float
    sharpness: float
    brightness: float
    contrast: float
    feature_count: int
    is_blurry: bool
    overexposed: bool
    underexposed: bool


class FrameQualityEvaluator:
    def __init__(
        self,
        *,
        blur_threshold: float = 100.0,
        exposure_threshold: float = 0.2,
        orb_features: int = 2000,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.exposure_threshold = exposure_threshold
        self.orb_features = orb_features
        self._orb = None

    def evaluate(self, frame_bgr: np.ndarray) -> FrameQuality:
        return evaluate_bgr_frame(
            frame_bgr,
            blur_threshold=self.blur_threshold,
            exposure_threshold=self.exposure_threshold,
            orb=self._maybe_orb(),
        )

    def feature_descriptors(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        orb = self._maybe_orb()
        if orb is None:
            return None
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        _keypoints, descriptors = orb.detectAndCompute(gray, None)
        return descriptors

    def match_count(self, desc_a: np.ndarray | None, desc_b: np.ndarray | None) -> int:
        if desc_a is None or desc_b is None:
            return 0
        import cv2

        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = matcher.match(desc_a, desc_b)
        except cv2.error:
            return 0
        return len(matches)

    def histogram_similarity(self, frame_a: np.ndarray, frame_b: np.ndarray) -> float:
        from egocentric_spatial_analysis.preprocessing.image_ops import resize_hw, to_gray

        gray_a = resize_hw(to_gray(frame_a), 45, 80)
        gray_b = resize_hw(to_gray(frame_b), 45, 80)
        a = gray_a.astype(np.float64).ravel()
        b = gray_b.astype(np.float64).ravel()
        a = a - a.mean()
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / denom, 0.0, 1.0))

    def _maybe_orb(self):
        if self._orb is not None:
            return self._orb
        try:
            import cv2

            if not hasattr(cv2, "ORB_create"):
                return None
            self._orb = cv2.ORB_create(nfeatures=self.orb_features)
            return self._orb
        except Exception:
            return None


def evaluate_bgr_frame(
    frame_bgr: np.ndarray,
    *,
    blur_threshold: float = 100.0,
    exposure_threshold: float = 0.2,
    orb=None,
) -> FrameQuality:
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"expected BGR image HxWx3, got shape {frame_bgr.shape}")
    gray = to_gray(frame_bgr)
    sharpness = laplacian_variance(gray)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    feature_count = 0
    if orb is not None:
        try:
            import cv2

            keypoints = orb.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), None)
            feature_count = len(keypoints)
        except Exception:
            feature_count = gradient_feature_count(gray)
    else:
        feature_count = gradient_feature_count(gray)

    overexposed, underexposed = exposure_flags(gray, exposure_threshold)
    is_blurry = sharpness < blur_threshold

    sharpness_score = min(100.0, sharpness / 5.0)
    brightness_score = 100.0 - abs(brightness - 127.0) / 1.27
    contrast_score = min(100.0, contrast / 0.7)
    feature_score = min(100.0, feature_count / 15.0)
    score = (
        sharpness_score * 0.30
        + brightness_score * 0.20
        + contrast_score * 0.20
        + feature_score * 0.30
    )
    if is_blurry:
        score *= 0.7
    if overexposed or underexposed:
        score *= 0.8
    score = float(np.clip(score, 0.0, 100.0))
    return FrameQuality(
        quality_score=score,
        sharpness=sharpness,
        brightness=brightness,
        contrast=contrast,
        feature_count=feature_count,
        is_blurry=is_blurry,
        overexposed=overexposed,
        underexposed=underexposed,
    )

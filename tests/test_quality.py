import numpy as np
import pytest

from egocentric_spatial_analysis.preprocessing.quality import evaluate_bgr_frame


def test_sharp_textured_frame_scores_higher_than_blur() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8)
    blur = np.full((64, 64, 3), 127, dtype=np.uint8)
    sharp_q = evaluate_bgr_frame(sharp, blur_threshold=1.0)
    blur_q = evaluate_bgr_frame(blur, blur_threshold=1.0)
    assert sharp_q.quality_score > blur_q.quality_score
    assert blur_q.is_blurry


def test_rejects_non_bgr() -> None:
    with pytest.raises(ValueError):
        evaluate_bgr_frame(np.zeros((10, 10), dtype=np.uint8))

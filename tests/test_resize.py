import numpy as np

from egocentric_spatial_analysis.preprocessing.resize import resize_image


def test_center_crop_outputs_target_square() -> None:
    image = np.zeros((108, 192, 3), dtype=np.uint8)
    image[:, 40:140] = 255
    out = resize_image(image, target_size=32, method="center_crop")
    assert out.shape == (32, 32, 3)


def test_pad_keeps_full_content_bounds() -> None:
    image = np.full((10, 20, 3), 200, dtype=np.uint8)
    out = resize_image(image, target_size=40, method="pad", pad_color="black")
    assert out.shape == (40, 40, 3)
    assert out[0, 0, 0] == 0

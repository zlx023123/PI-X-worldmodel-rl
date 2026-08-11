import numpy as np
import pytest

from pi0fast_wm_rl.cameras.mock import MockCamera


def test_mock_camera_is_deterministic_rgb() -> None:
    first = MockCamera(width=16, height=12, seed=7)
    second = MockCamera(width=16, height=12, seed=7)
    first.connect()
    second.connect()
    image_a = first.read()
    image_b = second.read()
    assert image_a.shape == (12, 16, 3)
    assert image_a.dtype == np.uint8
    np.testing.assert_array_equal(image_a, image_b)


def test_mock_camera_requires_connection() -> None:
    with pytest.raises(RuntimeError, match="not connected"):
        MockCamera().read()

"""Deterministic RGB camera."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseCamera


class MockCamera(BaseCamera):
    """Generate an RGB gradient with a deterministic moving marker."""

    def __init__(
        self,
        width: int = 160,
        height: int = 120,
        fps: float = 10,
        seed: int = 42,
        name: str = "front",
    ) -> None:
        if min(width, height, fps) <= 0:
            raise ValueError("width, height, and fps must be positive")
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.seed = int(seed)
        self.name = name
        self._connected = False
        self._frame_index = 0

    def connect(self) -> None:
        self._connected = True
        self._frame_index = 0

    def read(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("MockCamera is not connected")
        x = np.arange(self.width, dtype=np.uint16)[None, :]
        y = np.arange(self.height, dtype=np.uint16)[:, None]
        frame = np.empty((self.height, self.width, 3), dtype=np.uint8)
        frame[..., 0] = (x + self.seed) % 256
        frame[..., 1] = (y + 2 * self.seed) % 256
        frame[..., 2] = (x + y + self._frame_index) % 256
        marker_x = (self.seed + 3 * self._frame_index) % self.width
        marker_y = (self.seed + 2 * self._frame_index) % self.height
        frame[
            marker_y : min(marker_y + 4, self.height), marker_x : min(marker_x + 4, self.width)
        ] = 255
        self._frame_index += 1
        return frame

    def close(self) -> None:
        self._connected = False

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "mock",
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "seed": self.seed,
        }

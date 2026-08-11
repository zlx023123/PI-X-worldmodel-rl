"""Lifecycle manager for synchronized camera groups."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .base import BaseCamera


@dataclass(frozen=True, slots=True)
class CameraRead:
    images: dict[str, np.ndarray]
    timestamp: float


class CameraManager:
    """Open, read, and close a named collection of cameras."""

    def __init__(self, cameras: Mapping[str, BaseCamera]) -> None:
        if not cameras:
            raise ValueError("At least one camera is required")
        self.cameras = dict(cameras)

    def connect(self) -> None:
        connected: list[BaseCamera] = []
        try:
            for camera in self.cameras.values():
                camera.connect()
                connected.append(camera)
        except Exception:
            for camera in reversed(connected):
                camera.close()
            raise

    def read(self) -> CameraRead:
        images = {name: camera.read() for name, camera in self.cameras.items()}
        return CameraRead(images=images, timestamp=time.monotonic())

    def close(self) -> None:
        for camera in self.cameras.values():
            camera.close()

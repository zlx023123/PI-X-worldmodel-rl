"""Optional OpenCV USB camera adapter with delayed imports."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseCamera


class USBCamera(BaseCamera):
    """Read RGB frames from an OpenCV VideoCapture device."""

    def __init__(
        self, device: int | str, width: int, height: int, fps: float, name: str = "front"
    ) -> None:
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.name = name
        self._capture: Any = None

    @staticmethod
    def _cv2() -> Any:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "OpenCV is required only for USBCamera. Install with: pip install -e '.[opencv]'"
            ) from exc
        return cv2

    def connect(self) -> None:
        cv2 = self._cv2()
        self._capture = cv2.VideoCapture(self.device)
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture.set(cv2.CAP_PROP_FPS, self.fps)
        if not self._capture.isOpened():
            self._capture.release()
            self._capture = None
            raise RuntimeError(f"Could not open USB camera device: {self.device}")

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("USBCamera is not connected")
        ok, bgr = self._capture.read()
        if not ok:
            raise RuntimeError("USB camera read failed")
        return self._cv2().cvtColor(bgr, self._cv2().COLOR_BGR2RGB)

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "usb",
            "name": self.name,
            "device": self.device,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

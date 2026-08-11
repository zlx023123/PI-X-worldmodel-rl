"""Template and checklist for future real robot integrations."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRobot


class RealRobotAdapterTemplate(BaseRobot):
    """Non-runnable template: replace every method for the selected hardware."""

    def _pending(self) -> None:
        raise NotImplementedError(
            "Implement the vendor SDK lifecycle, calibration, watchdog, and physical e-stop first"
        )

    def connect(self) -> None:
        self._pending()

    def disconnect(self) -> None:
        self._pending()

    def reset(self) -> None:
        self._pending()

    def get_state(self) -> np.ndarray:
        self._pending()

    def send_action(self, action: np.ndarray) -> np.ndarray:
        self._pending()

    def stop(self) -> None:
        self._pending()

    def emergency_stop(self) -> None:
        self._pending()

    def metadata(self) -> dict[str, Any]:
        self._pending()

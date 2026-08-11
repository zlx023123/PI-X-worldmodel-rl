"""Basic key-to-joint-delta teleoperator."""

from __future__ import annotations

import numpy as np

from .base import BaseTeleoperator


class KeyboardTeleoperator(BaseTeleoperator):
    """Map a supplied key to a fixed joint delta without implementing kinematics."""

    _KEYS = {
        "q": (0, 1),
        "a": (0, -1),
        "w": (1, 1),
        "s": (1, -1),
        "e": (2, 1),
        "d": (2, -1),
        "r": (3, 1),
        "f": (3, -1),
        "t": (4, 1),
        "g": (4, -1),
        "y": (5, 1),
        "h": (5, -1),
        "u": (6, 1),
        "j": (6, -1),
    }

    def __init__(self, action_dim: int, step_size: float = 0.01) -> None:
        self.action_dim = action_dim
        self.step_size = step_size
        self._pending_key: str | None = None

    def reset(self) -> None:
        self._pending_key = None

    def feed_key(self, key: str) -> None:
        """Supply a key from any UI/event-loop integration."""
        self._pending_key = key.lower()

    def get_action(self, state: np.ndarray | None = None) -> np.ndarray:
        action = np.zeros(self.action_dim, dtype=np.float64)
        mapping = self._KEYS.get(self._pending_key or "")
        self._pending_key = None
        if mapping is not None and mapping[0] < self.action_dim:
            action[mapping[0]] = mapping[1] * self.step_size
        return action

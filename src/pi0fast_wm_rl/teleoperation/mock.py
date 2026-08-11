"""Deterministic synthetic demonstration source."""

from __future__ import annotations

import numpy as np

from .base import BaseTeleoperator


class MockTeleoperator(BaseTeleoperator):
    """Generate smooth deterministic joint deltas."""

    def __init__(self, action_dim: int, amplitude: float = 0.01) -> None:
        if action_dim <= 0 or amplitude <= 0:
            raise ValueError("action_dim and amplitude must be positive")
        self.action_dim = action_dim
        self.amplitude = amplitude
        self._step = 0

    def reset(self) -> None:
        self._step = 0

    def get_action(self, state: np.ndarray | None = None) -> np.ndarray:
        phase = 0.15 * self._step
        action = self.amplitude * np.sin(phase + np.arange(self.action_dim, dtype=np.float64))
        self._step += 1
        return action

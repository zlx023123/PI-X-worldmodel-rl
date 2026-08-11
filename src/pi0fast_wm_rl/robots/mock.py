"""Deterministic mock robot for hardware-free development."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRobot


class MockRobot(BaseRobot):
    """A delta-action robot simulator with joint-limit enforcement."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        initial_state: np.ndarray,
        joint_min: np.ndarray,
        joint_max: np.ndarray,
        deterministic: bool = True,
        seed: int = 0,
    ) -> None:
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.initial_state = self._vector(initial_state, state_dim, "initial_state")
        self.joint_min = self._vector(joint_min, state_dim, "joint_min")
        self.joint_max = self._vector(joint_max, state_dim, "joint_max")
        if np.any(self.joint_min >= self.joint_max):
            raise ValueError("Each joint_min must be smaller than joint_max")
        self.deterministic = deterministic
        self._rng = np.random.default_rng(seed)
        self._state = np.clip(self.initial_state, self.joint_min, self.joint_max)
        self._connected = False
        self._stopped = False
        self._emergency_stopped = False
        self.executed_actions: list[np.ndarray] = []

    @staticmethod
    def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (size,):
            raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains NaN or Inf")
        return array.copy()

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def reset(self) -> None:
        self._state = np.clip(self.initial_state, self.joint_min, self.joint_max)
        self.executed_actions.clear()
        self._stopped = False
        self._emergency_stopped = False

    def get_state(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("MockRobot is not connected")
        return self._state.copy()

    def send_action(self, action: np.ndarray) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("MockRobot is not connected")
        if self._emergency_stopped:
            raise RuntimeError("MockRobot emergency stop is active; reset is required")
        if self._stopped:
            raise RuntimeError("MockRobot is stopped; reset is required")
        command = self._vector(action, self.action_dim, "action")
        applied = np.zeros(self.state_dim, dtype=np.float64)
        width = min(self.state_dim, self.action_dim)
        applied[:width] = command[:width]
        if not self.deterministic:
            applied += self._rng.normal(0.0, 1e-4, size=self.state_dim)
        previous = self._state.copy()
        self._state = np.clip(previous + applied, self.joint_min, self.joint_max)
        executed = self._state - previous
        self.executed_actions.append(executed.copy())
        return executed

    def stop(self) -> None:
        self._stopped = True

    def emergency_stop(self) -> None:
        self._emergency_stopped = True
        self._stopped = True

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "mock",
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "deterministic": self.deterministic,
            "connected": self._connected,
            "emergency_stopped": self._emergency_stopped,
        }

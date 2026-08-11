"""Deterministic policy for complete hardware-free rollouts."""

from __future__ import annotations

from typing import Any

import numpy as np

from pi0fast_wm_rl.data.schema import Observation

from .base import BasePolicy


class MockPolicy(BasePolicy):
    """Move visible state dimensions toward a fixed target with safe deltas."""

    def __init__(
        self,
        action_dim: int,
        chunk_size: int = 10,
        target: float = 0.25,
        max_delta: float = 0.01,
    ) -> None:
        if min(action_dim, chunk_size, max_delta) <= 0:
            raise ValueError("action_dim, chunk_size, and max_delta must be positive")
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.target = float(target)
        self.max_delta = float(max_delta)

    def reset(self) -> None:
        return None

    def predict_action_chunk(self, observation: Observation) -> np.ndarray:
        state = np.asarray(observation.state, dtype=np.float64)
        action = np.zeros(self.action_dim, dtype=np.float64)
        width = min(len(state), self.action_dim)
        action[:width] = np.clip(self.target - state[:width], -self.max_delta, self.max_delta)
        return np.repeat(action[None, :], self.chunk_size, axis=0)

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "mock",
            "action_dim": self.action_dim,
            "chunk_size": self.chunk_size,
            "target": self.target,
        }

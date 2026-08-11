"""Future offline-RL protocol; no phase-one implementation."""

from __future__ import annotations

from typing import Any, Protocol


class OfflineRLTrainer(Protocol):
    """Fit a future policy/value method against an explicit replay buffer."""

    def fit(self, replay_buffer: Any) -> dict[str, float]: ...

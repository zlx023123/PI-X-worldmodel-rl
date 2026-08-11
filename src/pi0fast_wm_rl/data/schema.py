"""Typed internal data structures independent from LeRobot versions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class EpisodeMetadata:
    """Episode metadata, including fields reserved for later research stages."""

    episode_index: int
    task: str
    robot_type: str
    created_at: str
    fps: float
    success: bool | None = None
    quality: float | None = None
    mistake: str | None = None
    speed: str | None = None
    control_mode: str = "delta_joint"
    intervention: bool | None = None
    reward: float | None = None
    subtask: str | None = None
    subgoal_source: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)


@dataclass(slots=True)
class Observation:
    """Synchronized visual and robot observation."""

    images: dict[str, np.ndarray]
    state: np.ndarray
    timestamp: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation, including image pixels."""
        return {
            "images": {name: np.asarray(image).tolist() for name, image in self.images.items()},
            "state": np.asarray(self.state).tolist(),
            "timestamp": float(self.timestamp),
        }


@dataclass(slots=True)
class Transition:
    """A policy transition before episode indexing is assigned."""

    observation: Observation
    action: np.ndarray
    task: str
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable transition."""
        return {
            "observation": self.observation.to_dict(),
            "action": np.asarray(self.action).tolist(),
            "task": self.task,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class EpisodeStep:
    """One indexed transition in an episode."""

    observation: Observation
    action: np.ndarray
    task: str
    episode_index: int
    frame_index: int
    success: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable indexed step."""
        return {
            "observation": self.observation.to_dict(),
            "action": np.asarray(self.action).tolist(),
            "task": self.task,
            "episode_index": self.episode_index,
            "frame_index": self.frame_index,
            "success": self.success,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class Episode:
    """In-memory episode used by tests and small utilities."""

    metadata: EpisodeMetadata
    steps: list[EpisodeStep]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable episode."""
        return {
            "metadata": self.metadata.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
        }

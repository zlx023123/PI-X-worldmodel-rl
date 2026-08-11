"""Simple, inspectable PNG + NPZ + JSON episode recorder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .schema import EpisodeMetadata, Observation

SCHEMA_VERSION = "1.0"


class EpisodeRecorder:
    """Synchronously record one episode without depending on LeRobot."""

    def __init__(
        self,
        dataset_dir: str | Path,
        metadata: EpisodeMetadata,
        state_dim: int,
        action_dim: int,
        camera_names: tuple[str, ...] = ("front",),
    ) -> None:
        if state_dim <= 0 or action_dim <= 0:
            raise ValueError("state_dim and action_dim must be positive")
        if not camera_names or "front" not in camera_names:
            raise ValueError("camera_names must include 'front'")
        self.dataset_dir = Path(dataset_dir)
        self.metadata = metadata
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.camera_names = camera_names
        self.episode_dir = self.dataset_dir / f"episode_{metadata.episode_index:06d}"
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._timestamps: list[float] = []
        self._success: list[int] = []
        self._started = False
        self._finished = False

    def start(self) -> None:
        """Create a new episode directory, refusing to overwrite existing data."""
        if self._started:
            raise RuntimeError("Recorder has already started")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.episode_dir.mkdir(parents=False, exist_ok=False)
        for name in self.camera_names:
            (self.episode_dir / "images" / name).mkdir(parents=True)
        self._started = True

    @staticmethod
    def _finite_vector(value: np.ndarray, dim: int, field: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (dim,):
            raise ValueError(f"{field} must have shape ({dim},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{field} contains NaN or Inf")
        return array.copy()

    def record(
        self, observation: Observation, action: np.ndarray, success: bool | None = None
    ) -> None:
        """Synchronously write images and buffer numeric arrays for one frame."""
        if not self._started or self._finished:
            raise RuntimeError("Recorder must be started and not yet finished")
        state = self._finite_vector(observation.state, self.state_dim, "state")
        command = self._finite_vector(action, self.action_dim, "action")
        if not np.isfinite(observation.timestamp):
            raise ValueError("timestamp must be finite")
        missing = set(self.camera_names) - set(observation.images)
        if missing:
            raise ValueError(f"Missing camera image(s): {', '.join(sorted(missing))}")
        frame_index = len(self._states)
        for name in self.camera_names:
            image = np.asarray(observation.images[name])
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Image '{name}' must be uint8 HxWx3 RGB")
            path = self.episode_dir / "images" / name / f"frame_{frame_index:06d}.png"
            Image.fromarray(image, mode="RGB").save(path, format="PNG", compress_level=3)
        self._states.append(state)
        self._actions.append(command)
        self._timestamps.append(float(observation.timestamp))
        self._success.append(-1 if success is None else int(success))

    def finish(self, success: bool | None = None) -> Path:
        """Atomically expose numeric data and metadata for the completed episode."""
        if not self._started or self._finished:
            raise RuntimeError("Recorder must be started exactly once before finish")
        if not self._states:
            raise RuntimeError("Cannot finish an empty episode")
        if success is not None:
            self.metadata.success = bool(success)
        np.savez_compressed(
            self.episode_dir / "steps.npz",
            state=np.stack(self._states).astype(np.float32),
            action=np.stack(self._actions).astype(np.float32),
            timestamp=np.asarray(self._timestamps, dtype=np.float64),
            success=np.asarray(self._success, dtype=np.int8),
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "num_frames": len(self._states),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "cameras": list(self.camera_names),
            "episode": self.metadata.to_dict(),
        }
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        self._finished = True
        self._update_dataset_manifest()
        return self.episode_dir

    def _update_dataset_manifest(self) -> None:
        episode_dirs = sorted(
            path.name for path in self.dataset_dir.glob("episode_*") if path.is_dir()
        )
        payload = {
            "schema_version": SCHEMA_VERSION,
            "format": "png+npz+json",
            "episodes": episode_dirs,
            "total_episodes": len(episode_dirs),
        }
        target = self.dataset_dir / "dataset.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(target)

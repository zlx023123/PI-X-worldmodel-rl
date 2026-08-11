from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pi0fast_wm_rl.data.recorder import EpisodeRecorder
from pi0fast_wm_rl.data.schema import EpisodeMetadata, Observation


@pytest.fixture
def dataset_factory() -> Callable[..., Path]:
    def create(root: Path, episodes: int = 3, frames: int = 4, dim: int = 3) -> Path:
        dataset = root / "dataset"
        for episode_index in range(episodes):
            metadata = EpisodeMetadata(
                episode_index=episode_index,
                task="put the cube in the bowl" if episode_index % 2 == 0 else "touch the cube",
                robot_type="mock",
                created_at="2026-01-01T00:00:00Z",
                fps=10.0,
                extra={"action_bounds": {"min": [-0.1] * dim, "max": [0.1] * dim}},
            )
            recorder = EpisodeRecorder(dataset, metadata, dim, dim)
            recorder.start()
            for frame_index in range(frames):
                value = episode_index + frame_index / 10
                recorder.record(
                    Observation(
                        images={"front": np.full((8, 10, 3), frame_index, dtype=np.uint8)},
                        state=np.full(dim, value, dtype=np.float64),
                        timestamp=frame_index / 10,
                    ),
                    np.full(dim, frame_index / 100, dtype=np.float64),
                )
            recorder.finish(success=True)
        return dataset

    return create

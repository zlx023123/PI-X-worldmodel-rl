"""Read the internal episode format."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .schema import EpisodeMetadata, EpisodeStep, Observation


def episode_directories(dataset_dir: str | Path) -> list[Path]:
    """Return episode directories ordered by their zero-padded index."""
    root = Path(dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    return sorted(path for path in root.glob("episode_*") if path.is_dir())


def load_episode_payload(episode_dir: str | Path) -> dict[str, Any]:
    """Load episode metadata JSON."""
    path = Path(episode_dir) / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Episode metadata does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Episode metadata root must be an object: {path}")
    return value


def load_episode_arrays(episode_dir: str | Path) -> dict[str, np.ndarray]:
    """Load state, action, timestamp, and success arrays without pickles."""
    path = Path(episode_dir) / "steps.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Episode arrays do not exist: {path}")
    with np.load(path, allow_pickle=False) as values:
        return {name: values[name].copy() for name in values.files}


def iter_episode_steps(episode_dir: str | Path, load_images: bool = True) -> Iterator[EpisodeStep]:
    """Yield typed steps from a recorded episode."""
    root = Path(episode_dir)
    payload = load_episode_payload(root)
    arrays = load_episode_arrays(root)
    metadata = EpisodeMetadata(**payload["episode"])
    cameras = tuple(payload["cameras"])
    for frame_index in range(len(arrays["timestamp"])):
        images: dict[str, np.ndarray] = {}
        if load_images:
            for camera in cameras:
                image_path = root / "images" / camera / f"frame_{frame_index:06d}.png"
                with Image.open(image_path) as image:
                    images[camera] = np.asarray(image.convert("RGB"), dtype=np.uint8)
        marker = int(arrays["success"][frame_index]) if "success" in arrays else -1
        yield EpisodeStep(
            observation=Observation(
                images=images,
                state=arrays["state"][frame_index],
                timestamp=float(arrays["timestamp"][frame_index]),
            ),
            action=arrays["action"][frame_index],
            task=metadata.task,
            episode_index=metadata.episode_index,
            frame_index=frame_index,
            success=None if marker < 0 else bool(marker),
        )

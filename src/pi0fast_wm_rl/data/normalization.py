"""State and action normalization statistics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .episode_reader import episode_directories, load_episode_arrays


def _statistics(values: np.ndarray) -> dict[str, Any]:
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("Statistics require a non-empty 2D array")
    if not np.all(np.isfinite(values)):
        raise ValueError("Cannot compute statistics for NaN or Inf values")
    quantile_levels = (0.01, 0.1, 0.5, 0.9, 0.99)
    return {
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "quantiles": {
            str(level): np.quantile(values, level, axis=0).tolist() for level in quantile_levels
        },
    }


def compute_normalization_stats(dataset_dir: str | Path) -> dict[str, Any]:
    """Compute state/action statistics across every frame of every episode."""
    directories = episode_directories(dataset_dir)
    if not directories:
        raise ValueError("Cannot compute normalization statistics for an empty dataset")
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    for path in directories:
        arrays = load_episode_arrays(path)
        states.append(np.asarray(arrays["state"], dtype=np.float64))
        actions.append(np.asarray(arrays["action"], dtype=np.float64))
    state = np.concatenate(states)
    action = np.concatenate(actions)
    return {
        "schema_version": "1.0",
        "episode_count": len(directories),
        "frame_count": len(state),
        "state": _statistics(state),
        "action": _statistics(action),
    }


def save_normalization_stats(stats: dict[str, Any], output_path: str | Path) -> Path:
    """Persist normalization statistics as JSON."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return target

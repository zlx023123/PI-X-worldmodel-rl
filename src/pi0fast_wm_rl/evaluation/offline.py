"""Offline one-step action prediction evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from pi0fast_wm_rl.data.episode_reader import episode_directories, iter_episode_steps
from pi0fast_wm_rl.policies.base import BasePolicy

from .metrics import action_mse, latency_summary


def evaluate_offline(dataset_dir: str | Path, policy: BasePolicy) -> dict[str, Any]:
    """Evaluate the first predicted action against every recorded action."""
    predicted: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    latencies: list[float] = []
    episodes = episode_directories(dataset_dir)
    if not episodes:
        raise ValueError("Cannot evaluate an empty dataset")
    for episode in episodes:
        policy.reset()
        for step in iter_episode_steps(episode, load_images=True):
            started = time.perf_counter()
            chunk = np.asarray(policy.predict_action_chunk(step.observation), dtype=np.float64)
            latencies.append((time.perf_counter() - started) * 1000.0)
            if chunk.ndim != 2 or chunk.shape[1] != len(step.action):
                raise ValueError("Policy action dimension does not match dataset action dimension")
            predicted.append(chunk[0])
            targets.append(np.asarray(step.action, dtype=np.float64))
    pred = np.stack(predicted)
    target = np.stack(targets)
    return {
        "episode_count": len(episodes),
        "frame_count": len(targets),
        "action_mse": action_mse(pred, target),
        "inference_latency": latency_summary(latencies),
        "policy": policy.metadata(),
    }

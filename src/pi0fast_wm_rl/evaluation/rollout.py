"""Aggregate rollout results."""

from __future__ import annotations

from typing import Any

import numpy as np

from pi0fast_wm_rl.inference.runner import RolloutResult


def summarize_rollouts(results: list[RolloutResult]) -> dict[str, Any]:
    """Summarize mock or real rollout outcomes."""
    if not results:
        raise ValueError("At least one rollout result is required")
    return {
        "episode_count": len(results),
        "completed_count": sum(result.completed for result in results),
        "completion_rate": float(np.mean([result.completed for result in results])),
        "total_steps": sum(result.steps for result in results),
        "safety_clipped_values": sum(result.clipped_values for result in results),
        "safety_clipped_steps": sum(result.clipped_steps for result in results),
        "mean_inference_latency_ms": float(
            np.mean([result.mean_inference_latency_ms for result in results])
        ),
        "episodes": [result.to_dict() for result in results],
    }

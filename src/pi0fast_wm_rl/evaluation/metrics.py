"""Reusable policy and safety metrics."""

from __future__ import annotations

import numpy as np


def action_mse(predicted: np.ndarray, target: np.ndarray) -> float:
    """Return mean squared error for same-shaped action arrays."""
    lhs = np.asarray(predicted, dtype=np.float64)
    rhs = np.asarray(target, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.size == 0:
        raise ValueError("predicted and target must be non-empty and have the same shape")
    return float(np.mean(np.square(lhs - rhs)))


def action_out_of_bounds_rate(
    actions: np.ndarray, minimum: np.ndarray, maximum: np.ndarray
) -> float:
    """Return the fraction of action rows containing at least one out-of-range value."""
    values = np.asarray(actions, dtype=np.float64)
    low = np.asarray(minimum, dtype=np.float64)
    high = np.asarray(maximum, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != low.shape or low.shape != high.shape:
        raise ValueError("Action/bound dimensions do not match")
    if len(values) == 0:
        return 0.0
    violated = np.any((values < low) | (values > high), axis=1)
    return float(np.mean(violated))


def latency_summary(latencies_ms: list[float]) -> dict[str, float]:
    """Summarize inference latency without external statistics libraries."""
    if not latencies_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    values = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(np.quantile(values, 0.5)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "max_ms": float(values.max()),
    }

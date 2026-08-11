"""Timed, safety-filtered execution of policy action chunks."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from pi0fast_wm_rl.robots.base import BaseRobot
from pi0fast_wm_rl.robots.safety import SafetyFilter


@dataclass(frozen=True, slots=True)
class ExecutionMetrics:
    requested_steps: int
    executed_steps: int
    clipped_values: int
    clipped_steps: int
    latencies_ms: tuple[float, ...]
    stopped_early: bool
    dry_run: bool

    @property
    def mean_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0


class ActionChunkExecutor:
    """Execute [chunk_size, action_dim] commands at a controlled frequency."""

    def __init__(
        self,
        robot: BaseRobot,
        safety_filter: SafetyFilter,
        control_hz: float,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        self.robot = robot
        self.safety_filter = safety_filter
        self.control_hz = control_hz
        self.clock = clock
        self.sleeper = sleeper

    def execute(
        self,
        action_chunk: np.ndarray,
        n_action_steps: int | None = None,
        dry_run: bool = False,
        stop_condition: Callable[[], bool] | None = None,
    ) -> ExecutionMetrics:
        """Filter each step, optionally send it, and return an execution audit."""
        chunk = np.asarray(action_chunk, dtype=np.float64)
        if chunk.ndim != 2 or chunk.shape[0] == 0:
            raise ValueError("action_chunk must be a non-empty [chunk_size, action_dim] array")
        steps = len(chunk) if n_action_steps is None else min(n_action_steps, len(chunk))
        if steps <= 0:
            raise ValueError("n_action_steps must be positive")
        if self.safety_filter.last_communication_time is None:
            self.safety_filter.heartbeat(self.clock())
        latency: list[float] = []
        clipped_values = 0
        clipped_steps = 0
        executed = 0
        stopped_early = False
        period = 1.0 / self.control_hz
        for action in chunk[:steps]:
            if stop_condition is not None and stop_condition():
                stopped_early = True
                break
            cycle_start = self.clock()
            state = self.robot.get_state()
            result = self.safety_filter.filter(action, state, now=cycle_start)
            clipped_values += result.clipped_values
            clipped_steps += int(result.clipped)
            if not dry_run:
                self.robot.send_action(result.action)
            self.safety_filter.heartbeat(self.clock())
            latency.append((self.clock() - cycle_start) * 1000.0)
            executed += 1
            if not dry_run:
                remaining = period - (self.clock() - cycle_start)
                if remaining > 0:
                    self.sleeper(remaining)
        return ExecutionMetrics(
            requested_steps=steps,
            executed_steps=executed,
            clipped_values=clipped_values,
            clipped_steps=clipped_steps,
            latencies_ms=tuple(latency),
            stopped_early=stopped_early,
            dry_run=dry_run,
        )

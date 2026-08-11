"""End-to-end rollout orchestration."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

from pi0fast_wm_rl.cameras.manager import CameraManager
from pi0fast_wm_rl.policies.base import BasePolicy
from pi0fast_wm_rl.robots.base import BaseRobot
from pi0fast_wm_rl.robots.safety import SafetyFilter, SafetyViolation

from .action_chunk_executor import ActionChunkExecutor
from .observation_builder import ObservationBuilder


@dataclass(slots=True)
class RolloutResult:
    completed: bool
    steps: int
    chunks: int
    clipped_values: int
    clipped_steps: int
    mean_inference_latency_ms: float
    mean_execution_latency_ms: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RolloutRunner:
    """Own hardware lifecycle and execute policy chunks through SafetyFilter."""

    def __init__(
        self,
        robot: BaseRobot,
        cameras: CameraManager,
        policy: BasePolicy,
        safety_filter: SafetyFilter,
        control_hz: float,
        execute_steps: int,
        max_episode_steps: int,
    ) -> None:
        if execute_steps <= 0 or max_episode_steps <= 0:
            raise ValueError("execute_steps and max_episode_steps must be positive")
        self.robot = robot
        self.cameras = cameras
        self.policy = policy
        self.safety_filter = safety_filter
        self.execute_steps = execute_steps
        self.max_episode_steps = max_episode_steps
        self.builder = ObservationBuilder(robot, cameras)
        self.executor = ActionChunkExecutor(robot, safety_filter, control_hz)

    def run_episode(self, dry_run: bool = False) -> RolloutResult:
        """Run a bounded episode; completion means the configured horizon was safely reached."""
        self.robot.reset()
        self.policy.reset()
        self.safety_filter.reset()
        self.safety_filter.heartbeat()
        steps = chunks = clipped_values = clipped_steps = 0
        inference_latency: list[float] = []
        execution_latency: list[float] = []
        error: str | None = None
        try:
            while steps < self.max_episode_steps:
                observation = self.builder.build()
                started = time.perf_counter()
                chunk = np.asarray(self.policy.predict_action_chunk(observation), dtype=np.float64)
                inference_latency.append((time.perf_counter() - started) * 1000.0)
                limit = min(self.execute_steps, self.max_episode_steps - steps)
                metrics = self.executor.execute(chunk, n_action_steps=limit, dry_run=dry_run)
                steps += metrics.executed_steps
                chunks += 1
                clipped_values += metrics.clipped_values
                clipped_steps += metrics.clipped_steps
                execution_latency.extend(metrics.latencies_ms)
                if metrics.stopped_early or metrics.executed_steps == 0:
                    break
        except (SafetyViolation, RuntimeError, ValueError) as exc:
            error = str(exc)
            self.robot.stop()
        return RolloutResult(
            completed=steps >= self.max_episode_steps and error is None,
            steps=steps,
            chunks=chunks,
            clipped_values=clipped_values,
            clipped_steps=clipped_steps,
            mean_inference_latency_ms=float(np.mean(inference_latency))
            if inference_latency
            else 0.0,
            mean_execution_latency_ms=float(np.mean(execution_latency))
            if execution_latency
            else 0.0,
            error=error,
        )

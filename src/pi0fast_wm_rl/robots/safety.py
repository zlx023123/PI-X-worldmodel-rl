"""Model-independent robot action safety filtering."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


class SafetyViolation(RuntimeError):
    """Raised when an action cannot safely be executed."""


@dataclass(frozen=True, slots=True)
class SafetyResult:
    """A filtered action and audit information."""

    action: np.ndarray
    clipped_values: int
    reasons: tuple[str, ...]

    @property
    def clipped(self) -> bool:
        return self.clipped_values > 0


class SafetyFilter:
    """Enforce finite values, deltas, velocity, limits, timeout, and e-stop."""

    def __init__(
        self,
        joint_min: np.ndarray,
        joint_max: np.ndarray,
        max_action_delta: float,
        max_velocity: float,
        control_hz: float,
        timeout_seconds: float,
        action_mode: str = "delta_joint",
        workspace_hook: Callable[[np.ndarray, np.ndarray], bool] | None = None,
    ) -> None:
        self.joint_min = np.asarray(joint_min, dtype=np.float64)
        self.joint_max = np.asarray(joint_max, dtype=np.float64)
        if self.joint_min.ndim != 1 or self.joint_min.shape != self.joint_max.shape:
            raise ValueError("joint_min and joint_max must be same-sized vectors")
        if np.any(self.joint_min >= self.joint_max):
            raise ValueError("Each joint_min must be smaller than joint_max")
        if min(max_action_delta, max_velocity, control_hz, timeout_seconds) <= 0:
            raise ValueError("Safety limits, control_hz, and timeout must be positive")
        if action_mode not in {"delta_joint", "absolute_joint"}:
            raise ValueError("Unsupported action_mode")
        self.max_action_delta = float(max_action_delta)
        self.max_velocity = float(max_velocity)
        self.control_hz = float(control_hz)
        self.timeout_seconds = float(timeout_seconds)
        self.action_mode = action_mode
        self.workspace_hook = workspace_hook
        self.emergency_stopped = False
        self.timed_out = False
        self.last_communication_time: float | None = None
        self.total_clipped_values = 0

    def heartbeat(self, now: float | None = None) -> None:
        """Record a successful communication event."""
        self.last_communication_time = time.monotonic() if now is None else float(now)
        self.timed_out = False

    def check_timeout(self, now: float | None = None) -> bool:
        """Latch and report a communication timeout."""
        current = time.monotonic() if now is None else float(now)
        if (
            self.last_communication_time is not None
            and current - self.last_communication_time > self.timeout_seconds
        ):
            self.timed_out = True
        return self.timed_out

    def emergency_stop(self) -> None:
        """Latch the emergency-stop state."""
        self.emergency_stopped = True

    def reset(self) -> None:
        """Clear latched state; real adapters must require operator authorization."""
        self.emergency_stopped = False
        self.timed_out = False
        self.last_communication_time = None
        self.total_clipped_values = 0

    def filter(
        self,
        action: np.ndarray,
        current_state: np.ndarray,
        now: float | None = None,
    ) -> SafetyResult:
        """Return a safe command in the configured absolute or delta action mode."""
        if self.emergency_stopped:
            raise SafetyViolation("Emergency stop is active")
        if self.check_timeout(now):
            raise SafetyViolation("Communication timeout is active")
        raw = np.asarray(action, dtype=np.float64)
        state = np.asarray(current_state, dtype=np.float64)
        expected = self.joint_min.shape
        if raw.shape != expected or state.shape != expected:
            raise SafetyViolation(f"Action and state must have shape {expected}")
        if not np.all(np.isfinite(raw)):
            raise SafetyViolation("Action contains NaN or Inf")
        if not np.all(np.isfinite(state)):
            raise SafetyViolation("Robot state contains NaN or Inf")

        requested_delta = raw if self.action_mode == "delta_joint" else raw - state
        reasons: list[str] = []
        delta = np.clip(requested_delta, -self.max_action_delta, self.max_action_delta)
        if not np.array_equal(delta, requested_delta):
            reasons.append("max_action_delta")
        velocity_step = self.max_velocity / self.control_hz
        velocity_limited = np.clip(delta, -velocity_step, velocity_step)
        if not np.array_equal(velocity_limited, delta):
            reasons.append("max_velocity")
        target = np.clip(state + velocity_limited, self.joint_min, self.joint_max)
        if not np.array_equal(target, state + velocity_limited):
            reasons.append("joint_limits")
        if self.workspace_hook is not None and not self.workspace_hook(target.copy(), state.copy()):
            raise SafetyViolation("Workspace safety hook rejected the target")
        safe = target - state if self.action_mode == "delta_joint" else target
        clipped_values = int(np.count_nonzero(~np.isclose(safe, raw, rtol=0.0, atol=1e-12)))
        self.total_clipped_values += clipped_values
        return SafetyResult(safe, clipped_values, tuple(reasons))

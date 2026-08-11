import numpy as np
import pytest

from pi0fast_wm_rl.robots.safety import SafetyFilter, SafetyViolation


def make_filter(**overrides: float) -> SafetyFilter:
    values = {
        "max_action_delta": 0.2,
        "max_velocity": 1.0,
        "control_hz": 10.0,
        "timeout_seconds": 1.0,
    }
    values.update(overrides)
    return SafetyFilter(-np.ones(2), np.ones(2), **values)


def test_safety_filter_clips_delta_velocity_and_joint_limits() -> None:
    safety = make_filter()
    result = safety.filter(np.array([0.5, -0.5]), np.array([0.95, -0.95]))
    np.testing.assert_allclose(result.action, [0.05, -0.05])
    assert result.clipped
    assert "max_action_delta" in result.reasons
    assert "max_velocity" in result.reasons
    assert "joint_limits" in result.reasons


def test_safety_filter_rejects_nan() -> None:
    with pytest.raises(SafetyViolation, match="NaN or Inf"):
        make_filter().filter(np.array([np.nan, 0.0]), np.zeros(2))


def test_safety_filter_timeout_and_emergency_stop_are_latched() -> None:
    safety = make_filter(timeout_seconds=0.5)
    safety.heartbeat(now=1.0)
    with pytest.raises(SafetyViolation, match="timeout"):
        safety.filter(np.zeros(2), np.zeros(2), now=1.6)
    safety.reset()
    safety.emergency_stop()
    with pytest.raises(SafetyViolation, match="Emergency"):
        safety.filter(np.zeros(2), np.zeros(2))

import numpy as np
import pytest

from pi0fast_wm_rl.inference.action_chunk_executor import ActionChunkExecutor
from pi0fast_wm_rl.robots.safety import SafetyResult, SafetyViolation


class StopOnSecondCall:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls == 2


class SpyRobot:
    def __init__(
        self,
        state: np.ndarray,
        send_error: RuntimeError | None = None,
    ) -> None:
        self.state = np.asarray(state, dtype=np.float64)
        self.send_error = send_error
        self.get_state_calls = 0
        self.send_attempts: list[np.ndarray] = []

    def get_state(self) -> np.ndarray:
        self.get_state_calls += 1
        return self.state.copy()

    def send_action(self, action: np.ndarray) -> np.ndarray:
        command = np.asarray(action, dtype=np.float64).copy()
        self.send_attempts.append(command)
        if self.send_error is not None:
            raise self.send_error
        return command


class SpySafetyFilter:
    def __init__(
        self,
        results: list[SafetyResult] | None = None,
        violation_at: int | None = None,
        violation: SafetyViolation | None = None,
    ) -> None:
        self.results = results
        self.violation_at = violation_at
        self.violation = violation or SafetyViolation("rejected action")
        self.last_communication_time: float | None = None
        self.filter_calls: list[tuple[np.ndarray, np.ndarray, float | None]] = []

    def heartbeat(self, now: float | None = None) -> None:
        self.last_communication_time = now

    def filter(
        self,
        action: np.ndarray,
        current_state: np.ndarray,
        now: float | None = None,
    ) -> SafetyResult:
        call_index = len(self.filter_calls)
        self.filter_calls.append((action.copy(), current_state.copy(), now))
        if self.violation_at == call_index:
            raise self.violation
        if self.results is not None:
            return self.results[call_index]
        return SafetyResult(action.copy(), 0, ())


def make_executor(
    robot: SpyRobot | None = None,
    safety_filter: SpySafetyFilter | None = None,
    control_hz: float = 10.0,
) -> tuple[ActionChunkExecutor, SpyRobot, SpySafetyFilter, list[float]]:
    selected_robot = robot or SpyRobot(np.zeros(2))
    selected_filter = safety_filter or SpySafetyFilter()
    sleep_calls: list[float] = []
    executor = ActionChunkExecutor(
        selected_robot,
        selected_filter,
        control_hz,
        clock=lambda: 0.0,
        sleeper=sleep_calls.append,
    )
    return executor, selected_robot, selected_filter, sleep_calls


def test_execute_filters_sends_limited_steps_and_reports_metrics() -> None:
    filtered_actions = [
        SafetyResult(np.array([0.01, 0.02]), 1, ("limit",)),
        SafetyResult(np.array([0.03, 0.04]), 2, ("limit",)),
    ]
    safety = SpySafetyFilter(results=filtered_actions)
    executor, robot, _, _ = make_executor(safety_filter=safety)
    chunk = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    metrics = executor.execute(chunk, n_action_steps=2)

    assert len(safety.filter_calls) == 2
    np.testing.assert_allclose(safety.filter_calls[0][0], chunk[0])
    np.testing.assert_allclose(safety.filter_calls[1][0], chunk[1])
    assert robot.get_state_calls == 2
    assert len(robot.send_attempts) == 2
    np.testing.assert_allclose(robot.send_attempts[0], filtered_actions[0].action)
    np.testing.assert_allclose(robot.send_attempts[1], filtered_actions[1].action)
    assert metrics.requested_steps == 2
    assert metrics.executed_steps == 2
    assert metrics.clipped_values == 3
    assert metrics.clipped_steps == 2
    assert len(metrics.latencies_ms) == 2
    assert not metrics.stopped_early
    assert not metrics.dry_run


def test_execute_dry_run_filters_without_sending_or_sleeping() -> None:
    safety = SpySafetyFilter(
        results=[
            SafetyResult(np.array([0.05, 0.0]), 1, ("limit",)),
            SafetyResult(np.array([0.0, 0.05]), 0, ()),
        ]
    )
    executor, robot, _, sleep_calls = make_executor(safety_filter=safety)

    metrics = executor.execute(np.array([[0.1, 0.0], [0.0, 0.1]]), dry_run=True)

    assert len(safety.filter_calls) == 2
    assert robot.get_state_calls == 2
    assert robot.send_attempts == []
    assert sleep_calls == []
    assert metrics.requested_steps == 2
    assert metrics.executed_steps == 2
    assert metrics.clipped_values == 1
    assert metrics.clipped_steps == 1
    assert len(metrics.latencies_ms) == 2
    assert metrics.dry_run


def test_execute_stops_early_when_stop_condition_matches() -> None:
    executor, robot, safety, _ = make_executor()
    stop_condition = StopOnSecondCall()
    chunk = np.array([[0.1, 0.0], [0.2, 0.0], [0.3, 0.0]])

    metrics = executor.execute(chunk, stop_condition=stop_condition)

    assert metrics.requested_steps == 3
    assert metrics.executed_steps == 1
    assert metrics.stopped_early
    assert stop_condition.calls == 2
    assert len(safety.filter_calls) == 1
    assert robot.get_state_calls == 1
    assert len(robot.send_attempts) == 1
    np.testing.assert_allclose(robot.send_attempts[0], chunk[0])


def test_execute_propagates_safety_violation_without_sending() -> None:
    violation = SafetyViolation("workspace rejected action")
    safety = SpySafetyFilter(violation_at=0, violation=violation)
    executor, robot, _, sleep_calls = make_executor(safety_filter=safety)

    with pytest.raises(SafetyViolation) as raised:
        executor.execute(np.array([[0.1, 0.2], [0.3, 0.4]]))

    assert raised.value is violation
    assert len(safety.filter_calls) == 1
    assert robot.get_state_calls == 1
    assert robot.send_attempts == []
    assert sleep_calls == []


def test_execute_propagates_robot_send_error() -> None:
    send_error = RuntimeError("robot communication failed")
    robot = SpyRobot(np.zeros(2), send_error=send_error)
    executor, _, safety, sleep_calls = make_executor(robot=robot)

    with pytest.raises(RuntimeError) as raised:
        executor.execute(np.array([[0.1, 0.2], [0.3, 0.4]]))

    assert raised.value is send_error
    assert len(safety.filter_calls) == 1
    assert robot.get_state_calls == 1
    assert len(robot.send_attempts) == 1
    np.testing.assert_allclose(robot.send_attempts[0], [0.1, 0.2])
    assert sleep_calls == []


@pytest.mark.parametrize(
    "action_chunk",
    [np.empty((0, 2)), np.array([0.1, 0.2])],
)
def test_execute_rejects_invalid_action_chunk_shape(action_chunk: np.ndarray) -> None:
    executor = make_executor()[0]

    with pytest.raises(ValueError, match="non-empty"):
        executor.execute(action_chunk)


@pytest.mark.parametrize("n_action_steps", [0, -1])
def test_execute_rejects_nonpositive_action_steps(n_action_steps: int) -> None:
    executor = make_executor()[0]

    with pytest.raises(ValueError, match="n_action_steps must be positive"):
        executor.execute(np.array([[0.1, 0.2]]), n_action_steps=n_action_steps)


@pytest.mark.parametrize("control_hz", [0.0, -1.0])
def test_init_rejects_nonpositive_control_hz(control_hz: float) -> None:
    with pytest.raises(ValueError, match="control_hz must be positive"):
        make_executor(control_hz=control_hz)

import numpy as np
import pytest

from pi0fast_wm_rl.robots.mock import MockRobot


def make_robot() -> MockRobot:
    return MockRobot(3, 3, np.zeros(3), -np.ones(3), np.ones(3))


def test_mock_robot_updates_and_clips_state() -> None:
    robot = make_robot()
    robot.connect()
    executed = robot.send_action(np.array([0.2, -0.4, 2.0]))
    np.testing.assert_allclose(robot.get_state(), [0.2, -0.4, 1.0])
    np.testing.assert_allclose(executed, [0.2, -0.4, 1.0])
    assert len(robot.executed_actions) == 1


def test_mock_robot_emergency_stop_requires_reset() -> None:
    robot = make_robot()
    robot.connect()
    robot.emergency_stop()
    with pytest.raises(RuntimeError, match="emergency stop"):
        robot.send_action(np.zeros(3))
    robot.reset()
    robot.send_action(np.zeros(3))

import numpy as np

from pi0fast_wm_rl.cameras.manager import CameraManager
from pi0fast_wm_rl.cameras.mock import MockCamera
from pi0fast_wm_rl.inference.runner import RolloutRunner
from pi0fast_wm_rl.policies.mock_policy import MockPolicy
from pi0fast_wm_rl.robots.mock import MockRobot
from pi0fast_wm_rl.robots.safety import SafetyFilter


def test_mock_policy_robot_rollout_completes_without_hardware() -> None:
    robot = MockRobot(3, 3, np.zeros(3), -np.ones(3), np.ones(3))
    cameras = CameraManager({"front": MockCamera(16, 12, fps=100)})
    safety = SafetyFilter(-np.ones(3), np.ones(3), 0.05, 10.0, 100.0, 1.0)
    runner = RolloutRunner(robot, cameras, MockPolicy(3, chunk_size=4), safety, 100.0, 2, 6)
    robot.connect()
    cameras.connect()
    try:
        result = runner.run_episode(dry_run=True)
    finally:
        cameras.close()
        robot.disconnect()
    assert result.completed
    assert result.steps == 6
    assert result.error is None

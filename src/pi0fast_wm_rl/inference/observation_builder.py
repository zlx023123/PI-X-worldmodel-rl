"""Synchronized observation construction."""

from __future__ import annotations

from pi0fast_wm_rl.cameras.manager import CameraManager
from pi0fast_wm_rl.data.schema import Observation
from pi0fast_wm_rl.robots.base import BaseRobot


class ObservationBuilder:
    """Combine a camera group read and current robot state."""

    def __init__(self, robot: BaseRobot, cameras: CameraManager) -> None:
        self.robot = robot
        self.cameras = cameras

    def build(self) -> Observation:
        camera_read = self.cameras.read()
        return Observation(
            images=camera_read.images,
            state=self.robot.get_state(),
            timestamp=camera_read.timestamp,
        )

"""Headless, deterministic 1000-command rollout on the real Menagerie Panda."""

import numpy as np
import pytest

from pi0fast_wm_rl.policies.action_adapter import ActionAdapter
from pi0fast_wm_rl.robots.mujoco_panda import MujocoPandaRobot
from pi0fast_wm_rl.robots.safety import SafetyFilter


def test_panda_rollout_1000_steps() -> None:
    mujoco = pytest.importorskip("mujoco")
    robot = MujocoPandaRobot()
    if not robot.model_path.is_file():
        pytest.skip(f"MuJoCo Menagerie Panda scene is not installed: {robot.model_path}")

    executed_steps = 0
    try:
        robot.connect()  # Also resets to the XML's home keyframe.
        model, data = robot._model, robot._data
        home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert home_id >= 0, "Stress test requires the existing Panda home keyframe"
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ]
        assert all(joint_id >= 0 for joint_id in joint_ids)
        # send_action uses ctrl[:7]; fail if the selected XML changes that mapping.
        np.testing.assert_array_equal(model.actuator_trnid[:7, 0], joint_ids)
        assert np.all(model.jnt_limited[joint_ids])
        joint_min = model.jnt_range[joint_ids, 0].copy()
        joint_max = model.jnt_range[joint_ids, 1].copy()

        def check_vector(vector: np.ndarray, label: str) -> None:
            assert vector.shape == (7,), f"{label}, step {executed_steps}: {vector.shape}"
            assert np.all(np.isfinite(vector)), f"{label}, step {executed_steps}: {vector}"
            assert np.all(vector >= joint_min), f"{label} below limits at {executed_steps}"
            assert np.all(vector <= joint_max), f"{label} above limits at {executed_steps}"

        home = robot.get_state()
        check_vector(home, "home")
        np.testing.assert_array_equal(home, model.key_qpos[home_id, robot._qpos_ids])
        timestep = float(model.opt.timestep)
        # One send advances one physics step. Use simulation time, not host timing.
        safety = SafetyFilter(
            joint_min=joint_min,
            joint_max=joint_max,
            max_action_delta=0.01,
            max_velocity=1.0,
            control_hz=1.0 / timestep,
            timeout_seconds=1.0,
            action_mode="absolute_joint",
        )
        adapter = ActionAdapter(
            7, 7, model_action_mode="delta_joint", robot_action_mode="absolute_joint"
        )
        step_limit = min(safety.max_action_delta, safety.max_velocity / safety.control_hz)
        initial_time = float(data.time)
        safety.heartbeat(initial_time)
        previous_target = home.copy()
        previous_requested = home.copy()
        max_motion = 0.0
        amplitudes = np.linspace(0.002, 0.005, 7)  # Radians, bounded around home.
        assert np.all(home - amplitudes > joint_min)
        assert np.all(home + amplitudes < joint_max)

        for step in range(1, 1001):
            state = robot.get_state()
            check_vector(state, "state before send")
            requested = home + amplitudes * np.sin(2.0 * np.pi * step / 1000)
            # Replan a state-relative delta each time, never accumulate the trajectory.
            action = adapter.model_action_to_robot(requested - state, state)
            check_vector(action, "adapted action")
            np.testing.assert_allclose(action, requested, rtol=0, atol=1e-12)
            assert np.all(np.abs(action - previous_requested) <= step_limit + 1e-12)
            result = safety.filter(action, state, now=float(data.time))
            target = result.action
            check_vector(target, "filtered action")
            # Check both the filter's state-relative bound and successive sent targets.
            assert np.all(np.abs(target - state) <= step_limit + 1e-12), f"step {step}"
            assert np.all(np.abs(target - previous_target) <= step_limit + 1e-12), (
                f"Target slew exceeded at step {step}"
            )
            sent = robot.send_action(target)
            executed_steps += 1
            check_vector(sent, "returned action")
            np.testing.assert_array_equal(sent, target)
            np.testing.assert_array_equal(data.ctrl[:7], target)
            safety.heartbeat(float(data.time))
            state = robot.get_state()
            check_vector(state, "state after send")
            assert np.all(np.isfinite(data.qvel)), f"Nonfinite velocity at step {step}"
            assert not np.any(data.warning.number), f"MuJoCo warning at step {step}"
            assert data.time == pytest.approx(initial_time + step * timestep, abs=1e-10)
            max_motion = max(max_motion, float(np.max(np.abs(state - home))))
            previous_target = target.copy()
            previous_requested = action.copy()

        assert executed_steps == 1000
        check_vector(robot.get_state(), "final state")
        assert max_motion > 1e-5, "Simulation did not move the arm"
    finally:
        robot.disconnect()
        assert robot.metadata()["connected"] is False
        assert robot._model is None and robot._data is None

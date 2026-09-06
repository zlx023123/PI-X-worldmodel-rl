"""Exercise the real control components with a deterministic MuJoCo backend."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pi0fast_wm_rl.cameras.manager import CameraManager
from pi0fast_wm_rl.cameras.mock import MockCamera
from pi0fast_wm_rl.inference.runner import RolloutRunner
from pi0fast_wm_rl.policies.action_adapter import ActionAdapter
from pi0fast_wm_rl.policies.mock_policy import MockPolicy
from pi0fast_wm_rl.robots.mujoco_panda import MujocoPandaRobot
from pi0fast_wm_rl.robots.safety import SafetyFilter


def make_runner(robot: MujocoPandaRobot) -> RolloutRunner:
    return RolloutRunner(
        robot,
        CameraManager({"front": MockCamera(16, 12, fps=100)}),
        MockPolicy(8, chunk_size=3, target=1.0, max_delta=0.2),
        SafetyFilter(-np.ones(7), np.ones(7), 0.05, 10.0, 100.0, 1.0,
                     action_mode="absolute_joint"),
        control_hz=100.0,
        execute_steps=2,
        max_episode_steps=3,
        action_adapter=ActionAdapter(
            model_dim=8, robot_dim=7, robot_from_model=[6, 5, 4, 3, 2, 1, 0],
            robot_action_mode="absolute_joint",
        ),
    )


@pytest.fixture
def panda(monkeypatch, tmp_path):
    import sys

    initial = np.arange(7, dtype=float) * 0.1
    commands = []

    def reset(model, data):
        data.qpos[:] = initial
        data.ctrl[:] = 0.0

    def step(model, data):
        commands.append(data.ctrl.copy())
        data.qpos[:] = data.ctrl[:7]

    backend = SimpleNamespace(
        MjModel=SimpleNamespace(from_xml_path=lambda path: SimpleNamespace(
            jnt_qposadr=np.arange(7), nu=8)),
        MjData=lambda model: SimpleNamespace(qpos=np.zeros(7), ctrl=np.zeros(8)),
        mjtObj=SimpleNamespace(mjOBJ_JOINT=0, mjOBJ_KEY=1),
        mj_name2id=lambda model, kind, name: -1 if kind == 1 else int(name[5:]) - 1,
        mj_resetData=reset,
        mj_forward=lambda model, data: None,
        mj_step=step,
    )
    monkeypatch.setitem(sys.modules, "mujoco", backend)
    model_path = tmp_path / "scene.xml"
    model_path.touch()
    robot = MujocoPandaRobot(model_path)
    robot.connect()
    yield robot, initial, commands
    robot.disconnect()


@pytest.mark.parametrize("dry_run", [False, True])
def test_full_chain_adapts_clips_and_replans(panda, monkeypatch, dry_run):
    robot, initial, commands = panda
    runner = make_runner(robot)
    runner.executor.sleeper = lambda seconds: None
    observations = []
    predict = runner.policy.predict_action_chunk

    def record(observation):
        observations.append(observation.state.copy())
        return predict(observation)

    monkeypatch.setattr(runner.policy, "predict_action_chunk", record)
    runner.cameras.connect()
    try:
        result = runner.run_episode(dry_run=dry_run)
    finally:
        runner.cameras.close()

    assert result.completed and result.error is None
    assert (result.steps, result.chunks, result.clipped_steps) == (3, 2, 3)
    assert result.clipped_values == 21
    np.testing.assert_allclose(observations[0], np.r_[initial[::-1], 0.0])
    if dry_run:
        assert commands == []
        np.testing.assert_allclose(robot.get_state(), initial)
    else:
        assert len(commands) == 3
        for index, command in enumerate(commands, start=1):
            np.testing.assert_allclose(command[:7], initial + index * 0.05)
            assert command[7] == 0.0
        np.testing.assert_allclose(observations[1], np.r_[(initial + 0.1)[::-1], 0.0])
        np.testing.assert_allclose(robot.get_state(), initial + 0.15)


@pytest.mark.parametrize("fault", ["nan", "emergency_stop"])
def test_safety_rejection_never_reaches_panda(panda, monkeypatch, fault):
    robot, initial, commands = panda
    runner = make_runner(robot)
    predict = runner.policy.predict_action_chunk

    def unsafe_prediction(observation):
        if fault == "emergency_stop":
            runner.safety_filter.emergency_stop()
            return predict(observation)
        return np.full((3, 8), np.nan)

    monkeypatch.setattr(runner.policy, "predict_action_chunk", unsafe_prediction)
    runner.cameras.connect()
    try:
        result = runner.run_episode()
    finally:
        runner.cameras.close()
    assert not result.completed
    assert result.error is not None
    assert commands == []
    np.testing.assert_allclose(robot.get_state(), initial)
    with pytest.raises(RuntimeError, match="stopped"):
        robot.send_action(initial)


@pytest.mark.parametrize("mismatch", ["robot_mode", "adapter_mode", "dimensions"])
def test_reject_incompatible_control_configuration(mismatch):
    runner = make_runner(MujocoPandaRobot())
    if mismatch == "robot_mode":
        runner.safety_filter.action_mode = "delta_joint"
    elif mismatch == "adapter_mode":
        runner.action_adapter.robot_action_mode = "delta_joint"
    else:
        runner.action_adapter = ActionAdapter(8, 6, robot_action_mode="absolute_joint")
    with pytest.raises(ValueError, match="must match"):
        RolloutRunner(
            runner.robot, runner.cameras, runner.policy, runner.safety_filter,
            100.0, 2, 3, action_adapter=runner.action_adapter,
        )


def test_real_mujoco_panda_control_chain():
    pytest.importorskip("mujoco")
    model_path = Path("~/mujoco_menagerie/franka_emika_panda/scene.xml").expanduser()
    if not model_path.is_file():
        pytest.skip("MuJoCo Menagerie Panda scene is not installed")
    robot = MujocoPandaRobot(model_path)
    runner = make_runner(robot)
    # Use the actual model limits, including Panda joint4's negative range.
    robot.connect()
    runner.safety_filter.joint_min = robot._model.jnt_range[:7, 0].copy()
    runner.safety_filter.joint_max = robot._model.jnt_range[:7, 1].copy()
    runner.cameras.connect()
    try:
        initial = robot.get_state()
        result = runner.run_episode()
        assert result.completed and result.steps == 3
        assert robot._data.time > 0
        assert np.all(np.isfinite(robot.get_state()))
        assert not np.array_equal(robot.get_state(), initial)
        assert np.all(robot._data.ctrl[:7] >= runner.safety_filter.joint_min)
        assert np.all(robot._data.ctrl[:7] <= runner.safety_filter.joint_max)
    finally:
        runner.cameras.close()
        robot.disconnect()

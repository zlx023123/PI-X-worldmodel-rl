import numpy as np

from pi0fast_wm_rl.policies.action_adapter import ActionAdapter, GripperMapping, NormalizationSpec


def test_action_adapter_pads_and_reorders_dimensions() -> None:
    adapter = ActionAdapter(model_dim=5, robot_dim=3, robot_from_model=[2, 0, 4])
    np.testing.assert_allclose(
        adapter.robot_state_to_model(np.array([10, 20, 30])), [20, 0, 10, 0, 30]
    )
    np.testing.assert_allclose(adapter.model_action_to_robot(np.array([1, 2, 3])), [3, 1, 0])


def test_action_adapter_denormalizes_maps_gripper_and_converts_mode() -> None:
    normalization = NormalizationSpec(np.zeros(3), np.full(3, 2.0))
    adapter = ActionAdapter(
        model_dim=3,
        robot_dim=3,
        action_normalization=normalization,
        gripper=GripperMapping(2, 2, (-2.0, 2.0), (0.0, 1.0)),
        model_action_mode="absolute_joint",
        robot_action_mode="delta_joint",
    )
    result = adapter.model_action_to_robot(
        np.array([0.25, -0.25, 0.0]), np.array([0.1, -0.1, 0.25])
    )
    np.testing.assert_allclose(result, [0.4, -0.4, 0.25])

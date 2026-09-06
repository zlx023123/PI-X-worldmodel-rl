# Panda control integration

`RolloutRunner` accepts an optional `action_adapter`. Existing callers can omit
it. For Panda, configure both the adapter output and `SafetyFilter` as
`absolute_joint`, with seven arm joints (no gripper command).

The runner builds an observation, maps its robot state into model space, calls
`BasePolicy.predict_action_chunk`, and converts the returned chunk with
`ActionAdapter.model_chunk_to_robot`. `ActionChunkExecutor` then reads the current
robot state for **each** command, applies `SafetyFilter`, and sends the filtered
absolute target to `MujocoPandaRobot.send_action`.

Mode conversion retains the existing adapter semantics: every row of a delta
chunk is relative to the observation at inference time, not cumulatively added
to previous rows. Joint/velocity limits are still checked against the latest
state at execution time. Policies producing incremental per-step deltas should
use `execute_steps=1` and replan after each step.

## Minimal simulation example

From the repository root, install the project with the MuJoCo optional extra
(and the development tools needed to run the tests):

```bash
python -m pip install -e ".[mujoco,dev]"
```

The `mujoco` extra supports `mujoco>=3.12,<4`. It does not include the model assets.
The example also requires the MuJoCo Menagerie Panda model at
`~/mujoco_menagerie/franka_emika_panda/scene.xml`. The limits below match that model;
if using another model, supply its limits. The camera and policy are mocks; the
robot uses actual MuJoCo simulation.

```python
import numpy as np

from pi0fast_wm_rl.cameras.manager import CameraManager
from pi0fast_wm_rl.cameras.mock import MockCamera
from pi0fast_wm_rl.inference.runner import RolloutRunner
from pi0fast_wm_rl.policies.action_adapter import ActionAdapter
from pi0fast_wm_rl.policies.mock_policy import MockPolicy
from pi0fast_wm_rl.robots.mujoco_panda import MujocoPandaRobot
from pi0fast_wm_rl.robots.safety import SafetyFilter

robot = MujocoPandaRobot()
cameras = CameraManager({"front": MockCamera(16, 12, fps=100)})
safety = SafetyFilter(
    joint_min=np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    joint_max=np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
    max_action_delta=0.01,
    max_velocity=1.0,
    control_hz=100.0,
    timeout_seconds=1.0,
    action_mode="absolute_joint",
)
runner = RolloutRunner(
    robot, cameras, MockPolicy(7, chunk_size=1), safety,
    control_hz=100.0, execute_steps=1, max_episode_steps=10,
    action_adapter=ActionAdapter(7, 7, robot_action_mode="absolute_joint"),
)
try:
    robot.connect()
    cameras.connect()
    print(runner.run_episode().to_dict())
finally:
    cameras.close()
    robot.disconnect()
```

Connection ownership remains with the caller. The Panda adapter still performs
one `mj_step` per command; executor wall-clock frequency does not change the
model's simulation timestep. This integration does not add a dynamics controller
or change that behavior.

## Tests

Run `pytest tests/test_panda_control_integration.py`. Deterministic tests replace
only the MuJoCo backend and exercise all actual control components, including
Panda's lifecycle and command dispatch. The real simulation test runs when both
MuJoCo and the default Menagerie scene are available; otherwise it reports a skip.

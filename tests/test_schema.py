import json

import numpy as np

from pi0fast_wm_rl.data.schema import EpisodeStep, Observation


def test_episode_step_is_json_serializable() -> None:
    step = EpisodeStep(
        observation=Observation(
            images={"front": np.zeros((2, 3, 3), dtype=np.uint8)},
            state=np.array([0.1, 0.2]),
            timestamp=0.5,
        ),
        action=np.array([0.01, -0.01]),
        task="pick up the cube",
        episode_index=1,
        frame_index=2,
    )
    encoded = json.dumps(step.to_dict())
    assert "pick up the cube" in encoded


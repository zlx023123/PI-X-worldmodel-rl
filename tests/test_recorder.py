from pathlib import Path

import numpy as np

from pi0fast_wm_rl.data.episode_reader import iter_episode_steps, load_episode_arrays
from pi0fast_wm_rl.data.validator import DatasetValidator


def test_recorder_generates_readable_valid_episode(tmp_path: Path, dataset_factory) -> None:
    dataset = dataset_factory(tmp_path, episodes=1, frames=3)
    episode = dataset / "episode_000000"
    assert (episode / "metadata.json").is_file()
    assert (episode / "steps.npz").is_file()
    assert len(list((episode / "images/front").glob("*.png"))) == 3
    arrays = load_episode_arrays(episode)
    assert arrays["state"].shape == (3, 3)
    steps = list(iter_episode_steps(episode))
    assert len(steps) == 3
    assert steps[0].task == "put the cube in the bowl"
    assert DatasetValidator().validate(dataset).valid


def test_recorder_preserves_numeric_values(tmp_path: Path, dataset_factory) -> None:
    dataset = dataset_factory(tmp_path, episodes=1, frames=2)
    arrays = load_episode_arrays(dataset / "episode_000000")
    np.testing.assert_allclose(arrays["action"][1], [0.01, 0.01, 0.01])

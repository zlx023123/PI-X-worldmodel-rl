from pathlib import Path

from pi0fast_wm_rl.data.splitter import split_dataset


def test_splitter_uses_disjoint_episode_indices(tmp_path: Path, dataset_factory) -> None:
    dataset = dataset_factory(tmp_path, episodes=10, frames=1)
    result = split_dataset(dataset, 0.8, 0.1, 0.1, seed=42)
    assert [len(result[name]) for name in ("train", "validation", "test")] == [8, 1, 1]
    sets = [set(values) for values in result.values()]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert set.union(*sets) == set(range(10))


def test_splitter_is_seed_deterministic(tmp_path: Path, dataset_factory) -> None:
    dataset = dataset_factory(tmp_path, episodes=6, frames=1)
    assert split_dataset(dataset, 0.5, 0.25, 0.25, 7) == split_dataset(dataset, 0.5, 0.25, 0.25, 7)

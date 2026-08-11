"""Deterministic episode-level dataset splitting."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .episode_reader import episode_directories, load_episode_payload


def _allocate(total: int, ratios: tuple[float, float, float]) -> tuple[int, int, int]:
    exact = np.asarray(ratios, dtype=np.float64) * total
    counts = np.floor(exact).astype(int)
    for index in np.argsort(-(exact - counts))[: total - int(counts.sum())]:
        counts[index] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def split_dataset(
    dataset_dir: str | Path,
    train: float,
    val: float,
    test: float,
    seed: int = 42,
    stratify_by_task: bool = False,
) -> dict[str, list[int]]:
    """Return train/validation/test episode indices; frames are never split."""
    ratios = (float(train), float(val), float(test))
    if any(value < 0 for value in ratios) or not np.isclose(sum(ratios), 1.0, atol=1e-9):
        raise ValueError("train, val, and test must be non-negative and sum to 1")
    directories = episode_directories(dataset_dir)
    if not directories:
        raise ValueError("Cannot split an empty dataset")
    groups: dict[str, list[int]] = defaultdict(list)
    for path in directories:
        payload = load_episode_payload(path)
        metadata = payload.get("episode", {})
        index = int(metadata["episode_index"])
        key = str(metadata.get("task", "")) if stratify_by_task else "all"
        groups[key].append(index)
    rng = np.random.default_rng(seed)
    result: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for key in sorted(groups):
        indices = np.asarray(sorted(groups[key]), dtype=np.int64)
        rng.shuffle(indices)
        train_count, val_count, _ = _allocate(len(indices), ratios)
        result["train"].extend(int(value) for value in indices[:train_count])
        result["validation"].extend(
            int(value) for value in indices[train_count : train_count + val_count]
        )
        result["test"].extend(int(value) for value in indices[train_count + val_count :])
    for values in result.values():
        values.sort()
    return result


def save_splits(
    splits: dict[str, list[int]],
    output_path: str | Path,
    dataset_dir: str | Path,
    seed: int,
    stratify_by_task: bool,
) -> Path:
    """Save a portable split manifest containing episode indices only."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "dataset": str(Path(dataset_dir).resolve()),
        "seed": seed,
        "stratify_by_task": stratify_by_task,
        "splits": splits,
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target

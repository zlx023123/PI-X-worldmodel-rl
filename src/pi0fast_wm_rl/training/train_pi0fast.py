"""PI0-FAST fine-tuning command adapter."""

from __future__ import annotations

from pathlib import Path

from pi0fast_wm_rl.utils.config import load_yaml

from .common import TrainingPlan, build_training_plan, run_training_plan


def prepare(config_path: str | Path) -> TrainingPlan:
    return build_training_plan(load_yaml(config_path), "pi0_fast")


def run(config_path: str | Path, dry_run: bool = False) -> int:
    return run_training_plan(prepare(config_path), dry_run)

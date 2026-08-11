"""Build and optionally execute verified LeRobot 0.6 training commands."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from typing import Any

from pi0fast_wm_rl.utils.config import resolve_path, validate_training_config

from .dependency_check import dependency_report


@dataclass(slots=True)
class TrainingPlan:
    policy_type: str
    command: list[str]
    dataset_exists: bool
    weights_exist: bool | None
    dependencies: list[dict[str, Any]]
    ready: bool
    next_steps: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def pretty_command(self) -> str:
        return shlex.join(self.command)


def _arg(name: str, value: Any) -> str:
    if isinstance(value, bool):
        value = str(value).lower()
    return f"--{name}={value}"


def build_training_plan(config: dict[str, Any], policy_type: str) -> TrainingPlan:
    """Validate YAML content and generate a LeRobot 0.6 CLI invocation."""
    validate_training_config(config, policy_type)
    training = config["training"]
    policy = config["policy"]
    dataset_path = resolve_path(training["dataset_root"])
    command = [
        "lerobot-train",
        _arg("dataset.repo_id", training["dataset_repo_id"]),
        _arg("dataset.root", dataset_path),
        _arg("output_dir", resolve_path(training["output_dir"])),
        _arg("job_name", training["job_name"]),
        _arg("steps", training["steps"]),
        _arg("batch_size", training["batch_size"]),
        _arg("seed", training["seed"]),
    ]
    weights_exist: bool | None = None
    if policy_type == "pi0_fast":
        for required in (
            "pretrained_path",
            "action_tokenizer_name",
            "max_action_tokens",
            "dtype",
            "gradient_checkpointing",
        ):
            if required not in policy:
                raise ValueError(f"Missing required PI0-FAST field: policy.{required}")
        pretrained_path = resolve_path(policy["pretrained_path"])
        tokenizer_path = resolve_path(policy["action_tokenizer_name"])
        weights_exist = pretrained_path.is_dir() and tokenizer_path.is_dir()
        command.append(_arg("policy.path", pretrained_path))
        command.extend(
            [
                _arg("policy.action_tokenizer_name", tokenizer_path),
                _arg("policy.max_action_tokens", policy["max_action_tokens"]),
                _arg("policy.dtype", policy["dtype"]),
                _arg("policy.gradient_checkpointing", policy["gradient_checkpointing"]),
            ]
        )
    else:
        command.append(_arg("policy.type", "act"))
    command.extend(
        [
            _arg("policy.device", training["device"]),
            _arg("policy.chunk_size", policy["chunk_size"]),
            _arg("policy.n_action_steps", policy["n_action_steps"]),
        ]
    )
    dependencies = [row.to_dict() for row in dependency_report()]
    deps_ready = all(
        row["available"]
        for row in dependencies
        if row["name"] in {"torch", "lerobot", "lerobot-train"}
    )
    next_steps: list[str] = []
    if not dataset_path.is_dir():
        next_steps.append(
            "Convert the internal dataset with data.lerobot_adapter.convert_to_lerobot()."
        )
    if weights_exist is False:
        next_steps.append(
            "Run scripts/download_models.py after reviewing the model IDs and license terms."
        )
    if not deps_ready:
        next_steps.append("Install the training stack with: pip install -e '.[lerobot]'")
    ready = dataset_path.is_dir() and deps_ready and weights_exist is not False
    return TrainingPlan(
        policy_type, command, dataset_path.is_dir(), weights_exist, dependencies, ready, next_steps
    )


def run_training_plan(plan: TrainingPlan, dry_run: bool) -> int:
    """Print a plan in dry-run mode or execute only when every preflight check passes."""
    print(json.dumps(plan.to_dict(), indent=2))
    print(f"command: {plan.pretty_command()}")
    if dry_run:
        return 0
    if not plan.ready:
        raise RuntimeError(
            "Training preflight failed; resolve next_steps before running without --dry-run"
        )
    completed = subprocess.run(plan.command, check=False)
    return completed.returncode

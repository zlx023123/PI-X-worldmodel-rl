"""Unified command-line interface for the phase-one pipeline."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from importlib import metadata, util
from pathlib import Path
from typing import Any

import numpy as np

from pi0fast_wm_rl.cameras.manager import CameraManager
from pi0fast_wm_rl.cameras.mock import MockCamera
from pi0fast_wm_rl.data.episode_reader import episode_directories, load_episode_payload
from pi0fast_wm_rl.data.lerobot_adapter import convert_to_lerobot
from pi0fast_wm_rl.data.normalization import compute_normalization_stats, save_normalization_stats
from pi0fast_wm_rl.data.recorder import EpisodeRecorder
from pi0fast_wm_rl.data.schema import EpisodeMetadata, Observation
from pi0fast_wm_rl.data.splitter import save_splits, split_dataset
from pi0fast_wm_rl.data.validator import DatasetValidator
from pi0fast_wm_rl.evaluation.offline import evaluate_offline
from pi0fast_wm_rl.evaluation.report import write_report
from pi0fast_wm_rl.evaluation.rollout import summarize_rollouts
from pi0fast_wm_rl.inference.runner import RolloutRunner
from pi0fast_wm_rl.policies.mock_policy import MockPolicy
from pi0fast_wm_rl.robots.mock import MockRobot
from pi0fast_wm_rl.robots.safety import SafetyFilter
from pi0fast_wm_rl.teleoperation.keyboard import KeyboardTeleoperator
from pi0fast_wm_rl.teleoperation.mock import MockTeleoperator
from pi0fast_wm_rl.training import train_act, train_pi0fast
from pi0fast_wm_rl.utils.config import (
    ConfigError,
    load_yaml,
    project_root,
    resolve_path,
    validate_camera_config,
    validate_robot_config,
    validate_task_config,
)
from pi0fast_wm_rl.utils.device import torch_device_info


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def doctor() -> dict[str, Any]:
    """Collect environment status while treating optional components as warnings."""
    root = project_root()
    config_checks = [
        (root / "configs" / "pretrained_models.yaml", None),
        (root / "configs" / "robot" / "mock_robot.yaml", validate_robot_config),
        (root / "configs" / "camera" / "mock_camera.yaml", validate_camera_config),
        (root / "configs" / "task" / "pick_place.yaml", validate_task_config),
    ]
    config_status: dict[str, Any] = {}
    pretrained: dict[str, Any] = {}
    for path, validator in config_checks:
        relative = str(path.relative_to(root))
        try:
            loaded = load_yaml(path)
            if validator is not None:
                validator(loaded)
            elif path.name == "pretrained_models.yaml":
                pretrained = loaded
            config_status[relative] = {"exists": True, "valid": True, "error": None}
        except (ConfigError, KeyError, TypeError, ValueError) as exc:
            config_status[relative] = {
                "exists": path.is_file(),
                "valid": False,
                "error": str(exc),
            }
    lerobot_version = _version("lerobot")
    lerobot_compatible = False
    if lerobot_version:
        try:
            lerobot_compatible = tuple(int(value) for value in lerobot_version.split(".")[:2]) == (
                0,
                6,
            )
        except ValueError:
            pass
    model_status = {
        name: {
            "model_id": value["model_id"],
            "local_path": str(root / value["local_path"]),
            "available": any(
                path.is_file() and path.name != ".gitkeep"
                for path in (root / value["local_path"]).rglob("*")
            )
            if (root / value["local_path"]).is_dir()
            else False,
        }
        for name, value in pretrained.items()
        if name != "compatibility" and isinstance(value, dict)
    }
    return {
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "supported": sys.version_info[:2] in {(3, 12), (3, 13)},
            "target": "3.12",
        },
        "torch": torch_device_info(),
        "lerobot": {
            "installed": lerobot_version is not None,
            "version": lerobot_version,
            "verified_compatible": lerobot_compatible,
            "required_for_mock": False,
            "verified_range": ">=0.6,<0.7",
        },
        "opencv": {
            "installed": util.find_spec("cv2") is not None,
            "version": _version("opencv-python") or _version("opencv-python-headless"),
            "required_for_mock": False,
        },
        "configs": config_status,
        "models": model_status,
        "recommendations": [
            "Mock mode is available without LeRobot, CUDA, cameras, or model weights.",
            "For training, install: pip install -e '.[lerobot]' (this may upgrade PyTorch).",
            "Review model licenses before explicitly running scripts/download_models.py.",
        ],
    }


def _load_runtime_configs(
    task_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    task = validate_task_config(load_yaml(task_path))
    robot = validate_robot_config(load_yaml(project_root() / "configs/robot/mock_robot.yaml"))
    camera = validate_camera_config(load_yaml(project_root() / "configs/camera/mock_camera.yaml"))
    control = task["control"]
    if int(robot["robot"]["state_dim"]) != int(control["state_dim"]):
        raise ConfigError("Robot and task state dimensions differ")
    if int(robot["robot"]["action_dim"]) != int(control["action_dim"]):
        raise ConfigError("Robot and task action dimensions differ")
    return task, robot, camera


def _mock_components(
    task_config: dict[str, Any], robot_config: dict[str, Any], camera_config: dict[str, Any]
) -> tuple[MockRobot, CameraManager, SafetyFilter]:
    robot_values = robot_config["robot"]
    camera_values = camera_config["camera"]
    control = task_config["control"]
    safety_values = task_config["safety"]
    robot = MockRobot(
        state_dim=int(robot_values["state_dim"]),
        action_dim=int(robot_values["action_dim"]),
        initial_state=np.asarray(robot_values["initial_state"]),
        joint_min=np.asarray(robot_values["joint_min"]),
        joint_max=np.asarray(robot_values["joint_max"]),
        deterministic=bool(robot_values.get("deterministic", True)),
    )
    camera = MockCamera(
        width=int(camera_values["width"]),
        height=int(camera_values["height"]),
        fps=float(camera_values["fps"]),
        seed=int(camera_values.get("seed", 42)),
        name=str(camera_values["name"]),
    )
    safety_filter = SafetyFilter(
        joint_min=np.asarray(robot_values["joint_min"]),
        joint_max=np.asarray(robot_values["joint_max"]),
        max_action_delta=float(safety_values["max_action_delta"]),
        max_velocity=float(safety_values["max_velocity"]),
        control_hz=float(control["frequency_hz"]),
        timeout_seconds=float(safety_values["timeout_seconds"]),
        action_mode=str(control["action_mode"]),
    )
    return robot, CameraManager({str(camera_values["name"]): camera}), safety_filter


def record_mock(args: argparse.Namespace) -> dict[str, Any]:
    """Record deterministic demonstrations as quickly as the local filesystem permits."""
    if (args.robot, args.camera, args.teleop) not in {
        ("mock", "mock", "mock"),
        ("mock", "mock", "keyboard"),
    }:
        raise ValueError("Phase one record supports mock robot/camera with mock or keyboard teleop")
    task_config, robot_config, camera_config = _load_runtime_configs(args.task)
    task_values, control, safety_values = (
        task_config["task"],
        task_config["control"],
        task_config["safety"],
    )
    dataset_dir = (
        Path(args.output) if args.output else project_root() / "data/raw" / task_values["name"]
    )
    existing = episode_directories(dataset_dir) if dataset_dir.is_dir() else []
    start_index = (
        max(
            (int(load_episode_payload(path)["episode"]["episode_index"]) for path in existing),
            default=-1,
        )
        + 1
    )
    steps_per_episode = int(args.max_steps or task_values["max_episode_steps"])
    robot, cameras, safety_filter = _mock_components(task_config, robot_config, camera_config)
    teleop = (
        MockTeleoperator(int(control["action_dim"]))
        if args.teleop == "mock"
        else KeyboardTeleoperator(int(control["action_dim"]))
    )
    robot.connect()
    cameras.connect()
    created: list[str] = []
    try:
        for offset in range(args.episodes):
            episode_index = start_index + offset
            robot.reset()
            teleop.reset()
            safety_filter.reset()
            safety_filter.heartbeat()
            action_bound = min(
                float(safety_values["max_action_delta"]),
                float(safety_values["max_velocity"]) / float(control["frequency_hz"]),
            )
            episode_metadata = EpisodeMetadata(
                episode_index=episode_index,
                task=str(task_values["instruction"]),
                robot_type="mock",
                created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                fps=float(control["frequency_hz"]),
                control_mode=str(control["action_mode"]),
                extra={
                    "task_name": task_values["name"],
                    "synthetic": True,
                    "action_bounds": {
                        "min": [-action_bound] * int(control["action_dim"]),
                        "max": [action_bound] * int(control["action_dim"]),
                    },
                },
            )
            recorder = EpisodeRecorder(
                dataset_dir,
                episode_metadata,
                int(control["state_dim"]),
                int(control["action_dim"]),
                ("front",),
            )
            recorder.start()
            for frame_index in range(steps_per_episode):
                read = cameras.read()
                state = robot.get_state()
                observation = Observation(
                    read.images,
                    state,
                    frame_index / float(control["frequency_hz"]),
                )
                proposed = teleop.get_action(state)
                safe = safety_filter.filter(proposed, state)
                recorder.record(observation, safe.action)
                robot.send_action(safe.action)
                safety_filter.heartbeat()
            created.append(str(recorder.finish(success=True)))
    finally:
        cameras.close()
        robot.disconnect()
    return {"dataset": str(dataset_dir), "episodes_created": len(created), "paths": created}


def run_rollouts(args: argparse.Namespace) -> dict[str, Any]:
    if (args.robot, args.camera, args.policy) != ("mock", "mock", "mock"):
        raise ValueError("Phase one CLI rollout currently supports only mock components")
    task_config, robot_config, camera_config = _load_runtime_configs(args.task)
    robot, cameras, safety_filter = _mock_components(task_config, robot_config, camera_config)
    control = task_config["control"]
    policy = MockPolicy(int(control["action_dim"]), int(control["chunk_size"]))
    runner = RolloutRunner(
        robot,
        cameras,
        policy,
        safety_filter,
        float(control["frequency_hz"]),
        int(control["execute_steps"]),
        int(args.max_steps or task_config["task"]["max_episode_steps"]),
    )
    robot.connect()
    cameras.connect()
    try:
        results = [runner.run_episode(dry_run=args.dry_run) for _ in range(args.episodes)]
    finally:
        cameras.close()
        robot.disconnect()
    return summarize_rollouts(results)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pi0fast-wm-rl", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="Inspect required and optional environment components")

    record = subparsers.add_parser("record", help="Record a hardware-free demonstration dataset")
    record.add_argument("--robot", default="mock")
    record.add_argument("--camera", default="mock")
    record.add_argument("--teleop", default="mock")
    record.add_argument("--task", default="configs/task/pick_place.yaml")
    record.add_argument("--episodes", type=int, required=True)
    record.add_argument("--output")
    record.add_argument("--max-steps", type=int)

    inspect = subparsers.add_parser("inspect", help="Validate an internal dataset")
    inspect.add_argument("--dataset", required=True)

    split = subparsers.add_parser("split", help="Create episode-level split manifests")
    split.add_argument("--dataset", required=True)
    split.add_argument("--train", type=float, required=True)
    split.add_argument("--val", type=float, required=True)
    split.add_argument("--test", type=float, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--stratify-by-task", action="store_true")
    split.add_argument("--output")

    norm = subparsers.add_parser("norm-stats", help="Compute state/action normalization statistics")
    norm.add_argument("--dataset", required=True)
    norm.add_argument("--output")

    convert = subparsers.add_parser(
        "convert-lerobot", help="Convert internal data using LeRobot 0.6"
    )
    convert.add_argument("--dataset", required=True)
    convert.add_argument("--output", required=True)
    convert.add_argument("--repo-id", required=True)
    convert.add_argument("--fps", type=int, required=True)

    for command, default in (
        ("train-act", "configs/policy/act.yaml"),
        ("train-pi0fast", "configs/policy/pi0fast_bc.yaml"),
    ):
        train = subparsers.add_parser(
            command, help=f"Validate and invoke {command} through LeRobot"
        )
        train.add_argument("--config", default=default)
        train.add_argument("--dry-run", action="store_true")

    evaluate = subparsers.add_parser("eval-offline", help="Evaluate MockPolicy on recorded actions")
    evaluate.add_argument("--dataset", required=True)
    evaluate.add_argument("--output", default="outputs/reports")

    rollout = subparsers.add_parser("rollout", help="Run a safety-filtered bounded rollout")
    rollout.add_argument("--robot", default="mock")
    rollout.add_argument("--camera", default="mock")
    rollout.add_argument("--policy", default="mock")
    rollout.add_argument("--task", default="configs/task/pick_place.yaml")
    rollout.add_argument("--episodes", type=int, required=True)
    rollout.add_argument("--max-steps", type=int)
    rollout.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            print(json.dumps(doctor(), indent=2))
            return 0
        if args.command == "record":
            if args.episodes <= 0:
                raise ValueError("--episodes must be positive")
            result = record_mock(args)
        elif args.command == "inspect":
            result = DatasetValidator().validate(resolve_path(args.dataset)).to_dict()
            print(json.dumps(result, indent=2))
            return 0 if result["valid"] else 1
        elif args.command == "split":
            dataset = resolve_path(args.dataset)
            splits = split_dataset(
                dataset, args.train, args.val, args.test, args.seed, args.stratify_by_task
            )
            output = (
                Path(args.output)
                if args.output
                else project_root() / "data/splits" / f"{dataset.name}.json"
            )
            saved = save_splits(splits, output, dataset, args.seed, args.stratify_by_task)
            result = {"output": str(saved), "splits": splits}
        elif args.command == "norm-stats":
            dataset = resolve_path(args.dataset)
            stats = compute_normalization_stats(dataset)
            output = (
                Path(args.output)
                if args.output
                else project_root() / "models/normalization" / f"{dataset.name}.json"
            )
            saved = save_normalization_stats(stats, output)
            result = {"output": str(saved), **stats}
        elif args.command == "convert-lerobot":
            result = {
                "output": str(
                    convert_to_lerobot(
                        resolve_path(args.dataset),
                        resolve_path(args.output),
                        args.repo_id,
                        args.fps,
                    )
                )
            }
        elif args.command == "train-act":
            return train_act.run(args.config, args.dry_run)
        elif args.command == "train-pi0fast":
            return train_pi0fast.run(args.config, args.dry_run)
        elif args.command == "eval-offline":
            dataset = resolve_path(args.dataset)
            first = load_episode_payload(episode_directories(dataset)[0])
            policy = MockPolicy(int(first["action_dim"]))
            metrics = evaluate_offline(dataset, policy)
            paths = write_report(metrics, resolve_path(args.output))
            result = {**metrics, "reports": [str(path) for path in paths]}
        elif args.command == "rollout":
            if args.episodes <= 0:
                raise ValueError("--episodes must be positive")
            result = run_rollouts(args)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
        print(json.dumps(result, indent=2))
        return 0
    except (ConfigError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

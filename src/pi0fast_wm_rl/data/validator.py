"""Structural and numerical validation for internal datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from .episode_reader import episode_directories, load_episode_arrays, load_episode_payload


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    code: str
    message: str
    episode: str | None = None


@dataclass(slots=True)
class ValidationReport:
    dataset: str
    episode_count: int = 0
    frame_count: int = 0
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "valid": self.valid,
            "episode_count": self.episode_count,
            "frame_count": self.frame_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


class DatasetValidator:
    """Validate counts, dimensions, values, timestamps, bounds, and tasks."""

    def __init__(
        self,
        action_min: np.ndarray | None = None,
        action_max: np.ndarray | None = None,
    ) -> None:
        self.action_min = None if action_min is None else np.asarray(action_min, dtype=np.float64)
        self.action_max = None if action_max is None else np.asarray(action_max, dtype=np.float64)
        if (self.action_min is None) != (self.action_max is None):
            raise ValueError("action_min and action_max must be supplied together")

    def validate(self, dataset_dir: str | Path) -> ValidationReport:
        root = Path(dataset_dir)
        report = ValidationReport(dataset=str(root))
        try:
            directories = episode_directories(root)
        except FileNotFoundError as exc:
            report.issues.append(ValidationIssue("error", "dataset_missing", str(exc)))
            return report
        report.episode_count = len(directories)
        if not directories:
            report.issues.append(
                ValidationIssue("error", "dataset_empty", "Dataset has no episodes")
            )
            return report
        for episode_dir in directories:
            self._validate_episode(episode_dir, report)
        return report

    @staticmethod
    def _issue(report: ValidationReport, code: str, message: str, episode: Path) -> None:
        report.issues.append(ValidationIssue("error", code, message, episode.name))

    def _validate_episode(self, episode_dir: Path, report: ValidationReport) -> None:
        try:
            payload = load_episode_payload(episode_dir)
            arrays = load_episode_arrays(episode_dir)
        except (FileNotFoundError, ValueError, KeyError) as exc:
            self._issue(report, "episode_unreadable", str(exc), episode_dir)
            return
        required_arrays = {"state", "action", "timestamp"}
        missing = required_arrays - arrays.keys()
        if missing:
            self._issue(report, "arrays_missing", f"Missing arrays: {sorted(missing)}", episode_dir)
            return
        state, action, timestamps = arrays["state"], arrays["action"], arrays["timestamp"]
        lengths = {len(state), len(action), len(timestamps)}
        expected_frames = int(payload.get("num_frames", -1))
        if len(lengths) != 1 or (expected_frames >= 0 and len(state) != expected_frames):
            self._issue(
                report,
                "frame_count_mismatch",
                "Numeric frame counts disagree: "
                f"state={len(state)}, action={len(action)}, timestamp={len(timestamps)}, "
                f"metadata={expected_frames}",
                episode_dir,
            )
        frame_count = min(len(state), len(action), len(timestamps))
        report.frame_count += frame_count
        if frame_count == 0:
            self._issue(report, "episode_empty", "Episode contains no frames", episode_dir)
        state_dim = int(payload.get("state_dim", -1))
        action_dim = int(payload.get("action_dim", -1))
        if state.ndim != 2 or state.shape[1:] != (state_dim,):
            self._issue(
                report,
                "state_dimension",
                f"Expected state (*, {state_dim}), got {state.shape}",
                episode_dir,
            )
        if action.ndim != 2 or action.shape[1:] != (action_dim,):
            self._issue(
                report,
                "action_dimension",
                f"Expected action (*, {action_dim}), got {action.shape}",
                episode_dir,
            )
        if not np.all(np.isfinite(state)):
            self._issue(report, "state_nonfinite", "State contains NaN or Inf", episode_dir)
        if not np.all(np.isfinite(action)):
            self._issue(report, "action_nonfinite", "Action contains NaN or Inf", episode_dir)
        if not np.all(np.isfinite(timestamps)):
            self._issue(report, "timestamp_nonfinite", "Timestamp contains NaN or Inf", episode_dir)
        elif len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
            self._issue(
                report,
                "timestamp_nonmonotonic",
                "Timestamps must be strictly increasing",
                episode_dir,
            )
        episode_metadata = payload.get("episode", {})
        if (
            not isinstance(episode_metadata, dict)
            or not str(episode_metadata.get("task", "")).strip()
        ):
            self._issue(report, "task_missing", "Episode task is missing", episode_dir)
        cameras = payload.get("cameras", [])
        if not isinstance(cameras, list) or "front" not in cameras:
            self._issue(
                report, "front_camera_missing", "Front camera metadata is missing", episode_dir
            )
        else:
            for camera in cameras:
                count = len(list((episode_dir / "images" / str(camera)).glob("frame_*.png")))
                if count != len(action):
                    self._issue(
                        report,
                        "image_count_mismatch",
                        f"Camera '{camera}' has {count} images but {len(action)} actions",
                        episode_dir,
                    )
        minimum, maximum = self._bounds_from_metadata(episode_metadata, action_dim)
        if minimum is not None and action.ndim == 2 and action.shape[1] == len(minimum):
            if np.any(action < minimum) or np.any(action > maximum):
                self._issue(
                    report, "action_out_of_bounds", "Action exceeds configured bounds", episode_dir
                )

    def _bounds_from_metadata(
        self, metadata: dict[str, object], action_dim: int
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.action_min is not None:
            if self.action_min.shape != (action_dim,) or self.action_max.shape != (action_dim,):
                return None, None
            return self.action_min, self.action_max
        extra = metadata.get("extra", {})
        if not isinstance(extra, dict):
            return None, None
        bounds = extra.get("action_bounds")
        if not isinstance(bounds, dict):
            return None, None
        minimum = np.asarray(bounds.get("min", []), dtype=np.float64)
        maximum = np.asarray(bounds.get("max", []), dtype=np.float64)
        if minimum.shape != (action_dim,) or maximum.shape != (action_dim,):
            return None, None
        return minimum, maximum

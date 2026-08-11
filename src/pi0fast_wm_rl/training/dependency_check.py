"""Training dependency and interface checks."""

from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from importlib import metadata, util
from typing import Any


@dataclass(frozen=True, slots=True)
class DependencyStatus:
    name: str
    available: bool
    version: str | None
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dependency_report() -> list[DependencyStatus]:
    """Inspect optional training components without importing model implementations."""
    rows: list[DependencyStatus] = []
    for distribution, module in (("torch", "torch"), ("lerobot", "lerobot")):
        importable = util.find_spec(module) is not None
        try:
            version = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            version = None
        compatible = importable and version is not None
        if compatible:
            try:
                major_minor = tuple(int(part) for part in version.split("+")[0].split(".")[:2])
                compatible = (
                    major_minor >= (2, 7) if distribution == "torch" else major_minor == (0, 6)
                )
            except ValueError:
                compatible = False
        if compatible:
            detail = "installed and compatible"
        elif importable:
            required = ">=2.7" if distribution == "torch" else ">=0.6,<0.7"
            detail = f"installed version is outside the verified {required} range"
        else:
            detail = f"install optional dependency for {distribution}"
        rows.append(
            DependencyStatus(
                distribution,
                compatible,
                version,
                detail,
            )
        )
    executable = shutil.which("lerobot-train")
    rows.append(
        DependencyStatus(
            "lerobot-train",
            executable is not None,
            None,
            executable
            or "console command not found; install pip package 'lerobot[training,pi]>=0.6,<0.7'",
        )
    )
    return rows

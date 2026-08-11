from importlib import metadata

import pytest


def test_all_top_level_modules_import_without_lerobot() -> None:
    pass


def test_missing_lerobot_has_friendly_pi0fast_error(monkeypatch) -> None:
    from pi0fast_wm_rl.data import lerobot_adapter
    from pi0fast_wm_rl.policies.pi0fast_adapter import Pi0FastAdapter

    original = lerobot_adapter.metadata.version

    def missing(name: str) -> str:
        if name == "lerobot":
            raise metadata.PackageNotFoundError(name)
        return original(name)

    monkeypatch.setattr(lerobot_adapter.metadata, "version", missing)
    with pytest.raises(RuntimeError, match="LeRobot is not installed"):
        Pi0FastAdapter("models/pretrained/pi0fast-base", "pick up the cube")


def test_cli_doctor_runs_with_optional_dependencies_missing(capsys) -> None:
    from pi0fast_wm_rl.cli import main

    assert main(["doctor"]) == 0
    output = capsys.readouterr().out
    assert '"python"' in output
    assert '"lerobot"' in output

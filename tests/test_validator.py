from pathlib import Path

from pi0fast_wm_rl.data.validator import DatasetValidator


def test_validator_detects_image_action_count_mismatch(tmp_path: Path, dataset_factory) -> None:
    dataset = dataset_factory(tmp_path, episodes=1, frames=3)
    (dataset / "episode_000000/images/front/frame_000002.png").unlink()
    report = DatasetValidator().validate(dataset)
    assert not report.valid
    assert "image_count_mismatch" in {issue.code for issue in report.issues}


def test_validator_reports_empty_dataset(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    report = DatasetValidator().validate(tmp_path)
    assert not report.valid
    assert report.issues[0].code == "dataset_empty"

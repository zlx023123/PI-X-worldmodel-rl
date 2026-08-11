from pathlib import Path

import numpy as np

from pi0fast_wm_rl.data.normalization import compute_normalization_stats


def test_normalization_statistics_include_required_fields(tmp_path: Path, dataset_factory) -> None:
    dataset = dataset_factory(tmp_path, episodes=1, frames=3, dim=2)
    stats = compute_normalization_stats(dataset)
    assert stats["frame_count"] == 3
    np.testing.assert_allclose(stats["state"]["mean"], [0.1, 0.1])
    np.testing.assert_allclose(stats["action"]["min"], [0.0, 0.0])
    np.testing.assert_allclose(stats["action"]["max"], [0.02, 0.02])
    assert set(stats["action"]["quantiles"]) == {"0.01", "0.1", "0.5", "0.9", "0.99"}

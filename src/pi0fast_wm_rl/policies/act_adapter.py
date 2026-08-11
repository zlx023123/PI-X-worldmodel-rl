"""Delayed adapter for the source-verified LeRobot 0.6 ACT policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pi0fast_wm_rl.data.lerobot_adapter import check_lerobot_compatibility
from pi0fast_wm_rl.data.schema import Observation

from .base import BasePolicy


class ACTAdapter(BasePolicy):
    """Load ACTPolicy without making LeRobot a Mock-mode dependency."""

    def __init__(
        self, checkpoint: str | Path, device: str = "cuda", allow_download: bool = False
    ) -> None:
        check_lerobot_compatibility()
        source = str(checkpoint)
        if not allow_download and not Path(source).exists():
            raise FileNotFoundError(
                f"ACT checkpoint is not local: {source}. Download it explicitly or set "
                "allow_download=True."
            )
        try:
            import torch
            from lerobot.policies.act.modeling_act import ACTPolicy
            from lerobot.policies.factory import make_pre_post_processors
        except ImportError as exc:
            raise RuntimeError(
                "LeRobot ACT dependencies are incomplete; install '.[lerobot]'"
            ) from exc
        self._torch = torch
        self.device = device
        self.checkpoint = source
        self.policy = ACTPolicy.from_pretrained(source).to(device).eval()
        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            source,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

    def reset(self) -> None:
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def _batch(self, observation: Observation) -> dict[str, Any]:
        torch = self._torch
        batch: dict[str, Any] = {
            "observation.state": torch.as_tensor(observation.state, dtype=torch.float32).unsqueeze(
                0
            )
        }
        for name, image in observation.images.items():
            batch[f"observation.images.{name}"] = (
                torch.as_tensor(image).permute(2, 0, 1).float().div(255).unsqueeze(0)
            )
        return batch

    def predict_action_chunk(self, observation: Observation) -> np.ndarray:
        with self._torch.inference_mode():
            batch = self.preprocess(self._batch(observation))
            action = self.policy.predict_action_chunk(batch)
            action = self.postprocess(action)
        return action.squeeze(0).detach().cpu().numpy()

    def metadata(self) -> dict[str, Any]:
        return {
            "type": "act",
            "checkpoint": self.checkpoint,
            "device": self.device,
            "backend": "lerobot-0.6",
        }

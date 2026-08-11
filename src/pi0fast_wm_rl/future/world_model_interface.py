"""Future visual subgoal world-model protocol; no phase-one implementation."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class WorldModel(Protocol):
    """Predict a future visual subgoal from the current view and instruction."""

    def predict_subgoal(self, current_image: np.ndarray, instruction: str) -> np.ndarray: ...

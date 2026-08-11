"""Policy abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from pi0fast_wm_rl.data.schema import Observation


class BasePolicy(ABC):
    """Predict action chunks while hiding policy-library details."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def predict_action_chunk(self, observation: Observation) -> np.ndarray: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

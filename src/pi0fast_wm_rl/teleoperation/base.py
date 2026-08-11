"""Teleoperator abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseTeleoperator(ABC):
    """Produce action deltas for the configured robot action space."""

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_action(self, state: np.ndarray | None = None) -> np.ndarray: ...

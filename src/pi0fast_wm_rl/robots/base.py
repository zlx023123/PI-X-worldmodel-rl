"""Robot abstraction used by collection and deployment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseRobot(ABC):
    """Minimal lifecycle and action API for one robot arm."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...

    @abstractmethod
    def get_state(self) -> np.ndarray: ...

    @abstractmethod
    def send_action(self, action: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def emergency_stop(self) -> None: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

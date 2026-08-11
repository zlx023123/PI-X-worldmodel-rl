"""Camera abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseCamera(ABC):
    """Lifecycle and RGB read interface for cameras."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def read(self) -> np.ndarray: ...

    @abstractmethod
    def close(self) -> None: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

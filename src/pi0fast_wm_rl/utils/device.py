"""Optional PyTorch device inspection."""

from __future__ import annotations

from typing import Any


def torch_device_info() -> dict[str, Any]:
    """Return PyTorch/CUDA status without requiring PyTorch."""
    try:
        import torch
    except ImportError:
        return {"installed": False, "version": None, "cuda_available": False, "cuda_version": None}
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }

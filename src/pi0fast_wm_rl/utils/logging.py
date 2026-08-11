"""Logging setup shared by CLI commands."""

from __future__ import annotations

import logging


def configure_logging(verbose: bool = False) -> None:
    """Configure predictable console logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

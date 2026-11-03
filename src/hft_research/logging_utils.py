"""Structured logging setup for the research engine."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from hft_research.config import LoggingConfig

_CONFIGURED = False


def setup_logging(cfg: Optional[LoggingConfig] = None, *, force: bool = False) -> None:
    """Configure root logger once with a consistent format."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return
    cfg = cfg or LoggingConfig()
    level = getattr(logging, cfg.level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(cfg.format))
    root.addHandler(handler)
    root.setLevel(level)
    # Quiet noisy third-party loggers
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger under the hft_research namespace."""
    if not name.startswith("hft_research"):
        name = f"hft_research.{name}"
    return logging.getLogger(name)

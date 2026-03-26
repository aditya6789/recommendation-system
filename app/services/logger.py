"""Logging setup."""

import logging

from app.db.config import get_settings


def configure_logging() -> None:
    """Initialize global logging configuration."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

"""Logging configuration for TOPReward using loguru."""

import sys
from typing import Any

from loguru import logger


def _format_record(record: Any) -> str:
    """Custom format function that replaces dots with slashes in module names."""
    # Replace dots with slashes in the module name
    module_path = record["name"].replace(".", "/")

    # Build the format string with the modified module path
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        f"<cyan>{module_path}</cyan>:<cyan>{{line}}</cyan> - "
        "<level>{message}</level>\n"
    )


def setup_logging(level: str = "INFO", format_type: str = "default") -> None:
    """Configure loguru logging format.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format preset:
            - "minimal": Level and message only
            - "default": Time, level, and message
            - "detailed": Time, level, file location with slashes, and message
    """
    logger.remove()  # Remove default handler

    if format_type == "detailed":
        # Use custom format function for detailed logging
        logger.add(
            sys.stderr,
            format=_format_record,  # type: ignore[call-overload]
            level=level.upper(),
            colorize=True,
        )
    else:
        # Use simple format strings for other types
        formats = {
            "minimal": "<level>{level}</level> | {message}",
            "default": "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        }

        logger.add(
            sys.stderr,
            format=formats.get(format_type, formats["default"]),
            level=level.upper(),
            colorize=True,
        )

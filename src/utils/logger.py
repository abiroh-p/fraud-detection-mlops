"""
Centralized logging configuration for the fraud detection pipeline.

Why centralized logging?
- Consistent format across all services
- Single place to change log level or format
- Structured JSON logging for production log aggregators (Datadog, ELK Stack)
"""

import logging
import sys
from typing import Optional


def get_logger(
    name: str,
    level: int = logging.INFO,
    fmt: Optional[str] = None,
) -> logging.Logger:
    """
    Create and return a configured logger instance.

    In production, this would output JSON for log aggregation tools.
    For development, we use a human-readable format.

    Args:
        name: Logger name — use __name__ from the calling module.
              This creates a hierarchy: src.data.simulator > src.data
        level: Logging level. Default INFO. Use DEBUG for local dev.
        fmt: Optional custom format string.

    Returns:
        Configured Logger instance.

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Simulator started")
        2024-01-15 10:23:45 | INFO     | src.data.simulator | Simulator started
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if get_logger is called multiple times
    # for the same logger name (common mistake in large codebases)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Handler — where logs go (stdout in containers, file in some configs)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format — what each log line looks like
    if fmt is None:
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

    formatter = logging.Formatter(
        fmt=fmt,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent logs from propagating to the root logger
    # Without this, you get duplicate log lines
    logger.propagate = False

    return logger

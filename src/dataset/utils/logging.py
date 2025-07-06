"""
Logging configuration for dataset preprocessing.

This module provides centralized logging setup and configuration.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
    logger_name: str = "dataset",
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_str: Log message format string
        log_file: Path to log file (optional)
        logger_name: Name of the logger

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(format_str)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "dataset") -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = None

    @property
    def logger(self) -> logging.Logger:
        """Get logger instance."""
        if self._logger is None:
            self._logger = get_logger(self.__class__.__name__)
        return self._logger

    def log_info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def log_error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def log_debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

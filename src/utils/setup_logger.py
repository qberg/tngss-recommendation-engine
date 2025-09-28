"""
Centralized logging configuration for the project.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logger(
    name: Optional[str] = None,
    log_file: Optional[Union[str, Path]] = None,
    level: int = logging.INFO,
    console_output: bool = True,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> logging.Logger:
    """
    Setup a logger with consistent formatting across the project.

    Args:
        name: Logger name. If None, uses calling module's __name__
        log_file: Path to log file. If None, only console logging
        level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)
        log_format: Custom log format string
        date_format: Custom date format string

    Returns:
        Configured logger instance

    Example:
        # Basic usage
        logger = setup_logger(__name__, "logs/service.log")

        # Custom configuration
        logger = setup_logger(
            name="MyService",
            log_file="logs/custom.log",
            level=logging.DEBUG
        )
    """
    if log_format is None:
        log_format = "[%(asctime)s] - %(levelname)s - %(message)s"
    if date_format is None:
        date_format = "%Y-%m-%d %H:%M:%S"

    logger_name = name if name else __name__
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(log_format, datefmt=date_format)

    handlers = []

    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    for handler in handlers:
        logger.addHandler(handler)

    return logger


def setup_project_logging(
    log_dir: Union[str, Path] = "logs",
    level: int = logging.INFO,
    console_output: bool = True,
) -> None:
    """
    Setup basic logging configuration for the entire project.

    This configures the root logger with standard settings.
    Individual modules can still create their own loggers.

    Args:
        log_dir: Directory for log files
        level: Default logging level
        console_output: Whether to output to console
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=level,
        format="[%(asctime)s] - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            (
                logging.StreamHandler(sys.stdout)
                if console_output
                else logging.NullHandler()
            ),
            logging.FileHandler(log_path / "project.log"),
        ],
        force=True,
    )


def get_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """
    Quick logger setup with project defaults.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional specific log file

    Returns:
        Configured logger
    """
    return setup_logger(name, log_file)

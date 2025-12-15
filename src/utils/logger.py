"""Structured logging configuration for FashionMatch application.

This module provides colored console logging and rotating file logging
with performance tracking utilities.
"""

import logging
import os
import time
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

try:
    import colorlog
    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False


# Default log format
CONSOLE_FORMAT = "%(levelname)-8s %(name)s - %(message)s"
CONSOLE_FORMAT_COLOR = "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s"
FILE_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def _setup_console_handler(level: int) -> logging.Handler:
    """Create and configure console handler with optional color support.
    
    Args:
        level: Logging level
        
    Returns:
        Configured console handler
    """
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    if HAS_COLORLOG:
        # Use colored output
        formatter = colorlog.ColoredFormatter(
            CONSOLE_FORMAT_COLOR,
            datefmt=DATE_FORMAT,
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    else:
        # Fallback to standard formatter
        formatter = logging.Formatter(CONSOLE_FORMAT, datefmt=DATE_FORMAT)
    
    console_handler.setFormatter(formatter)
    return console_handler


def _setup_file_handler(level: int, log_dir: Optional[Path] = None) -> logging.Handler:
    """Create and configure rotating file handler.
    
    Args:
        level: Logging level
        log_dir: Directory for log files (default: logs/)
        
    Returns:
        Configured rotating file handler
    """
    if log_dir is None:
        # Default to logs/ in project root
        project_root = Path(__file__).parent.parent.parent
        log_dir = project_root / "logs"
    
    # Ensure log directory exists
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "fashionmatch.log"
    
    # Rotating file handler: max 10MB, keep 5 backup files
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(level)
    
    formatter = logging.Formatter(FILE_FORMAT, datefmt=DATE_FORMAT)
    file_handler.setFormatter(formatter)
    
    return file_handler


def get_logger(name: str, log_dir: Optional[Path] = None, level: Optional[str] = None) -> logging.Logger:
    """Get or create a logger with console and file handlers.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        log_dir: Directory for log files (default: logs/)
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
              If None, uses LOG_LEVEL environment variable, defaulting to INFO.
              Pass config.log_level from AppConfig for config-driven logging.
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only configure if logger has no handlers (avoid duplicate handlers)
    if not logger.handlers:
        # Determine log level: explicit parameter > env var > default
        if level is not None:
            log_level_str = level.upper()
        else:
            log_level_str = os.environ.get('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        
        logger.setLevel(log_level)
        
        # Add console handler
        console_handler = _setup_console_handler(log_level)
        logger.addHandler(console_handler)
        
        # Add file handler
        file_handler = _setup_file_handler(log_level, log_dir)
        logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        logger.propagate = False
    
    return logger


def log_performance(logger: logging.Logger, operation: str, duration: float) -> None:
    """Log performance metrics for an operation.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration: Duration in seconds
    """
    logger.info(f"Performance: {operation} completed in {duration:.3f}s")


def log_model_info(logger: logging.Logger, model_name: str, device: str, params: int) -> None:
    """Log model information.
    
    Args:
        logger: Logger instance
        model_name: Name of the model
        device: Device the model is loaded on
        params: Number of parameters (or -1 if unknown)
    """
    if params > 0:
        params_str = f"{params:,}"
        logger.info(f"Model loaded: {model_name} on {device} ({params_str} parameters)")
    else:
        logger.info(f"Model loaded: {model_name} on {device}")


@contextmanager
def log_execution_time(logger: logging.Logger, operation: str):
    """Context manager to automatically log operation execution time.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        
    Usage:
        with log_execution_time(logger, "image preprocessing"):
            # Your code here
            preprocess_images()
    """
    logger.debug(f"Starting: {operation}")
    start_time = time.time()
    
    try:
        yield
    finally:
        duration = time.time() - start_time
        logger.debug(f"Completed: {operation} in {duration:.3f}s")


def set_log_level(logger: logging.Logger, level: str) -> None:
    """Change the log level of a logger and all its handlers.
    
    Args:
        logger: Logger instance
        level: New log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    level_upper = level.upper()
    log_level = getattr(logging, level_upper, logging.INFO)
    
    logger.setLevel(log_level)
    for handler in logger.handlers:
        handler.setLevel(log_level)
    
    logger.info(f"Log level changed to {level_upper}")


def log_exception(logger: logging.Logger, operation: str, exception: Exception) -> None:
    """Log an exception with context.
    
    Args:
        logger: Logger instance
        operation: Name of the operation that failed
        exception: The exception that was raised
    """
    logger.error(f"Failed: {operation}", exc_info=True)

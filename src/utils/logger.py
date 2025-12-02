"""
Structured logging configuration using Loguru.

Provides consistent, colorful logging across the application with:
- Console output with colors
- File rotation and retention
- Structured context support
- Integration with Settings

Example:
    >>> from src.utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Starting application")
    >>> logger.debug("Debug message with context", user_id=123)
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from loguru import logger

if TYPE_CHECKING:
    from src.utils.config import Settings


# ============================================
# Logger Configuration
# ============================================


class LoggerManager:
    """
    Centralized logger management using Loguru.
    
    Configures logging based on application settings with support for:
    - Console output with colors and formatting
    - File logging with rotation and retention
    - Dynamic log level adjustment
    - Contextual logging
    """

    _configured: bool = False
    _default_format: str = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    _file_format: str = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - "
        "{message}"
    )

    @classmethod
    def configure(
        cls,
        settings: Optional["Settings"] = None,
        force: bool = False,
    ) -> None:
        """
        Configure the global logger based on settings.
        
        Args:
            settings: Application settings. Loads from config if not provided.
            force: If True, reconfigure even if already configured.
        """
        if cls._configured and not force:
            return

        # Remove default handlers
        logger.remove()

        # Load settings if not provided
        if settings is None:
            try:
                from src.utils.config import get_settings
                settings = get_settings()
            except Exception:
                # Use defaults if settings can't be loaded
                settings = None

        # Get log level
        log_level = "INFO"
        log_format = cls._default_format
        file_enabled = True
        file_path = "./logs/fashionmatch.log"
        rotation = "10 MB"
        retention = "1 week"

        if settings is not None:
            log_level = settings.logging.level
            file_enabled = settings.logging.file.enabled
            file_path = settings.logging.file.path
            rotation = settings.logging.file.rotation
            retention = settings.logging.file.retention

        # Add console handler with colors
        logger.add(
            sys.stderr,
            format=log_format,
            level=log_level,
            colorize=True,
            backtrace=True,
            diagnose=True,
        )

        # Add file handler if enabled
        if file_enabled:
            log_path = Path(file_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            logger.add(
                str(log_path),
                format=cls._file_format,
                level=log_level,
                rotation=rotation,
                retention=retention,
                compression="zip",
                encoding="utf-8",
                backtrace=True,
                diagnose=True,
            )

        cls._configured = True
        logger.debug("Logger configured successfully")

    @classmethod
    def get_logger(cls, name: str) -> "logger":
        """
        Get a contextualized logger for a module.
        
        Args:
            name: Module name (typically __name__).
            
        Returns:
            Loguru logger bound with module context.
        """
        # Ensure logger is configured
        if not cls._configured:
            cls.configure()

        # Bind module name to logger context
        return logger.bind(name=name)

    @classmethod
    def reset(cls) -> None:
        """Reset logger configuration."""
        logger.remove()
        cls._configured = False


# ============================================
# Convenience Functions
# ============================================


def get_logger(name: str) -> "logger":
    """
    Get a logger instance for the given module.
    
    This is the main entry point for getting loggers throughout the application.
    The logger is automatically configured on first use.
    
    Args:
        name: Module name, typically pass __name__.
        
    Returns:
        Configured Loguru logger with module context.
        
    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
        >>> logger.warning("Low memory", available_mb=128)
        >>> logger.error("Failed to process", error=str(e))
    """
    return LoggerManager.get_logger(name)


def configure_logging(
    settings: Optional["Settings"] = None,
    force: bool = False,
) -> None:
    """
    Explicitly configure logging.
    
    Usually not needed as logging auto-configures on first use.
    
    Args:
        settings: Application settings to use.
        force: If True, reconfigure even if already done.
    """
    LoggerManager.configure(settings, force)


# ============================================
# Logging Decorators
# ============================================


def log_function_call(func):
    """
    Decorator to log function entry and exit.
    
    Logs function name, arguments on entry, and return value on exit.
    Also logs any exceptions raised.
    
    Example:
        >>> @log_function_call
        ... def process_image(path: str) -> bool:
        ...     return True
    """
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        func_name = func.__qualname__

        # Log entry
        func_logger.debug(
            f"Entering {func_name}",
            args=args[:3] if args else None,  # Limit logged args
            kwargs=list(kwargs.keys()) if kwargs else None,
        )

        try:
            result = func(*args, **kwargs)
            func_logger.debug(f"Exiting {func_name}", success=True)
            return result
        except Exception as e:
            func_logger.exception(f"Exception in {func_name}: {e}")
            raise

    return wrapper


def log_async_function_call(func):
    """
    Decorator to log async function entry and exit.
    
    Same as log_function_call but for async functions.
    """
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        func_logger = get_logger(func.__module__)
        func_name = func.__qualname__

        func_logger.debug(
            f"Entering async {func_name}",
            args=args[:3] if args else None,
            kwargs=list(kwargs.keys()) if kwargs else None,
        )

        try:
            result = await func(*args, **kwargs)
            func_logger.debug(f"Exiting async {func_name}", success=True)
            return result
        except Exception as e:
            func_logger.exception(f"Exception in async {func_name}: {e}")
            raise

    return wrapper


# ============================================
# Context Managers
# ============================================


class LogContext:
    """
    Context manager for adding temporary context to logs.
    
    Example:
        >>> with LogContext(request_id="abc-123", user="john"):
        ...     logger.info("Processing request")  # Includes request_id and user
    """

    def __init__(self, **context: Any):
        """
        Initialize log context.
        
        Args:
            **context: Key-value pairs to add to log context.
        """
        self.context = context
        self._token = None

    def __enter__(self) -> "LogContext":
        """Enter context and bind values to logger."""
        self._token = logger.configure(extra=self.context)
        return self

    def __exit__(self, *args) -> None:
        """Exit context."""
        # Loguru handles context cleanup automatically
        pass


# ============================================
# Module Initialization
# ============================================

# Auto-configure on import (lazy, will configure on first use)
# This allows the logger to be used immediately after import

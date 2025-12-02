# Utils Package
"""
Shared utilities for the FashionMatch application.

Provides:
- Configuration management (Singleton Settings)
- Structured logging (Loguru-based)
- Custom exception hierarchy
- Image utilities
- Validation helpers
"""

from src.utils.config import Settings, get_project_root, get_settings
from src.utils.exceptions import (
    AppException,
    ConfigError,
    ConfigFileNotFoundError,
    ConfigurationError,
    DatabaseError,
    EncoderError,
    ScraperError,
    ValidationError,
)
from src.utils.logger import configure_logging, get_logger

__all__ = [
    # Configuration
    "Settings",
    "get_settings",
    "get_project_root",
    # Logging
    "get_logger",
    "configure_logging",
    # Exceptions
    "AppException",
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigurationError",
    "EncoderError",
    "ScraperError",
    "DatabaseError",
    "ValidationError",
]

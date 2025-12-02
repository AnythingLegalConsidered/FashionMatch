"""
Configuration management using Pydantic with Singleton pattern.

Provides type-safe configuration loading from YAML files with validation.
Implements a thread-safe Singleton pattern for global settings access.
"""

from __future__ import annotations

import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar, Dict, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ============================================
# Configuration Sub-Models
# ============================================


class AppSettings(BaseModel):
    """Application-level settings."""

    name: str = "FashionMatch"
    version: str = "0.1.0"
    debug: bool = False


class CLIPSettings(BaseModel):
    """CLIP model configuration."""

    model_name: str = "ViT-B/32"
    embedding_dim: int = 512

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """Validate embedding dimension is positive."""
        if v <= 0:
            raise ValueError("embedding_dim must be positive")
        return v


class DINOSettings(BaseModel):
    """DINOv2 model configuration."""

    model_name: str = "dinov2_vits14"
    embedding_dim: int = 384

    @field_validator("embedding_dim")
    @classmethod
    def validate_embedding_dim(cls, v: int) -> int:
        """Validate embedding dimension is positive."""
        if v <= 0:
            raise ValueError("embedding_dim must be positive")
        return v


class FusionWeightsConfig(BaseModel):
    """Fusion weights configuration."""

    clip: float = 0.5
    dino: float = 0.5

    @model_validator(mode="after")
    def validate_weights_sum(self) -> "FusionWeightsConfig":
        """Validate that weights sum to 1."""
        total = self.clip + self.dino
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Fusion weights must sum to 1, got {total}")
        return self


class FusionSettings(BaseModel):
    """Fusion strategy configuration."""

    strategy: Literal["weighted_average", "max", "learned"] = "weighted_average"
    weights: FusionWeightsConfig = Field(default_factory=FusionWeightsConfig)


class ModelSettings(BaseModel):
    """AI model configuration settings."""

    device: Literal["cuda", "cpu", "mps"] = "cuda"
    clip: CLIPSettings = Field(default_factory=CLIPSettings)
    dino: DINOSettings = Field(default_factory=DINOSettings)
    fusion: FusionSettings = Field(default_factory=FusionSettings)


class RequestDelaySettings(BaseModel):
    """Request delay settings for rate limiting."""

    min_seconds: float = 1.0
    max_seconds: float = 3.0

    @model_validator(mode="after")
    def validate_delay_range(self) -> "RequestDelaySettings":
        """Validate min <= max."""
        if self.min_seconds > self.max_seconds:
            raise ValueError("min_seconds must be <= max_seconds")
        return self


class ScraperSettings(BaseModel):
    """Web scraper configuration."""

    base_url: str = "https://www.vinted.fr"
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    max_pages_per_category: int = Field(default=10, ge=1, le=100)
    request_delay: RequestDelaySettings = Field(default_factory=RequestDelaySettings)
    timeout_seconds: int = Field(default=30, ge=1, le=300)
    headless: bool = True

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        return v.rstrip("/")


class ChromaSettings(BaseModel):
    """ChromaDB configuration."""

    persist_directory: str = "./data/chroma"
    collection_name: str = "clothing_items"


class DatabaseSettings(BaseModel):
    """Database configuration."""

    provider: Literal["chroma"] = "chroma"
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)


class ImageSettings(BaseModel):
    """Image processing configuration."""

    max_size: Tuple[int, int] = (224, 224)
    normalize: bool = True
    cache_dir: str = "./data/scraped/images"

    @field_validator("max_size", mode="before")
    @classmethod
    def parse_max_size(cls, v: Any) -> Tuple[int, int]:
        """Parse max_size from list to tuple."""
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("max_size must have exactly 2 elements")
            return tuple(v)  # type: ignore
        return v

    @field_validator("max_size")
    @classmethod
    def validate_max_size(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Validate image dimensions are positive."""
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("Image dimensions must be positive")
        return v


class UISettings(BaseModel):
    """UI configuration."""

    items_per_page: int = Field(default=20, ge=1, le=100)
    show_scores: bool = True
    theme: Literal["light", "dark"] = "light"


class LogFileSettings(BaseModel):
    """Log file configuration."""

    enabled: bool = True
    path: str = "./logs/fashionmatch.log"
    rotation: str = "10 MB"
    retention: str = "1 week"


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    file: LogFileSettings = Field(default_factory=LogFileSettings)


# ============================================
# Main Settings Class with Singleton Pattern
# ============================================


class Settings(BaseModel):
    """
    Main application settings with Singleton pattern.
    
    Thread-safe singleton that loads configuration from YAML.
    Access via Settings.get_instance() or get_settings().
    
    Example:
        >>> settings = Settings.get_instance()
        >>> print(settings.app.name)
        'FashionMatch'
    """

    # Singleton instance storage
    _instance: ClassVar[Optional["Settings"]] = None
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _initialized: ClassVar[bool] = False

    # Configuration sections
    app: AppSettings = Field(default_factory=AppSettings)
    models: ModelSettings = Field(default_factory=ModelSettings)
    scraper: ScraperSettings = Field(default_factory=ScraperSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    images: ImageSettings = Field(default_factory=ImageSettings)
    ui: UISettings = Field(default_factory=UISettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = {
        "validate_assignment": True,
        "extra": "ignore",  # Ignore unknown fields in YAML
    }

    @classmethod
    def get_instance(
        cls,
        config_path: Optional[Path | str] = None,
        force_reload: bool = False,
    ) -> "Settings":
        """
        Get the singleton Settings instance.
        
        Thread-safe singleton pattern implementation.
        
        Args:
            config_path: Optional path to config file. Uses default if not provided.
            force_reload: If True, reload configuration even if already loaded.
            
        Returns:
            The singleton Settings instance.
        """
        if cls._instance is None or force_reload:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None or force_reload:
                    cls._instance = cls._load_config(config_path)
                    cls._initialized = True
        return cls._instance

    @classmethod
    def _load_config(cls, config_path: Optional[Path | str] = None) -> "Settings":
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config file. Searches default locations if None.
            
        Returns:
            Settings instance with loaded configuration.
        """
        # Import here to avoid circular imports
        from src.utils.exceptions import ConfigFileNotFoundError, ConfigurationError

        # Determine config path
        if config_path is not None:
            path = Path(config_path)
            if not path.exists():
                raise ConfigFileNotFoundError(f"Config file not found: {path}")
        else:
            path = cls._find_config_file()

        # Load YAML if path found
        if path is not None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                return cls(**data)
            except yaml.YAMLError as e:
                raise ConfigurationError(f"Invalid YAML in config file: {e}")
            except Exception as e:
                raise ConfigurationError(f"Failed to load config: {e}")

        # Return defaults if no config file found
        return cls()

    @classmethod
    def _find_config_file(cls) -> Optional[Path]:
        """
        Find configuration file in default locations.
        
        Search order:
        1. Environment variable FASHIONMATCH_CONFIG_PATH
        2. ./config/config.yaml
        3. ./config.yaml
        4. Project root config/config.yaml
        
        Returns:
            Path to config file if found, None otherwise.
        """
        # Check environment variable
        env_path = os.environ.get("FASHIONMATCH_CONFIG_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists():
                return path

        # Check default locations
        possible_paths = [
            Path("config/config.yaml"),
            Path("config.yaml"),
            Path(__file__).parent.parent.parent / "config" / "config.yaml",
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance.
        
        Useful for testing or reloading configuration.
        """
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    @classmethod
    def is_initialized(cls) -> bool:
        """Check if settings have been initialized."""
        return cls._initialized

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return self.model_dump()

    def to_yaml(self, path: Path | str) -> None:
        """
        Save current settings to YAML file.
        
        Args:
            path: Path to save the YAML file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


# ============================================
# Convenience Functions
# ============================================


def get_settings(
    config_path: Optional[str | Path] = None,
    force_reload: bool = False,
) -> Settings:
    """
    Get application settings (convenience function).
    
    Wrapper around Settings.get_instance() for easier imports.
    
    Args:
        config_path: Optional path to config file.
        force_reload: If True, reload configuration.
        
    Returns:
        The singleton Settings instance.
        
    Example:
        >>> from src.utils.config import get_settings
        >>> settings = get_settings()
        >>> print(settings.models.device)
        'cuda'
    """
    return Settings.get_instance(config_path, force_reload)


@lru_cache(maxsize=1)
def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to the project root (where config/ is located).
    """
    current = Path(__file__).resolve()

    # Walk up until we find config/ directory
    for parent in current.parents:
        if (parent / "config").is_dir():
            return parent

    # Fallback to current working directory
    return Path.cwd()

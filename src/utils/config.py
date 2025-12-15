"""Configuration management system using Pydantic v2 and YAML.

This module provides type-safe configuration loading with validation
for the FashionMatch application.
"""

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class FusionWeights(BaseModel):
    """Fusion weights for combining CLIP and DINOv2 embeddings."""
    
    clip: float = Field(default=0.5, ge=0.0, le=1.0, description="CLIP weight for semantic understanding")
    dino: float = Field(default=0.5, ge=0.0, le=1.0, description="DINOv2 weight for structural analysis")
    
    @model_validator(mode='after')
    def validate_weights_sum(self) -> 'FusionWeights':
        """Ensure fusion weights sum to 1.0."""
        total = self.clip + self.dino
        if not abs(total - 1.0) < 1e-3:
            raise ValueError(f"Fusion weights must sum to 1.0, got {total}")
        return self


class ModelConfig(BaseModel):
    """Configuration for AI models (CLIP and DINOv2)."""
    
    clip_model: str = Field(default="ViT-B/32", description="CLIP model variant")
    dino_model: str = Field(default="dinov2_vits14", description="DINOv2 model variant")
    fusion_weights: FusionWeights = Field(default_factory=FusionWeights)
    device: str = Field(default="auto", description="Device selection: cuda, cpu, or auto")
    
    @field_validator('clip_model')
    @classmethod
    def validate_clip_model(cls, v: str) -> str:
        """Validate CLIP model name."""
        valid_models = ["ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"]
        if v not in valid_models:
            raise ValueError(f"Invalid CLIP model. Choose from: {valid_models}")
        return v
    
    @field_validator('dino_model')
    @classmethod
    def validate_dino_model(cls, v: str) -> str:
        """Validate DINOv2 model name."""
        valid_models = ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]
        if v not in valid_models:
            raise ValueError(f"Invalid DINOv2 model. Choose from: {valid_models}")
        return v
    
    @field_validator('device')
    @classmethod
    def validate_device(cls, v: str) -> str:
        """Validate and normalize device configuration."""
        valid_devices = ["cuda", "cpu", "auto"]
        v_lower = v.lower()
        if v_lower not in valid_devices:
            raise ValueError(f"Invalid device. Choose from: {valid_devices}")
        return v_lower


class ScraperConfig(BaseModel):
    """Configuration for web scraper."""
    
    base_url: str = Field(default="https://www.vinted.fr", description="Base URL for scraping")
    max_pages: int = Field(default=10, ge=1, description="Maximum number of pages to scrape")
    delay_range: list[float] = Field(default=[1.0, 3.0], description="Random delay range between requests")
    timeout: int = Field(default=30, ge=5, description="Page load timeout in seconds")
    headless: bool = Field(default=True, description="Run browser in headless mode")
    user_agents: list[str] = Field(default_factory=list, description="User agent strings for rotation")
    
    @field_validator('delay_range')
    @classmethod
    def validate_delay_range(cls, v: list[float]) -> list[float]:
        """Ensure delay range has exactly 2 values with min < max."""
        if len(v) != 2:
            raise ValueError("delay_range must contain exactly 2 values [min, max]")
        if v[0] >= v[1]:
            raise ValueError("delay_range min must be less than max")
        if v[0] < 0:
            raise ValueError("delay_range values must be non-negative")
        return v
    
    @field_validator('user_agents')
    @classmethod
    def validate_user_agents(cls, v: list[str]) -> list[str]:
        """Ensure at least one user agent is provided."""
        if not v:
            raise ValueError("At least one user agent is required")
        return v


class DatabaseConfig(BaseModel):
    """Configuration for ChromaDB vector database."""
    
    persist_directory: str = Field(default="./data/chroma", description="Directory for ChromaDB persistence")
    collection_name: str = Field(default="vinted_items", description="ChromaDB collection name")
    batch_size: int = Field(default=100, ge=1, description="Batch size for embedding insertion")
    distance_metric: str = Field(default="cosine", description="Distance metric for similarity search")
    
    @field_validator('distance_metric')
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate distance metric."""
        valid_metrics = ["cosine", "l2", "ip"]
        if v not in valid_metrics:
            raise ValueError(f"Invalid distance metric. Choose from: {valid_metrics}")
        return v


class AppConfig(BaseModel):
    """Root configuration model containing all sub-configurations."""
    
    models: ModelConfig = Field(default_factory=ModelConfig)
    scraper: ScraperConfig = Field(default_factory=ScraperConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    data_dir: str = Field(default="./data", description="Root data directory")
    references_dir: str = Field(default="./data/references", description="Directory for reference images")
    scraped_dir: str = Field(default="./data/scraped", description="Directory for scraped images")
    log_level: str = Field(default="INFO", description="Logging level")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"Invalid log level. Choose from: {valid_levels}")
        return v_upper


# Singleton pattern for configuration
_config: Optional[AppConfig] = None


def load_config(config_path: Optional[Path | str] = None) -> AppConfig:
    """Load and validate configuration from YAML file.
    
    This function expects to be run from a project layout where src/utils/config.py
    is three levels below the repository root, or a config path can be explicitly
    provided or set via the FASHIONMATCH_CONFIG environment variable.
    
    Args:
        config_path: Path to configuration file. Defaults to config/config.yaml
                    relative to project root, or FASHIONMATCH_CONFIG env var
        
    Returns:
        Validated AppConfig instance
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If configuration is invalid
    """
    if config_path is None:
        # Try environment variable first
        env_config_path = os.environ.get('FASHIONMATCH_CONFIG')
        if env_config_path:
            config_path = Path(env_config_path)
        else:
            # Default to config/config.yaml in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        example_path = config_path.parent / "config.example.yaml"
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Please copy {example_path} to {config_path} and customize it.\n"
            f"Alternatively, set the FASHIONMATCH_CONFIG environment variable to the config file path."
        )
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    
    # Validate and create configuration
    config = AppConfig.model_validate(config_dict)
    return config


def get_config(config_path: Optional[Path | str] = None, reload: bool = False) -> AppConfig:
    """Get configuration instance (singleton pattern).
    
    Args:
        config_path: Path to configuration file (only used on first call or if reload=True)
        reload: Force reload of configuration
        
    Returns:
        Cached or newly loaded AppConfig instance
    """
    global _config
    
    if _config is None or reload:
        _config = load_config(config_path)
    
    return _config


def reset_config() -> None:
    """Reset cached configuration (useful for testing)."""
    global _config
    _config = None

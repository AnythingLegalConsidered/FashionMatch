"""Unit tests for configuration loading and validation."""

import tempfile
from pathlib import Path

import pytest
import yaml

from src.utils.config import AppConfig, FusionWeights, ModelConfig, get_config


class TestFusionWeights:
    """Test fusion weight validation."""
    
    def test_valid_weights(self):
        """Test valid fusion weights sum to 1.0."""
        weights = FusionWeights(clip=0.6, dino=0.4)
        assert weights.clip == 0.6
        assert weights.dino == 0.4
    
    def test_weights_sum_validation(self):
        """Test fusion weights must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            FusionWeights(clip=0.5, dino=0.6)
    
    def test_weights_range_validation(self):
        """Test weights must be in [0, 1] range."""
        with pytest.raises(ValueError):
            FusionWeights(clip=1.5, dino=-0.5)
    
    def test_equal_weights(self):
        """Test equal 50/50 weights."""
        weights = FusionWeights(clip=0.5, dino=0.5)
        assert weights.clip == 0.5
        assert weights.dino == 0.5


class TestModelConfig:
    """Test model configuration."""
    
    def test_default_clip_model(self):
        """Test default CLIP model."""
        config = ModelConfig()
        assert "clip" in config.clip_model.lower()
    
    def test_custom_dino_model(self):
        """Test custom DINOv2 model."""
        config = ModelConfig(dino_model="dinov2_vitl14")
        assert config.dino_model == "dinov2_vitl14"
    
    def test_device_auto(self):
        """Test auto device selection."""
        config = ModelConfig(device="auto")
        assert config.device == "auto"
    
    def test_fusion_weights_included(self):
        """Test fusion weights are part of model config."""
        config = ModelConfig(
            fusion_weights=FusionWeights(clip=0.7, dino=0.3)
        )
        assert config.fusion_weights.clip == 0.7
        assert config.fusion_weights.dino == 0.3


class TestAppConfig:
    """Test application configuration."""
    
    def test_load_from_yaml(self, tmp_path):
        """Test loading configuration from YAML file."""
        config_data = {
            "models": {
                "clip_model": "openai/clip-vit-base-patch32",
                "dino_model": "dinov2_vits14",
                "fusion_weights": {"clip": 0.6, "dino": 0.4},
                "device": "cpu"
            },
            "database": {
                "persist_directory": "./data/chroma",
                "collection_name": "test_items",
                "batch_size": 32,
                "distance_metric": "cosine"
            },
            "data_dir": "./data",
            "references_dir": "./data/references",
            "scraped_dir": "./data/scraped",
            "log_level": "INFO"
        }
        
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        config = AppConfig.from_yaml(str(config_file))
        
        assert config.models.clip_model == "openai/clip-vit-base-patch32"
        assert config.models.dino_model == "dinov2_vits14"
        assert config.models.fusion_weights.clip == 0.6
        assert config.database.collection_name == "test_items"
        assert config.log_level == "INFO"
    
    def test_invalid_yaml(self, tmp_path):
        """Test error handling for invalid YAML."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content:")
        
        # Should raise yaml.YAMLError or similar parsing error
        with pytest.raises((yaml.YAMLError, Exception)):
            AppConfig.from_yaml(str(config_file))
    
    def test_missing_file(self):
        """Test error handling for missing config file."""
        with pytest.raises(FileNotFoundError):
            AppConfig.from_yaml("/nonexistent/config.yaml")
    
    def test_default_paths(self):
        """Test default directory paths."""
        config = AppConfig()
        assert config.data_dir == "./data"
        assert config.references_dir == "./data/references"
        assert config.scraped_dir == "./data/scraped"


class TestConfigSingleton:
    """Test configuration singleton pattern."""
    
    def test_get_config_returns_same_instance(self):
        """Test get_config returns cached instance."""
        # Call get_config twice without arguments
        config1 = get_config()
        config2 = get_config()
        
        # Should return the same instance (singleton pattern)
        assert config1 is config2
    
    def test_config_caching(self, tmp_path):
        """Test configuration is cached when loaded from file."""
        # Create a test config file
        config_data = {
            "models": {
                "clip_model": "openai/clip-vit-base-patch32",
                "dino_model": "dinov2_vits14",
                "fusion_weights": {"clip": 0.5, "dino": 0.5},
                "device": "cpu"
            },
            "database": {
                "persist_directory": "./test_data/chroma",
                "collection_name": "test",
                "batch_size": 16,
                "distance_metric": "cosine"
            },
            "data_dir": "./test_data",
            "references_dir": "./test_data/references",
            "scraped_dir": "./test_data/scraped",
            "log_level": "DEBUG"
        }
        
        config_file = tmp_path / "singleton_test.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config_data, f)
        
        # Load config from file twice
        config1 = get_config(config_file)
        config2 = get_config(config_file)
        
        # Should return the same cached instance
        assert config1 is config2
        assert config1.models.clip_model == "openai/clip-vit-base-patch32"

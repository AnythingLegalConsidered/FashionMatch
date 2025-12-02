"""
Pytest fixtures and configuration.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def sample_config():
    """Provide a sample configuration dictionary."""
    return {
        "app": {
            "name": "FashionMatch",
            "version": "0.1.0",
            "debug": True,
        },
        "models": {
            "device": "cpu",
            "clip": {
                "model_name": "ViT-B/32",
                "embedding_dim": 512,
            },
            "dino": {
                "model_name": "dinov2_vits14",
                "embedding_dim": 384,
            },
        },
    }


@pytest.fixture
def sample_clothing_item():
    """Provide a sample ClothingItem data."""
    return {
        "id": "test-123",
        "title": "Vintage Denim Jacket",
        "price": 25.0,
        "currency": "EUR",
        "brand": "Levi's",
        "size": "M",
        "condition": "Good",
        "category": "Jackets",
    }


@pytest.fixture
def temp_image(tmp_path):
    """Create a temporary test image."""
    from PIL import Image
    
    # Create a simple test image
    img = Image.new("RGB", (224, 224), color="red")
    img_path = tmp_path / "test_image.jpg"
    img.save(img_path)
    
    return img_path

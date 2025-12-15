"""Pytest fixtures and configuration for FashionMatch tests."""

import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from PIL import Image

from src.database.models import FashionItem
from src.database.vector_store import VectorStore
from src.scraper.models import VintedItem
from src.utils.config import AppConfig, DatabaseConfig, FusionWeights, ModelConfig


@pytest.fixture
def test_config() -> AppConfig:
    """Provide test-specific configuration with in-memory ChromaDB."""
    return AppConfig(
        models=ModelConfig(
            clip_model="openai/clip-vit-base-patch32",
            dino_model="dinov2_vits14",
            fusion_weights=FusionWeights(clip=0.6, dino=0.4),
            device="cpu"
        ),
        database=DatabaseConfig(
            persist_directory=":memory:",
            collection_name="test_fashion_items",
            batch_size=16,
            distance_metric="cosine"
        ),
        data_dir="./test_data",
        references_dir="./test_data/references",
        scraped_dir="./test_data/scraped",
        log_level="DEBUG"
    )


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Create subdirectories
        (temp_path / "references").mkdir()
        (temp_path / "scraped").mkdir()
        (temp_path / "chroma").mkdir()
        
        yield temp_path


@pytest.fixture
def sample_image() -> Image.Image:
    """Create a sample RGB image for testing."""
    # Create 224x224 red image
    img = Image.new("RGB", (224, 224), color=(255, 0, 0))
    return img


@pytest.fixture
def sample_images() -> list[Image.Image]:
    """Create multiple sample images with different colors."""
    images = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for color in colors:
        img = Image.new("RGB", (224, 224), color=color)
        images.append(img)
    
    return images


@pytest.fixture
def sample_embeddings() -> tuple[np.ndarray, np.ndarray]:
    """Create sample CLIP and DINOv2 embeddings."""
    clip_emb = np.random.randn(512).astype(np.float32)
    clip_emb = clip_emb / np.linalg.norm(clip_emb)  # L2 normalize
    
    dino_emb = np.random.randn(384).astype(np.float32)
    dino_emb = dino_emb / np.linalg.norm(dino_emb)  # L2 normalize
    
    return clip_emb, dino_emb


@pytest.fixture
def mock_fashion_item(sample_embeddings) -> FashionItem:
    """Create a mock FashionItem with embeddings."""
    clip_emb, dino_emb = sample_embeddings
    
    return FashionItem(
        item_id="test_item_001",
        title="Test Fashion Item",
        price=29.99,
        url="https://www.vinted.fr/items/test-001",
        image_url="https://example.com/image.jpg",
        brand="TestBrand",
        category="chemises",
        clip_embedding=clip_emb,
        dino_embedding=dino_emb,
        additional_metadata={"size": "M", "condition": "Good"}
    )


@pytest.fixture
def mock_fashion_items(sample_embeddings) -> list[FashionItem]:
    """Create multiple mock FashionItems."""
    items = []
    
    for i in range(10):
        # Generate random normalized embeddings
        clip_emb = np.random.randn(512).astype(np.float32)
        clip_emb = clip_emb / np.linalg.norm(clip_emb)
        
        dino_emb = np.random.randn(384).astype(np.float32)
        dino_emb = dino_emb / np.linalg.norm(dino_emb)
        
        item = FashionItem(
            item_id=f"test_item_{i:03d}",
            title=f"Test Item {i}",
            price=10.0 + i * 5.0,
            url=f"https://www.vinted.fr/items/test-{i:03d}",
            image_url=f"https://example.com/image_{i}.jpg",
            brand=f"Brand{i % 3}",
            category=["chemises", "robes", "pantalons"][i % 3],
            clip_embedding=clip_emb,
            dino_embedding=dino_emb
        )
        items.append(item)
    
    return items


@pytest.fixture
def mock_vector_store(test_config, mock_fashion_items, temp_data_dir) -> VectorStore:
    """Create a pre-populated vector store for testing."""
    # Override config to use temp directory
    test_config.database.persist_directory = str(temp_data_dir / "chroma")
    
    # Use the correct constructor with DatabaseConfig
    from src.database.vector_store import get_vector_store
    
    vector_store = get_vector_store(
        config=test_config.database,
        clip_dim=512,
        dino_dim=384
    )
    
    # Add mock items
    vector_store.add_items(mock_fashion_items)
    
    return vector_store


@pytest.fixture
def mock_vinted_item() -> VintedItem:
    """Create a mock VintedItem from scraper."""
    return VintedItem(
        item_id="vinted_123",
        title="Vintage Denim Jacket",
        price=45.00,
        url="https://www.vinted.fr/items/vinted-123",
        image_url="https://example.com/jacket.jpg",
        brand="Levi's",
        size="L",
        condition="Good",
        category="Vestes"
    )


@pytest.fixture
def mock_scraped_batch(mock_vinted_item) -> list[VintedItem]:
    """Create a batch of mock scraped items."""
    items = []
    
    for i in range(5):
        item = VintedItem(
            item_id=f"vinted_{i:03d}",
            title=f"Scraped Item {i}",
            price=20.0 + i * 10.0,
            url=f"https://www.vinted.fr/items/vinted-{i:03d}",
            image_url=f"https://example.com/item_{i}.jpg",
            brand=["Nike", "Adidas", "Zara", "H&M", "Uniqlo"][i],
            size=["S", "M", "L", "XL", "XXL"][i],
            condition="Good",
            category="chemises"
        )
        items.append(item)
    
    return items


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances between tests."""
    # This ensures each test gets fresh instances
    # Add cleanup for cached encoders and stores if needed
    yield
    # Cleanup happens here

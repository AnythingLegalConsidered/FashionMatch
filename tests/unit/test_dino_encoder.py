"""Unit tests for DINOv2 encoder."""

import numpy as np
import pytest

from src.core.encoders.dino_encoder import DINOEncoder, get_dino_encoder


class TestDINOEncoder:
    """Test DINOv2 encoder functionality."""
    
    @pytest.fixture
    def encoder(self):
        """Create DINOv2 encoder instance."""
        return DINOEncoder(
            model_name="dinov2_vits14",
            device="cpu"
        )
    
    def test_encoder_initialization(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder.model is not None
        assert encoder.device == "cpu"
        assert encoder.embedding_dim == 384  # vits14 dimension
    
    def test_single_image_encoding(self, encoder, sample_image):
        """Test encoding single image."""
        embedding = encoder.encode([sample_image])
        
        assert embedding.shape == (1, 384)
        assert embedding.dtype == np.float32
        # Check L2 normalization
        assert np.allclose(np.linalg.norm(embedding[0]), 1.0, atol=1e-5)
    
    def test_batch_encoding(self, encoder, sample_images):
        """Test encoding batch of images."""
        embeddings = encoder.encode(sample_images)
        
        assert embeddings.shape == (len(sample_images), 384)
        assert embeddings.dtype == np.float32
        # Check all embeddings are L2 normalized
        for emb in embeddings:
            assert np.allclose(np.linalg.norm(emb), 1.0, atol=1e-5)
    
    def test_empty_batch(self, encoder):
        """Test error handling for empty batch."""
        with pytest.raises(ValueError):
            encoder.encode([])
    
    def test_encoding_consistency(self, encoder, sample_image):
        """Test same image produces same embedding."""
        emb1 = encoder.encode([sample_image])
        emb2 = encoder.encode([sample_image])
        
        assert np.allclose(emb1, emb2, atol=1e-6)
    
    def test_different_images_different_embeddings(self, encoder, sample_images):
        """Test different images produce different embeddings."""
        embeddings = encoder.encode(sample_images[:2])
        
        # Embeddings should not be identical
        assert not np.allclose(embeddings[0], embeddings[1])


class TestDINOEncoderSingleton:
    """Test DINOv2 encoder singleton pattern."""
    
    def test_get_dino_encoder_returns_instance(self):
        """Test get_dino_encoder returns encoder."""
        encoder = get_dino_encoder(
            model_name="dinov2_vits14",
            device="cpu"
        )
        
        assert isinstance(encoder, DINOEncoder)
    
    def test_singleton_caching(self):
        """Test encoder is cached."""
        encoder1 = get_dino_encoder("dinov2_vits14", "cpu")
        encoder2 = get_dino_encoder("dinov2_vits14", "cpu")
        
        # Should return same instance
        assert encoder1 is encoder2


class TestDINOModelVariants:
    """Test different DINOv2 model variants."""
    
    @pytest.mark.parametrize("model_name,expected_dim", [
        ("dinov2_vits14", 384),
        ("dinov2_vitb14", 768),
        ("dinov2_vitl14", 1024),
        ("dinov2_vitg14", 1536),
    ])
    def test_model_dimensions(self, model_name, expected_dim, sample_image):
        """Test different DINOv2 models have correct dimensions."""
        encoder = DINOEncoder(model_name=model_name, device="cpu")
        embedding = encoder.encode([sample_image])
        
        assert embedding.shape[1] == expected_dim

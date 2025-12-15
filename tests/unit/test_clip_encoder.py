"""Unit tests for CLIP encoder."""

import numpy as np
import pytest

from src.core.encoders.clip_encoder import CLIPEncoder, get_clip_encoder


class TestCLIPEncoder:
    """Test CLIP encoder functionality."""
    
    @pytest.fixture
    def encoder(self):
        """Create CLIP encoder instance."""
        return CLIPEncoder(
            model_name="openai/clip-vit-base-patch32",
            device="cpu"
        )
    
    def test_encoder_initialization(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder.model is not None
        assert encoder.processor is not None
        assert encoder.device == "cpu"
        assert encoder.embedding_dim == 512  # ViT-B/32 dimension
    
    def test_single_image_encoding(self, encoder, sample_image):
        """Test encoding single image."""
        embedding = encoder.encode([sample_image])
        
        assert embedding.shape == (1, 512)
        assert embedding.dtype == np.float32
        # Check L2 normalization
        assert np.allclose(np.linalg.norm(embedding[0]), 1.0, atol=1e-5)
    
    def test_batch_encoding(self, encoder, sample_images):
        """Test encoding batch of images."""
        embeddings = encoder.encode(sample_images)
        
        assert embeddings.shape == (len(sample_images), 512)
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
    
    def test_embedding_dimension_consistency(self, encoder, sample_images):
        """Test all embeddings have consistent dimensions."""
        for img in sample_images:
            emb = encoder.encode([img])
            assert emb.shape[1] == encoder.embedding_dim


class TestCLIPEncoderSingleton:
    """Test CLIP encoder singleton pattern."""
    
    def test_get_clip_encoder_returns_instance(self):
        """Test get_clip_encoder returns encoder."""
        encoder = get_clip_encoder(
            model_name="openai/clip-vit-base-patch32",
            device="cpu"
        )
        
        assert isinstance(encoder, CLIPEncoder)
    
    def test_singleton_caching(self):
        """Test encoder is cached."""
        encoder1 = get_clip_encoder("openai/clip-vit-base-patch32", "cpu")
        encoder2 = get_clip_encoder("openai/clip-vit-base-patch32", "cpu")
        
        # Should return same instance
        assert encoder1 is encoder2
    
    def test_different_models_different_instances(self):
        """Test different models create different instances."""
        encoder1 = get_clip_encoder("openai/clip-vit-base-patch32", "cpu")
        encoder2 = get_clip_encoder("openai/clip-vit-base-patch16", "cpu")
        
        # Should be different instances
        assert encoder1 is not encoder2


class TestCLIPModelVariants:
    """Test different CLIP model variants."""
    
    @pytest.mark.parametrize("model_name,expected_dim", [
        ("openai/clip-vit-base-patch32", 512),
        ("openai/clip-vit-base-patch16", 512),
        ("openai/clip-vit-large-patch14", 768),
    ])
    def test_model_dimensions(self, model_name, expected_dim, sample_image):
        """Test different CLIP models have correct dimensions."""
        encoder = CLIPEncoder(model_name=model_name, device="cpu")
        embedding = encoder.encode([sample_image])
        
        assert embedding.shape[1] == expected_dim

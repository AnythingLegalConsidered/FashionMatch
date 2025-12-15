"""Unit tests for image utility functions."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.utils.image_utils import load_image, preprocess_for_clip, preprocess_for_dino


class TestLoadImage:
    """Test image loading functionality."""
    
    def test_load_jpeg_image(self, tmp_path):
        """Test loading JPEG image."""
        img = Image.new("RGB", (224, 224), color=(255, 0, 0))
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        loaded = load_image(str(img_path))
        
        assert loaded.mode == "RGB"
        assert loaded.size == (224, 224)
    
    def test_load_png_image(self, tmp_path):
        """Test loading PNG image."""
        img = Image.new("RGB", (300, 300), color=(0, 255, 0))
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        loaded = load_image(str(img_path))
        
        assert loaded.mode == "RGB"
        assert loaded.size == (300, 300)
    
    def test_convert_rgba_to_rgb(self, tmp_path):
        """Test RGBA images are converted to RGB."""
        img = Image.new("RGBA", (200, 200), color=(0, 0, 255, 128))
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        loaded = load_image(str(img_path))
        
        assert loaded.mode == "RGB"
    
    def test_convert_grayscale_to_rgb(self, tmp_path):
        """Test grayscale images are converted to RGB."""
        img = Image.new("L", (150, 150), color=128)
        img_path = tmp_path / "test.jpg"
        img.save(img_path)
        
        loaded = load_image(str(img_path))
        
        assert loaded.mode == "RGB"
    
    def test_load_nonexistent_file(self):
        """Test error handling for missing file."""
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.jpg")
    
    def test_load_corrupted_image(self, tmp_path):
        """Test error handling for corrupted image."""
        corrupt_file = tmp_path / "corrupt.jpg"
        corrupt_file.write_bytes(b"not an image")
        
        with pytest.raises(Exception):
            load_image(str(corrupt_file))


class TestPreprocessForCLIP:
    """Test CLIP preprocessing."""
    
    def test_preprocess_single_image(self, sample_image):
        """Test preprocessing single image for CLIP."""
        tensor = preprocess_for_clip(sample_image)
        
        assert tensor.shape == (3, 224, 224)  # C, H, W
        assert tensor.dtype == np.float32
        assert -3.0 <= tensor.min() <= tensor.max() <= 3.0  # Normalized range
    
    def test_preprocess_batch(self, sample_images):
        """Test batch preprocessing for CLIP."""
        tensors = [preprocess_for_clip(img) for img in sample_images]
        batch = np.stack(tensors)
        
        assert batch.shape == (len(sample_images), 3, 224, 224)
        assert batch.dtype == np.float32
    
    def test_resize_large_image(self):
        """Test resizing large image for CLIP."""
        large_img = Image.new("RGB", (1024, 1024), color=(128, 128, 128))
        tensor = preprocess_for_clip(large_img)
        
        assert tensor.shape == (3, 224, 224)
    
    def test_resize_small_image(self):
        """Test resizing small image for CLIP."""
        small_img = Image.new("RGB", (50, 50), color=(64, 64, 64))
        tensor = preprocess_for_clip(small_img)
        
        assert tensor.shape == (3, 224, 224)


class TestPreprocessForDINO:
    """Test DINOv2 preprocessing."""
    
    def test_preprocess_single_image(self, sample_image):
        """Test preprocessing single image for DINOv2."""
        tensor = preprocess_for_dino(sample_image)
        
        assert tensor.shape == (3, 224, 224)  # C, H, W
        assert tensor.dtype == np.float32
        assert -3.0 <= tensor.min() <= tensor.max() <= 3.0
    
    def test_preprocess_batch(self, sample_images):
        """Test batch preprocessing for DINOv2."""
        tensors = [preprocess_for_dino(img) for img in sample_images]
        batch = np.stack(tensors)
        
        assert batch.shape == (len(sample_images), 3, 224, 224)
        assert batch.dtype == np.float32
    
    def test_different_normalization_than_clip(self, sample_image):
        """Test DINOv2 uses different normalization than CLIP."""
        clip_tensor = preprocess_for_clip(sample_image)
        dino_tensor = preprocess_for_dino(sample_image)
        
        # Should be different due to different normalization stats
        assert not np.allclose(clip_tensor, dino_tensor)


class TestImageUtils:
    """Test additional image utility functions."""
    
    def test_mixed_size_batch_preprocessing(self):
        """Test preprocessing batch with mixed sizes."""
        img1 = Image.new("RGB", (100, 100), color=(255, 0, 0))
        img2 = Image.new("RGB", (500, 500), color=(0, 255, 0))
        img3 = Image.new("RGB", (224, 224), color=(0, 0, 255))
        
        tensors = [preprocess_for_clip(img) for img in [img1, img2, img3]]
        
        # All should be resized to same dimensions
        assert all(t.shape == (3, 224, 224) for t in tensors)
    
    def test_aspect_ratio_handling(self):
        """Test handling of non-square images."""
        wide_img = Image.new("RGB", (400, 200), color=(128, 128, 128))
        tall_img = Image.new("RGB", (200, 400), color=(128, 128, 128))
        
        wide_tensor = preprocess_for_clip(wide_img)
        tall_tensor = preprocess_for_clip(tall_img)
        
        # Both should be resized to square
        assert wide_tensor.shape == (3, 224, 224)
        assert tall_tensor.shape == (3, 224, 224)

"""DINOv2 encoder for structural image understanding.

This module provides a wrapper around Meta's DINOv2 self-supervised vision model
for encoding images into structural embeddings.
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image

from src.utils import detect_device, get_logger, log_execution_time, log_model_info
from src.utils.config import ModelConfig
from src.utils.image_utils import batch_preprocess, preprocess_for_dino

logger = get_logger(__name__)

# Embedding dimensions for each DINOv2 variant
DINO_EMBEDDING_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2Encoder:
    """Encoder for extracting structural embeddings from images using DINOv2."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize DINOv2 encoder.
        
        Args:
            model_name: DINOv2 model variant (e.g., "dinov2_vits14")
            device: Device to run model on ("auto", "cuda", or "cpu")
            
        Raises:
            ValueError: If model_name is not supported
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.device = detect_device(device)
        
        # Validate model name
        if model_name not in DINO_EMBEDDING_DIMS:
            raise ValueError(
                f"Unsupported DINOv2 model: {model_name}. "
                f"Choose from: {list(DINO_EMBEDDING_DIMS.keys())}"
            )
        
        try:
            logger.info(f"Loading DINOv2 model: {model_name}")
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model.to(self.device)
            self.model.eval()
            
            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters())
            log_model_info(logger, model_name, self.device, num_params)
            
        except Exception as e:
            logger.error(f"Failed to load DINOv2 model {model_name}")
            raise RuntimeError(f"DINOv2 model loading failed: {str(e)}") from e
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension for this model.
        
        Returns:
            Embedding dimension (384/768/1024/1536 depending on variant)
        """
        return DINO_EMBEDDING_DIMS[self.model_name]
    
    def encode(self, images: Image.Image | list[Image.Image]) -> np.ndarray:
        """Encode images into structural embeddings.
        
        Args:
            images: Single PIL Image or list of PIL Images
            
        Returns:
            L2-normalized embeddings as numpy array with shape (N, embedding_dim)
            
        Raises:
            ValueError: If images have unsupported format
        """
        # Handle single image input
        if isinstance(images, Image.Image):
            images = [images]
        
        if not images:
            raise ValueError("Cannot encode empty image list")
        
        # Validate image types
        for i, img in enumerate(images):
            if not isinstance(img, Image.Image):
                raise ValueError(f"Image at index {i} is not a PIL Image: {type(img)}")
        
        try:
            with log_execution_time(logger, f"DINOv2 encoding {len(images)} images"):
                # Preprocess images
                batch_tensor = batch_preprocess(images, preprocess_for_dino)
                batch_tensor = batch_tensor.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    image_features = self.model(batch_tensor)
                    
                    # L2-normalize embeddings for consistency with CLIP
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Convert to numpy
                embeddings = image_features.cpu().numpy().astype(np.float32)
                
                return embeddings
                
        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA out of memory while encoding {len(images)} images")
            raise RuntimeError(
                "GPU out of memory. Try reducing batch size or using CPU."
            )
        except Exception as e:
            logger.error(f"Failed to encode images with DINOv2")
            raise RuntimeError(f"DINOv2 encoding failed: {str(e)}") from e


# Singleton pattern for encoder caching
_dino_encoder_cache: dict[tuple[str, str], DINOv2Encoder] = {}


def get_dino_encoder(config: ModelConfig) -> DINOv2Encoder:
    """Get or create a DINOv2 encoder instance (singleton pattern).
    
    Args:
        config: Model configuration containing DINOv2 model name and device
        
    Returns:
        Cached or newly created DINOv2Encoder instance
    """
    cache_key = (config.dino_model, config.device)
    
    if cache_key not in _dino_encoder_cache:
        logger.debug(f"Creating new DINOv2 encoder for {cache_key}")
        _dino_encoder_cache[cache_key] = DINOv2Encoder(
            model_name=config.dino_model,
            device=config.device
        )
    else:
        logger.debug(f"Reusing cached DINOv2 encoder for {cache_key}")
    
    return _dino_encoder_cache[cache_key]

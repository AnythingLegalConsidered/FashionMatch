"""CLIP encoder for semantic image understanding.

This module provides a wrapper around OpenAI's CLIP model via Hugging Face
Transformers for encoding images into semantic embeddings.
"""

from typing import Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils import detect_device, get_logger, log_execution_time, log_model_info
from src.utils.config import ModelConfig
from src.utils.image_utils import batch_preprocess, preprocess_for_clip

logger = get_logger(__name__)

# Mapping from config names to Hugging Face model IDs
CLIP_MODEL_MAPPING = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
}


class CLIPEncoder:
    """Encoder for extracting semantic embeddings from images using CLIP."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        """Initialize CLIP encoder.
        
        Args:
            model_name: CLIP model variant (e.g., "ViT-B/32")
            device: Device to run model on ("auto", "cuda", or "cpu")
            
        Raises:
            ValueError: If model_name is not supported
            RuntimeError: If model loading fails
        """
        self.model_name = model_name
        self.device = detect_device(device)
        
        # Map config name to Hugging Face model ID
        if model_name not in CLIP_MODEL_MAPPING:
            raise ValueError(
                f"Unsupported CLIP model: {model_name}. "
                f"Choose from: {list(CLIP_MODEL_MAPPING.keys())}"
            )
        
        hf_model_id = CLIP_MODEL_MAPPING[model_name]
        
        try:
            logger.info(f"Loading CLIP model: {hf_model_id}")
            self.processor = CLIPProcessor.from_pretrained(hf_model_id)
            self.model = CLIPModel.from_pretrained(hf_model_id)
            self.model.to(self.device)
            self.model.eval()
            
            # Log model info
            num_params = sum(p.numel() for p in self.model.parameters())
            log_model_info(logger, hf_model_id, self.device, num_params)
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model {hf_model_id}")
            raise RuntimeError(f"CLIP model loading failed: {str(e)}") from e
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension for this model.
        
        Returns:
            Embedding dimension (512 for ViT-B, 768 for ViT-L)
        """
        return self.model.config.projection_dim
    
    def encode(self, images: Image.Image | list[Image.Image]) -> np.ndarray:
        """Encode images into semantic embeddings.
        
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
            with log_execution_time(logger, f"CLIP encoding {len(images)} images"):
                # Determine target size based on model variant
                target_size = 336 if self.model_name == "ViT-L/14@336px" else 224
                
                # Preprocess images with correct size
                preprocessor = lambda img: preprocess_for_clip(img, size=target_size)
                batch_tensor = batch_preprocess(images, preprocessor)
                batch_tensor = batch_tensor.to(self.device)
                
                # Extract features
                with torch.no_grad():
                    image_features = self.model.get_image_features(pixel_values=batch_tensor)
                    
                    # L2-normalize embeddings
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
            logger.error(f"Failed to encode images with CLIP")
            raise RuntimeError(f"CLIP encoding failed: {str(e)}") from e


# Singleton pattern for encoder caching
_clip_encoder_cache: dict[tuple[str, str], CLIPEncoder] = {}


def get_clip_encoder(config: ModelConfig) -> CLIPEncoder:
    """Get or create a CLIP encoder instance (singleton pattern).
    
    Args:
        config: Model configuration containing CLIP model name and device
        
    Returns:
        Cached or newly created CLIPEncoder instance
    """
    cache_key = (config.clip_model, config.device)
    
    if cache_key not in _clip_encoder_cache:
        logger.debug(f"Creating new CLIP encoder for {cache_key}")
        _clip_encoder_cache[cache_key] = CLIPEncoder(
            model_name=config.clip_model,
            device=config.device
        )
    else:
        logger.debug(f"Reusing cached CLIP encoder for {cache_key}")
    
    return _clip_encoder_cache[cache_key]

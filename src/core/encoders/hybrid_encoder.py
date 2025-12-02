"""
Hybrid encoder orchestrating CLIP and DINOv2.

Combines semantic (CLIP) and structural (DINOv2) embeddings
for comprehensive clothing representation.

The HybridEncoder doesn't load models itself - it orchestrates
existing CLIPEncoder and DINOEncoder instances.

Example:
    >>> from src.core.encoders import CLIPEncoder, DINOEncoder, HybridEncoder
    >>> 
    >>> clip = CLIPEncoder()
    >>> dino = DINOEncoder()
    >>> hybrid = HybridEncoder(clip, dino)
    >>> hybrid.load_models()
    >>> 
    >>> result = hybrid.encode_all("image.jpg")
    >>> print(result.clip_embedding.shape)  # (512,)
    >>> print(result.dino_embedding.shape)  # (384,)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from src.core.encoders.clip_encoder import CLIPEncoder
from src.core.encoders.dino_encoder import DINOEncoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class HybridEmbedding:
    """
    Container for hybrid embeddings from CLIP and DINO.
    
    Attributes:
        clip_embedding: Semantic embedding from CLIP (512d or 768d).
        dino_embedding: Structural embedding from DINOv2 (384d, 768d, etc.).
        
    Example:
        >>> result = hybrid.encode_all("image.jpg")
        >>> clip_emb = result.clip_embedding
        >>> dino_emb = result.dino_embedding
        >>> combined = result.concatenate()
    """
    
    clip_embedding: np.ndarray
    dino_embedding: np.ndarray
    
    @property
    def clip_dim(self) -> int:
        """Return CLIP embedding dimension."""
        return self.clip_embedding.shape[0]
    
    @property
    def dino_dim(self) -> int:
        """Return DINO embedding dimension."""
        return self.dino_embedding.shape[0]
    
    @property
    def total_dim(self) -> int:
        """Return total dimension when concatenated."""
        return self.clip_dim + self.dino_dim
    
    def concatenate(self) -> np.ndarray:
        """
        Concatenate CLIP and DINO embeddings.
        
        Returns:
            Numpy array of shape (clip_dim + dino_dim,).
        """
        return np.concatenate([self.clip_embedding, self.dino_embedding])
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary format.
        
        Returns:
            Dictionary with 'clip' and 'dino' keys.
        """
        return {
            "clip": self.clip_embedding,
            "dino": self.dino_embedding,
        }


class HybridEncoder:
    """
    Orchestrates CLIP and DINOv2 encoders for hybrid embeddings.
    
    This class doesn't load models itself - it takes existing encoder
    instances and coordinates their use for comprehensive image encoding.
    
    The hybrid approach combines:
    - CLIP: Semantic understanding (what objects are, style, categories)
    - DINO: Structural features (textures, patterns, visual details)
    
    Attributes:
        clip_encoder: CLIPEncoder instance.
        dino_encoder: DINOEncoder instance.
        
    Example:
        >>> clip = CLIPEncoder()
        >>> dino = DINOEncoder()
        >>> hybrid = HybridEncoder(clip, dino)
        >>> 
        >>> # Load both models
        >>> hybrid.load_models()
        >>> 
        >>> # Encode image
        >>> result = hybrid.encode_all("image.jpg")
        >>> print(f"CLIP: {result.clip_embedding.shape}")
        >>> print(f"DINO: {result.dino_embedding.shape}")
    """
    
    def __init__(
        self,
        clip_encoder: Optional[CLIPEncoder] = None,
        dino_encoder: Optional[DINOEncoder] = None,
    ):
        """
        Initialize the hybrid encoder.
        
        Args:
            clip_encoder: CLIPEncoder instance. Creates new one if None.
            dino_encoder: DINOEncoder instance. Creates new one if None.
            
        Example:
            >>> # Use default encoders
            >>> hybrid = HybridEncoder()
            >>> 
            >>> # Use custom encoders
            >>> hybrid = HybridEncoder(
            ...     clip_encoder=CLIPEncoder(model_name="ViT-L/14"),
            ...     dino_encoder=DINOEncoder(model_name="dinov2_vitb14"),
            ... )
        """
        self._clip_encoder = clip_encoder or CLIPEncoder()
        self._dino_encoder = dino_encoder or DINOEncoder()
        
        logger.info(
            f"Initialized HybridEncoder with "
            f"CLIP={self._clip_encoder.model_name}, "
            f"DINO={self._dino_encoder.model_name}"
        )
    
    @property
    def clip_encoder(self) -> CLIPEncoder:
        """Return the CLIP encoder."""
        return self._clip_encoder
    
    @property
    def dino_encoder(self) -> DINOEncoder:
        """Return the DINO encoder."""
        return self._dino_encoder
    
    @property
    def clip_dim(self) -> int:
        """Return CLIP embedding dimension."""
        return self._clip_encoder.embedding_dim
    
    @property
    def dino_dim(self) -> int:
        """Return DINO embedding dimension."""
        return self._dino_encoder.embedding_dim
    
    @property
    def total_dim(self) -> int:
        """Return total embedding dimension (CLIP + DINO)."""
        return self.clip_dim + self.dino_dim
    
    def is_loaded(self) -> bool:
        """Check if both models are loaded."""
        return (
            self._clip_encoder.is_loaded() and 
            self._dino_encoder.is_loaded()
        )
    
    def load_models(self) -> None:
        """
        Load both CLIP and DINO models.
        
        Downloads models on first use (cached by Hugging Face).
        
        Example:
            >>> hybrid = HybridEncoder()
            >>> hybrid.load_models()
            >>> print(hybrid.is_loaded())  # True
        """
        logger.info("Loading hybrid encoder models...")
        
        if not self._clip_encoder.is_loaded():
            self._clip_encoder.load_model()
        
        if not self._dino_encoder.is_loaded():
            self._dino_encoder.load_model()
        
        logger.info(
            f"Hybrid encoder ready: "
            f"CLIP({self.clip_dim}d) + DINO({self.dino_dim}d) = {self.total_dim}d"
        )
    
    def unload_models(self) -> None:
        """Unload both models to free memory."""
        self._clip_encoder.unload_model()
        self._dino_encoder.unload_model()
        logger.info("Hybrid encoder models unloaded")
    
    def encode_all(
        self,
        image: Union[str, Path, Image.Image],
    ) -> HybridEmbedding:
        """
        Generate both CLIP and DINO embeddings for an image.
        
        Args:
            image: Image path (str/Path) or PIL Image.
            
        Returns:
            HybridEmbedding containing both embeddings.
            
        Raises:
            RuntimeError: If models not loaded.
            
        Example:
            >>> result = hybrid.encode_all("image.jpg")
            >>> print(result.clip_embedding.shape)  # (512,)
            >>> print(result.dino_embedding.shape)  # (384,)
            >>> combined = result.concatenate()  # (896,)
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Models not loaded. Call load_models() first."
            )
        
        # Generate both embeddings
        clip_emb = self._clip_encoder.encode(image)
        dino_emb = self._dino_encoder.encode(image)
        
        return HybridEmbedding(
            clip_embedding=clip_emb,
            dino_embedding=dino_emb,
        )
    
    def encode_all_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> List[HybridEmbedding]:
        """
        Generate hybrid embeddings for multiple images.
        
        Args:
            images: List of image paths or PIL Images.
            batch_size: Batch size for processing.
            
        Returns:
            List of HybridEmbedding objects.
        """
        if not self.is_loaded():
            raise RuntimeError(
                "Models not loaded. Call load_models() first."
            )
        
        # Batch encode with both models
        clip_embeddings = self._clip_encoder.encode_batch(images, batch_size)
        dino_embeddings = self._dino_encoder.encode_batch(images, batch_size)
        
        # Combine into HybridEmbedding objects
        results = [
            HybridEmbedding(
                clip_embedding=clip_embeddings[i],
                dino_embedding=dino_embeddings[i],
            )
            for i in range(len(images))
        ]
        
        logger.info(f"Generated {len(results)} hybrid embeddings")
        return results
    
    def encode_clip_only(
        self,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Generate only CLIP embedding.
        
        Useful when you only need semantic features.
        """
        return self._clip_encoder.encode(image)
    
    def encode_dino_only(
        self,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Generate only DINO embedding.
        
        Useful when you only need structural features.
        """
        return self._dino_encoder.encode(image)
    
    def __enter__(self) -> "HybridEncoder":
        """Context manager entry - load models."""
        self.load_models()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - unload models."""
        self.unload_models()
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"HybridEncoder("
            f"clip={self._clip_encoder.model_name}, "
            f"dino={self._dino_encoder.model_name}, "
            f"total_dim={self.total_dim}, "
            f"loaded={self.is_loaded()})"
        )

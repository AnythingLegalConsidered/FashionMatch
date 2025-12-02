# Encoders Package
"""
Image encoders for generating embeddings (CLIP, DINOv2, Hybrid).

Provides:
- BaseEncoder: Abstract base class for all encoders
- CLIPEncoder: OpenAI CLIP encoder for semantic embeddings
- DINOEncoder: Meta DINOv2 encoder for structural embeddings
- HybridEncoder: Orchestrates both encoders for combined embeddings
- HybridEmbedding: Dataclass containing both embedding types

Example:
    >>> from src.core.encoders import HybridEncoder
    >>> 
    >>> # Use hybrid encoder for combined CLIP + DINO
    >>> with HybridEncoder() as encoder:
    ...     result = encoder.encode_all("image.jpg")
    ...     print(result.clip_embedding.shape)  # (512,)
    ...     print(result.dino_embedding.shape)  # (384,)
"""

from .base_encoder import BaseEncoder
from .clip_encoder import CLIPEncoder
from .dino_encoder import DINOEncoder
from .hybrid_encoder import HybridEmbedding, HybridEncoder

__all__ = [
    "BaseEncoder",
    "CLIPEncoder",
    "DINOEncoder",
    "HybridEncoder",
    "HybridEmbedding",
]

# Encoders Package
"""
Image encoders for generating embeddings (CLIP, DINOv2, Hybrid).

Provides:
- BaseEncoder: Abstract base class for all encoders
- CLIPEncoder: OpenAI CLIP encoder for semantic embeddings
- DINOEncoder: Meta DINOv2 encoder for structural embeddings

Example:
    >>> from src.core.encoders import CLIPEncoder, DINOEncoder
    >>> 
    >>> # CLIP for semantic understanding
    >>> clip = CLIPEncoder()
    >>> clip.load_model()
    >>> semantic_emb = clip.encode("image.jpg")  # (512,)
    >>> 
    >>> # DINO for structural features
    >>> dino = DINOEncoder()
    >>> dino.load_model()
    >>> structural_emb = dino.encode("image.jpg")  # (384,)
"""

from .base_encoder import BaseEncoder
from .clip_encoder import CLIPEncoder
from .dino_encoder import DINOEncoder

__all__ = [
    "BaseEncoder",
    "CLIPEncoder",
    "DINOEncoder",
]

"""Core AI components for FashionMatch."""

from .embedding_pipeline import EmbeddingPipeline, PipelineStats
from .encoders import CLIPEncoder, DINOv2Encoder, get_clip_encoder, get_dino_encoder
from .scorer import HybridScorer, get_hybrid_scorer

__all__ = [
    "CLIPEncoder",
    "DINOv2Encoder",
    "get_clip_encoder",
    "get_dino_encoder",
    "HybridScorer",
    "get_hybrid_scorer",
    "EmbeddingPipeline",
    "PipelineStats",
]

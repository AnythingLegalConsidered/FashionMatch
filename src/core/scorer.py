"""Late fusion scorer for combining CLIP and DINOv2 embeddings.

This module implements weighted cosine similarity for fashion item matching
by fusing semantic (CLIP) and structural (DINOv2) embeddings.
"""

from typing import Optional

import numpy as np
from PIL import Image

from src.utils import get_logger
from src.utils.config import FusionWeights, ModelConfig

from .clip_encoder import CLIPEncoder, get_clip_encoder
from .dino_encoder import DINOv2Encoder, get_dino_encoder

logger = get_logger(__name__)


class HybridScorer:
    """Scorer for combining CLIP and DINOv2 embeddings with weighted cosine similarity."""
    
    def __init__(
        self,
        clip_encoder: CLIPEncoder,
        dino_encoder: DINOv2Encoder,
        fusion_weights: FusionWeights
    ):
        """Initialize hybrid scorer.
        
        Args:
            clip_encoder: CLIP encoder instance
            dino_encoder: DINOv2 encoder instance
            fusion_weights: Weights for combining CLIP and DINOv2 scores
        """
        self.clip_encoder = clip_encoder
        self.dino_encoder = dino_encoder
        self.alpha = fusion_weights.clip  # CLIP weight
        self.beta = fusion_weights.dino   # DINOv2 weight
        
        logger.info(
            f"Initialized HybridScorer with weights: "
            f"CLIP={self.alpha:.2f}, DINOv2={self.beta:.2f}"
        )
    
    def encode_dual(
        self, images: Image.Image | list[Image.Image]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode images with both CLIP and DINOv2.
        
        Args:
            images: Single PIL Image or list of PIL Images
            
        Returns:
            Tuple of (clip_embeddings, dino_embeddings) as numpy arrays
            Shapes: (N, clip_dim) and (N, dino_dim)
        """
        # Encode with both models
        clip_embeddings = self.clip_encoder.encode(images)
        dino_embeddings = self.dino_encoder.encode(images)
        
        return clip_embeddings, dino_embeddings
    
    def compute_similarity(
        self,
        query_clip: np.ndarray,
        query_dino: np.ndarray,
        candidate_clip: np.ndarray,
        candidate_dino: np.ndarray
    ) -> np.ndarray:
        """Compute weighted cosine similarity between query and candidates.
        
        Since embeddings are L2-normalized, cosine similarity simplifies to dot product.
        
        Args:
            query_clip: Query CLIP embeddings, shape (N_query, clip_dim)
            query_dino: Query DINOv2 embeddings, shape (N_query, dino_dim)
            candidate_clip: Candidate CLIP embeddings, shape (N_candidates, clip_dim)
            candidate_dino: Candidate DINOv2 embeddings, shape (N_candidates, dino_dim)
            
        Returns:
            Similarity scores, shape (N_query, N_candidates)
        """
        # Validate embedding shapes
        if query_clip.shape[1] != candidate_clip.shape[1]:
            raise ValueError(
                f"CLIP embedding dimension mismatch: query has {query_clip.shape[1]} dims, "
                f"candidates have {candidate_clip.shape[1]} dims"
            )
        
        if query_dino.shape[1] != candidate_dino.shape[1]:
            raise ValueError(
                f"DINOv2 embedding dimension mismatch: query has {query_dino.shape[1]} dims, "
                f"candidates have {candidate_dino.shape[1]} dims"
            )
        
        if query_clip.shape[0] != query_dino.shape[0]:
            raise ValueError(
                f"Number of queries mismatch: CLIP has {query_clip.shape[0]}, "
                f"DINOv2 has {query_dino.shape[0]}"
            )
        
        # Compute cosine similarity using dot product (embeddings are normalized)
        clip_similarity = np.dot(query_clip, candidate_clip.T)
        dino_similarity = np.dot(query_dino, candidate_dino.T)
        
        # Apply late fusion with weights
        fused_score = self.alpha * clip_similarity + self.beta * dino_similarity
        
        return fused_score
    
    def rank_candidates(
        self,
        query_clip: np.ndarray,
        query_dino: np.ndarray,
        candidate_clip: np.ndarray,
        candidate_dino: np.ndarray
    ) -> list[tuple[int, float]]:
        """Rank candidates by similarity to query.
        
        Args:
            query_clip: Query CLIP embedding, shape (1, clip_dim) or (clip_dim,)
            query_dino: Query DINOv2 embedding, shape (1, dino_dim) or (dino_dim,)
            candidate_clip: Candidate CLIP embeddings, shape (N, clip_dim)
            candidate_dino: Candidate DINOv2 embeddings, shape (N, dino_dim)
            
        Returns:
            List of (candidate_index, similarity_score) sorted by score descending
        """
        # Ensure query is 2D
        if query_clip.ndim == 1:
            query_clip = query_clip.reshape(1, -1)
        if query_dino.ndim == 1:
            query_dino = query_dino.reshape(1, -1)
        
        # Compute similarities
        scores = self.compute_similarity(
            query_clip, query_dino, candidate_clip, candidate_dino
        )
        
        # Flatten scores (single query)
        scores = scores.flatten()
        
        # Sort by score descending
        sorted_indices = np.argsort(scores)[::-1]
        
        # Create ranked list
        ranked = [(int(idx), float(scores[idx])) for idx in sorted_indices]
        
        return ranked
    
    def update_weights(self, fusion_weights: FusionWeights) -> None:
        """Update fusion weights dynamically.
        
        Args:
            fusion_weights: New weights for CLIP and DINOv2
        """
        old_alpha, old_beta = self.alpha, self.beta
        self.alpha = fusion_weights.clip
        self.beta = fusion_weights.dino
        
        logger.info(
            f"Updated fusion weights: "
            f"CLIP {old_alpha:.2f}→{self.alpha:.2f}, "
            f"DINOv2 {old_beta:.2f}→{self.beta:.2f}"
        )
    
    def get_clip_dim(self) -> int:
        """Get CLIP embedding dimension."""
        return self.clip_encoder.get_embedding_dim()
    
    def get_dino_dim(self) -> int:
        """Get DINOv2 embedding dimension."""
        return self.dino_encoder.get_embedding_dim()


# Singleton pattern for scorer caching
_scorer_cache: dict[str, HybridScorer] = {}


def get_hybrid_scorer(config: ModelConfig) -> HybridScorer:
    """Get or create a HybridScorer instance (singleton pattern).
    
    Args:
        config: Model configuration containing encoder and fusion settings
        
    Returns:
        Cached or newly created HybridScorer instance
    """
    # Create cache key from config
    cache_key = f"{config.clip_model}_{config.dino_model}_{config.device}_{config.fusion_weights.clip}"
    
    if cache_key not in _scorer_cache:
        logger.debug(f"Creating new HybridScorer")
        
        # Get encoders (these are cached internally)
        clip_encoder = get_clip_encoder(config)
        dino_encoder = get_dino_encoder(config)
        
        # Create scorer
        _scorer_cache[cache_key] = HybridScorer(
            clip_encoder=clip_encoder,
            dino_encoder=dino_encoder,
            fusion_weights=config.fusion_weights
        )
    else:
        logger.debug(f"Reusing cached HybridScorer")
    
    return _scorer_cache[cache_key]

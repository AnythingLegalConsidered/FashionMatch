# Scoring Package
"""
Similarity scoring and late fusion implementations.

Provides:
- cosine_similarity: Compute similarity between embeddings
- batch_cosine_similarity: Efficient batch similarity computation
- WeightedScorer: Late fusion of CLIP and DINO scores
- FusionWeights: Configuration for fusion weights

Example:
    >>> from src.core.scoring import cosine_similarity, WeightedScorer
    >>> 
    >>> # Compute similarity
    >>> sim = cosine_similarity(emb_a, emb_b)
    >>> 
    >>> # Late fusion scoring
    >>> scorer = WeightedScorer()  # Uses config weights
    >>> final = scorer.compute_score(clip_sim=0.8, dino_sim=0.7)
"""

from .similarity import (
    batch_cosine_similarity,
    batch_cosine_similarity_normalized,
    cosine_similarity,
    cosine_similarity_normalized,
)
from .weighted_scorer import FusionWeights, WeightedScorer

__all__ = [
    # Similarity functions
    "cosine_similarity",
    "cosine_similarity_normalized",
    "batch_cosine_similarity",
    "batch_cosine_similarity_normalized",
    # Weighted scoring
    "FusionWeights",
    "WeightedScorer",
]

"""
Weighted scorer for late fusion of CLIP and DINO similarities.

Implements configurable weighted combination of semantic (CLIP) and
structural (DINO) similarity scores.

Example:
    >>> from src.core.scoring import WeightedScorer
    >>> scorer = WeightedScorer()  # Uses config weights
    >>> final_score = scorer.compute_score(clip_sim=0.8, dino_sim=0.6)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FusionWeights:
    """
    Weights for combining CLIP and DINO similarities.
    
    The weights control how much each model contributes to the final score:
    - Higher CLIP weight → More semantic matching (categories, style)
    - Higher DINO weight → More structural matching (patterns, textures)
    
    Attributes:
        clip: Weight for CLIP similarity (0.0 to 1.0).
        dino: Weight for DINO similarity (0.0 to 1.0).
        
    Example:
        >>> weights = FusionWeights(clip=0.6, dino=0.4)
        >>> # 60% semantic, 40% structural
    """
    
    clip: float = 0.5
    dino: float = 0.5

    def __post_init__(self) -> None:
        """Validate that weights are valid."""
        if not (0 <= self.clip <= 1):
            raise ValueError(f"CLIP weight must be in [0, 1], got {self.clip}")
        if not (0 <= self.dino <= 1):
            raise ValueError(f"DINO weight must be in [0, 1], got {self.dino}")
        if abs(self.clip + self.dino - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {self.clip + self.dino:.4f}"
            )
    
    @classmethod
    def from_config(cls) -> "FusionWeights":
        """
        Create FusionWeights from config.yaml settings.
        
        Returns:
            FusionWeights with values from models.fusion.weights.
        """
        settings = get_settings()
        return cls(
            clip=settings.models.fusion.weights.clip,
            dino=settings.models.fusion.weights.dino,
        )
    
    def to_tuple(self) -> Tuple[float, float]:
        """Return weights as (clip, dino) tuple."""
        return (self.clip, self.dino)


class WeightedScorer:
    """
    Late fusion scorer combining CLIP and DINO similarities.
    
    Uses weighted average of similarity scores from both models:
    
        final_score = (clip_weight × clip_sim) + (dino_weight × dino_sim)
    
    The weights are read from config.yaml by default but can be overridden.
    
    Attributes:
        weights: Current fusion weights.
        
    Example:
        >>> scorer = WeightedScorer()  # Uses config weights
        >>> 
        >>> # Single score
        >>> score = scorer.compute_score(clip_sim=0.85, dino_sim=0.72)
        >>> 
        >>> # Batch scoring
        >>> scores = scorer.compute_score_batch(
        ...     clip_sims=[0.85, 0.72, 0.91],
        ...     dino_sims=[0.72, 0.88, 0.65],
        ... )
    """

    def __init__(self, weights: Optional[FusionWeights] = None):
        """
        Initialize the scorer.
        
        Args:
            weights: Fusion weights. If None, reads from config.yaml.
            
        Example:
            >>> # Use config weights
            >>> scorer = WeightedScorer()
            >>> 
            >>> # Use custom weights
            >>> scorer = WeightedScorer(FusionWeights(clip=0.7, dino=0.3))
        """
        if weights is None:
            self._weights = FusionWeights.from_config()
            logger.debug(
                f"WeightedScorer initialized from config: "
                f"CLIP={self._weights.clip}, DINO={self._weights.dino}"
            )
        else:
            self._weights = weights
            logger.debug(
                f"WeightedScorer initialized with custom weights: "
                f"CLIP={self._weights.clip}, DINO={self._weights.dino}"
            )
    
    @property
    def weights(self) -> FusionWeights:
        """Return current fusion weights."""
        return self._weights
    
    @property
    def clip_weight(self) -> float:
        """Return CLIP weight."""
        return self._weights.clip
    
    @property
    def dino_weight(self) -> float:
        """Return DINO weight."""
        return self._weights.dino

    def compute_score(
        self,
        clip_sim: float,
        dino_sim: float,
    ) -> float:
        """
        Compute weighted fusion score for a single pair.
        
        Args:
            clip_sim: CLIP similarity score (typically 0-1).
            dino_sim: DINO similarity score (typically 0-1).
            
        Returns:
            Weighted average similarity score.
            
        Example:
            >>> score = scorer.compute_score(clip_sim=0.85, dino_sim=0.72)
            >>> print(f"Final score: {score:.4f}")
        """
        return (
            self._weights.clip * clip_sim + 
            self._weights.dino * dino_sim
        )
    
    def compute_score_batch(
        self,
        clip_sims: List[float],
        dino_sims: List[float],
    ) -> np.ndarray:
        """
        Compute weighted fusion scores for multiple pairs.
        
        Args:
            clip_sims: List of CLIP similarity scores.
            dino_sims: List of DINO similarity scores.
            
        Returns:
            Numpy array of weighted scores.
            
        Raises:
            ValueError: If input lists have different lengths.
        """
        if len(clip_sims) != len(dino_sims):
            raise ValueError(
                f"Input lists must have same length: "
                f"{len(clip_sims)} vs {len(dino_sims)}"
            )
        
        clip_arr = np.asarray(clip_sims, dtype=np.float32)
        dino_arr = np.asarray(dino_sims, dtype=np.float32)
        
        return (
            self._weights.clip * clip_arr + 
            self._weights.dino * dino_arr
        )
    
    def compute_combination(
        self,
        clip_score: float,
        dino_score: float,
    ) -> float:
        """
        Alias for compute_score (matches PLAN.md naming).
        
        Args:
            clip_score: CLIP similarity score.
            dino_score: DINO similarity score.
            
        Returns:
            Combined weighted score.
        """
        return self.compute_score(clip_score, dino_score)

    def update_weights(
        self,
        clip_weight: Optional[float] = None,
        dino_weight: Optional[float] = None,
    ) -> None:
        """
        Update fusion weights.
        
        If only one weight is provided, the other is calculated
        to ensure they sum to 1.0.
        
        Args:
            clip_weight: New CLIP weight (0-1).
            dino_weight: New DINO weight (0-1).
            
        Example:
            >>> scorer.update_weights(clip_weight=0.7)
            >>> # dino_weight automatically set to 0.3
        """
        if clip_weight is not None and dino_weight is not None:
            new_weights = FusionWeights(clip=clip_weight, dino=dino_weight)
        elif clip_weight is not None:
            new_weights = FusionWeights(clip=clip_weight, dino=1.0 - clip_weight)
        elif dino_weight is not None:
            new_weights = FusionWeights(clip=1.0 - dino_weight, dino=dino_weight)
        else:
            return  # Nothing to update
        
        self._weights = new_weights
        logger.info(
            f"Updated weights: CLIP={self._weights.clip}, DINO={self._weights.dino}"
        )

    def reset_to_config(self) -> None:
        """Reset weights to config.yaml values."""
        self._weights = FusionWeights.from_config()
        logger.info(
            f"Reset weights to config: "
            f"CLIP={self._weights.clip}, DINO={self._weights.dino}"
        )

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"WeightedScorer(clip={self._weights.clip}, "
            f"dino={self._weights.dino})"
        )

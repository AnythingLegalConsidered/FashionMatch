"""
Weighted scorer for late fusion of CLIP and DINO similarities.
"""

from dataclasses import dataclass


@dataclass
class FusionWeights:
    """Weights for combining CLIP and DINO similarities."""

    clip: float = 0.5
    dino: float = 0.5

    def __post_init__(self) -> None:
        """Validate that weights sum to 1."""
        if not (0 <= self.clip <= 1):
            raise ValueError(f"CLIP weight must be in [0, 1], got {self.clip}")
        if not (0 <= self.dino <= 1):
            raise ValueError(f"DINO weight must be in [0, 1], got {self.dino}")
        if abs(self.clip + self.dino - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1, got {self.clip + self.dino}"
            )


class WeightedScorer:
    """
    Late fusion scorer combining CLIP and DINO similarities.
    
    Uses weighted average of similarity scores from both models.
    """

    def __init__(self, weights: FusionWeights | None = None):
        """
        Initialize the scorer.
        
        Args:
            weights: Fusion weights. Defaults to equal weights.
        """
        self.weights = weights or FusionWeights()

    def compute_score(
        self,
        clip_sim: float,
        dino_sim: float,
    ) -> float:
        """
        Compute weighted fusion score.
        
        Args:
            clip_sim: CLIP similarity score (0-1).
            dino_sim: DINO similarity score (0-1).
            
        Returns:
            Weighted average similarity score.
        """
        return (
            self.weights.clip * clip_sim + 
            self.weights.dino * dino_sim
        )

    def update_weights(self, new_weights: FusionWeights) -> None:
        """
        Update fusion weights.
        
        Args:
            new_weights: New fusion weights to use.
        """
        self.weights = new_weights

    def get_weights(self) -> FusionWeights:
        """Return current fusion weights."""
        return self.weights

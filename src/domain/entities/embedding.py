"""
Embedding value object for storing vector embeddings.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


class EmbeddingType(Enum):
    """Types of embeddings supported."""

    CLIP = "clip"
    DINO = "dino"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class Embedding:
    """
    Value object representing a vector embedding.
    
    Immutable to ensure embedding integrity once created.
    """

    vector: tuple[float, ...]  # Immutable tuple for hashability
    embedding_type: EmbeddingType
    dimension: int

    def __post_init__(self) -> None:
        """Validate embedding dimensions."""
        if len(self.vector) != self.dimension:
            raise ValueError(
                f"Vector length {len(self.vector)} does not match "
                f"declared dimension {self.dimension}"
            )

    @classmethod
    def from_list(
        cls, 
        vector: List[float], 
        embedding_type: EmbeddingType
    ) -> "Embedding":
        """Create an Embedding from a list of floats."""
        return cls(
            vector=tuple(vector),
            embedding_type=embedding_type,
            dimension=len(vector),
        )

    def to_list(self) -> List[float]:
        """Convert embedding vector to list."""
        return list(self.vector)

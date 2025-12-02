"""
ClothingItem entity representing a clothing item from Vinted.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class ClothingItem:
    """Represents a clothing item from Vinted."""

    id: str
    title: str
    price: float
    currency: str = "EUR"
    brand: Optional[str] = None
    size: Optional[str] = None
    condition: Optional[str] = None
    category: Optional[str] = None
    image_url: str = ""
    local_image_path: Optional[str] = None
    item_url: str = ""
    description: Optional[str] = None
    seller_id: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.utcnow)

    # Embeddings (populated later by encoders)
    clip_embedding: Optional[List[float]] = None
    dino_embedding: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate required fields after initialization."""
        if not self.id:
            raise ValueError("ClothingItem id cannot be empty")
        if not self.title:
            raise ValueError("ClothingItem title cannot be empty")
        if self.price < 0:
            raise ValueError("ClothingItem price cannot be negative")

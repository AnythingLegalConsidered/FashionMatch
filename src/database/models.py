"""Pydantic data models for fashion items and search results.

This module defines type-safe models for representing fashion items
with embeddings and search results from the vector store.
"""

from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class FashionItem(BaseModel):
    """Model representing a fashion item with embeddings and metadata."""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Identifiers
    item_id: str = Field(..., description="Unique item identifier")
    
    # Embeddings (optional during construction, populated during encoding)
    clip_embedding: Optional[np.ndarray] = Field(default=None, description="CLIP embedding vector")
    dino_embedding: Optional[np.ndarray] = Field(default=None, description="DINOv2 embedding vector")
    
    # Core metadata from VintedItem
    title: str = Field(..., description="Product title")
    price: float = Field(..., ge=0.0, description="Price in base currency")
    currency: str = Field(default="EUR", description="Currency code")
    description: Optional[str] = Field(default=None, description="Product description")
    brand: Optional[str] = Field(default=None, description="Brand name")
    size: Optional[str] = Field(default=None, description="Size information")
    condition: Optional[str] = Field(default=None, description="Item condition")
    image_url: str = Field(..., description="Primary image URL")
    url: str = Field(..., description="Product page URL")
    category: Optional[str] = Field(default=None, description="Product category")
    scraped_at: datetime = Field(default_factory=datetime.now, description="Timestamp when scraped")
    
    # Local storage
    local_image_path: Optional[str] = Field(default=None, description="Path to downloaded image")
    
    @field_validator('clip_embedding', 'dino_embedding')
    @classmethod
    def validate_embedding(cls, v: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Validate embedding is 1D numpy array with float32 dtype."""
        if v is None:
            return v
        
        if not isinstance(v, np.ndarray):
            raise ValueError("Embedding must be a numpy array")
        
        if v.ndim != 1:
            raise ValueError(f"Embedding must be 1D, got shape {v.shape}")
        
        if v.dtype != np.float32:
            # Convert to float32 if needed
            v = v.astype(np.float32)
        
        return v
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Ensure price is non-negative."""
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v


class SearchResult(BaseModel):
    """Model representing a ranked search result."""
    
    item_id: str = Field(..., description="Item identifier")
    similarity_score: float = Field(..., description="Fused similarity score")
    clip_score: float = Field(..., description="CLIP similarity score")
    dino_score: float = Field(..., description="DINOv2 similarity score")
    item: FashionItem = Field(..., description="Fashion item details")
    
    @field_validator('similarity_score', 'clip_score', 'dino_score')
    @classmethod
    def validate_score(cls, v: float) -> float:
        """Validate score is in valid range."""
        # Enforce [-1, 1] bounds for distance metric scores
        if not -1.0 <= v <= 1.0:
            raise ValueError(f"Score {v} is outside valid range [-1.0, 1.0]")
        return v


class BatchInsertResult(BaseModel):
    """Model tracking batch insertion statistics."""
    
    success_count: int = Field(..., ge=0, description="Number of successfully inserted items")
    failed_ids: list[str] = Field(default_factory=list, description="List of failed item IDs")
    total_time: float = Field(..., ge=0.0, description="Total operation time in seconds")


def vinted_item_to_fashion_item(vinted_item) -> FashionItem:
    """Convert VintedItem to FashionItem (without embeddings).
    
    Args:
        vinted_item: VintedItem instance from scraper
        
    Returns:
        FashionItem with metadata populated, embeddings set to None
    """
    # Get primary image URL and local path
    image_url = vinted_item.image_urls[0] if vinted_item.image_urls else ""
    local_image_path = vinted_item.local_image_paths[0] if vinted_item.local_image_paths else None
    
    return FashionItem(
        item_id=vinted_item.item_id,
        clip_embedding=None,
        dino_embedding=None,
        title=vinted_item.title,
        price=vinted_item.price,
        currency=vinted_item.currency,
        description=vinted_item.description,
        brand=vinted_item.brand,
        size=vinted_item.size,
        condition=vinted_item.condition,
        image_url=image_url,
        url=vinted_item.url,
        category=vinted_item.category,
        scraped_at=vinted_item.scraped_at,
        local_image_path=local_image_path
    )


def fashion_item_to_metadata(item: FashionItem) -> dict:
    """Extract metadata dict for ChromaDB storage (excluding embeddings).
    
    Args:
        item: FashionItem instance
        
    Returns:
        Dictionary with metadata fields for ChromaDB
    """
    return {
        'title': item.title,
        'price': item.price,
        'currency': item.currency,
        'description': item.description or '',
        'brand': item.brand or '',
        'size': item.size or '',
        'condition': item.condition or '',
        'image_url': item.image_url,
        'url': item.url,
        'category': item.category or '',
        'scraped_at': item.scraped_at.isoformat(),
        'local_image_path': item.local_image_path or '',
    }


def metadata_to_fashion_item(
    item_id: str,
    metadata: dict,
    clip_embedding: Optional[np.ndarray] = None,
    dino_embedding: Optional[np.ndarray] = None
) -> FashionItem:
    """Reconstruct FashionItem from ChromaDB metadata and embeddings.
    
    Args:
        item_id: Item identifier
        metadata: Metadata dictionary from ChromaDB
        clip_embedding: CLIP embedding vector
        dino_embedding: DINOv2 embedding vector
        
    Returns:
        Reconstructed FashionItem instance
    """
    return FashionItem(
        item_id=item_id,
        clip_embedding=clip_embedding,
        dino_embedding=dino_embedding,
        title=metadata.get('title', ''),
        price=float(metadata.get('price', 0.0)),
        currency=metadata.get('currency', 'EUR'),
        description=metadata.get('description') or None,
        brand=metadata.get('brand') or None,
        size=metadata.get('size') or None,
        condition=metadata.get('condition') or None,
        image_url=metadata.get('image_url', ''),
        url=metadata.get('url', ''),
        category=metadata.get('category') or None,
        scraped_at=datetime.fromisoformat(metadata.get('scraped_at', datetime.now().isoformat())),
        local_image_path=metadata.get('local_image_path') or None
    )

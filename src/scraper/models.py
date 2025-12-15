"""Pydantic data models for scraped Vinted items.

This module defines type-safe models for representing Vinted products
and scraping batches with validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator, HttpUrl


class VintedItem(BaseModel):
    """Model representing a single Vinted item."""
    
    item_id: str = Field(..., description="Unique Vinted item identifier")
    title: str = Field(..., description="Product title")
    price: float = Field(..., ge=0.0, description="Price in euros")
    currency: str = Field(default="EUR", description="Currency code")
    description: Optional[str] = Field(default=None, description="Product description")
    brand: Optional[str] = Field(default=None, description="Brand name")
    size: Optional[str] = Field(default=None, description="Size information")
    condition: Optional[str] = Field(default=None, description="Item condition")
    image_urls: list[str] = Field(default_factory=list, description="List of image URLs")
    url: str = Field(..., description="Product page URL")
    seller_id: Optional[str] = Field(default=None, description="Seller identifier")
    category: Optional[str] = Field(default=None, description="Product category")
    scraped_at: datetime = Field(default_factory=datetime.now, description="Timestamp of scraping")
    local_image_paths: list[str] = Field(default_factory=list, description="Paths to downloaded images")
    
    @field_validator('price')
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Ensure price is non-negative."""
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v
    
    @field_validator('image_urls')
    @classmethod
    def validate_images(cls, v: list[str]) -> list[str]:
        """Ensure at least one image URL exists."""
        if not v:
            raise ValueError("At least one image URL is required")
        return v
    
    @field_validator('url')
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Validate URL format."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
        return v


class ScrapedBatch(BaseModel):
    """Model representing a batch of scraped items with metadata."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    category: str = Field(..., description="Search category")
    total_items: int = Field(..., ge=0, description="Number of items scraped")
    pages_scraped: int = Field(..., ge=0, description="Number of pages processed")
    scraped_at: datetime = Field(default_factory=datetime.now, description="Batch timestamp")
    items: list[VintedItem] = Field(default_factory=list, description="List of scraped items")
    
    @field_validator('total_items')
    @classmethod
    def validate_total_items(cls, v: int, info) -> int:
        """Ensure total_items matches items list length if items are present."""
        # Note: This validation runs before items is set, so we can't check consistency here
        # Instead, we ensure consistency in the batch creation logic
        return v

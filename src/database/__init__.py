"""Database module for fashion item storage and retrieval."""

from .exceptions import (
    BatchInsertError,
    CollectionError,
    EmbeddingDimensionError,
    ItemNotFoundError,
    VectorStoreError,
)
from .models import (
    BatchInsertResult,
    FashionItem,
    SearchResult,
    fashion_item_to_metadata,
    metadata_to_fashion_item,
    vinted_item_to_fashion_item,
)
from .vector_store import VectorStore, get_vector_store

__all__ = [
    # Exceptions
    "VectorStoreError",
    "EmbeddingDimensionError",
    "ItemNotFoundError",
    "CollectionError",
    "BatchInsertError",
    # Models
    "FashionItem",
    "SearchResult",
    "BatchInsertResult",
    # Conversion helpers
    "vinted_item_to_fashion_item",
    "fashion_item_to_metadata",
    "metadata_to_fashion_item",
    # Vector store
    "VectorStore",
    "get_vector_store",
]

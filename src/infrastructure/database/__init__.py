# Database Package
"""
Database implementations for vector storage.

Provides:
- ChromaRepository: ChromaDB-based vector storage for clothing items

Note:
    ChromaDB requires Python < 3.14 due to onnxruntime dependency.
    For Python 3.14+, ChromaDB will not be available until onnxruntime
    adds support for Python 3.14.

Example:
    >>> from src.infrastructure.database import ChromaRepository
    >>> repo = ChromaRepository()
    >>> repo.add_item(item, clip_emb, dino_emb)
    >>> results = repo.search_similar(query_emb, "clip")
"""

from src.infrastructure.database.chroma_repository import (
    ChromaRepository,
    CLIP_COLLECTION_NAME,
    DINO_COLLECTION_NAME,
    CLIP_EMBEDDING_DIM,
    DINO_EMBEDDING_DIM,
    CHROMADB_AVAILABLE,
)

__all__ = [
    "ChromaRepository",
    "CLIP_COLLECTION_NAME",
    "DINO_COLLECTION_NAME",
    "CLIP_EMBEDDING_DIM",
    "DINO_EMBEDDING_DIM",
    "CHROMADB_AVAILABLE",
]

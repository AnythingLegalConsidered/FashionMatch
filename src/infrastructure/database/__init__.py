# Database Package
"""
Database implementations for vector storage.

Provides:
- ChromaRepository: ChromaDB-based vector storage for clothing items
- FusionWeights: Configuration for hybrid search late fusion

Note:
    ChromaDB requires Python < 3.14 due to onnxruntime dependency.
    For Python 3.14+, ChromaDB will not be available until onnxruntime
    adds support for Python 3.14.

Example:
    >>> from src.infrastructure.database import ChromaRepository, FusionWeights
    >>> repo = ChromaRepository()
    >>> weights = FusionWeights(clip=0.6, dino=0.4)
    >>> items = repo.hybrid_search(clip_emb, dino_emb, weights, limit=20)
"""

from src.infrastructure.database.chroma_repository import (
    ChromaRepository,
    FusionWeights,
    CLIP_COLLECTION_NAME,
    DINO_COLLECTION_NAME,
    CLIP_EMBEDDING_DIM,
    DINO_EMBEDDING_DIM,
    CHROMADB_AVAILABLE,
)

__all__ = [
    "ChromaRepository",
    "FusionWeights",
    "CLIP_COLLECTION_NAME",
    "DINO_COLLECTION_NAME",
    "CLIP_EMBEDDING_DIM",
    "DINO_EMBEDDING_DIM",
    "CHROMADB_AVAILABLE",
]

"""
ChromaDB repository implementation for vector storage.

Provides persistent storage for clothing item embeddings with
two separate collections for CLIP and DINO vectors.

Features:
- Dual collections for CLIP (512d) and DINO (384d) embeddings
- Metadata filtering support
- Hybrid search with late fusion
- Batch operations for efficiency

Note:
    ChromaDB requires Python < 3.14 due to onnxruntime dependency.
    For Python 3.14+, use: pip install chromadb with Python 3.12 or earlier.

Example:
    >>> from src.infrastructure.database import ChromaRepository
    >>> repo = ChromaRepository()
    >>> repo.add_item(item, clip_emb, dino_emb)
    >>> results = repo.search_similar(query_emb, "clip", n_results=10)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# ChromaDB import with fallback
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None  # type: ignore
    ChromaSettings = None  # type: ignore

from src.domain.entities.clothing_item import ClothingItem
from src.domain.interfaces.repository_interface import RepositoryInterface
from src.utils.config import get_settings
from src.utils.exceptions import DatabaseError, EmbeddingMismatchError, ItemNotFoundError
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================
# Constants
# ============================================

CLIP_COLLECTION_NAME = "clothing_items_clip"
DINO_COLLECTION_NAME = "clothing_items_dino"

CLIP_EMBEDDING_DIM = 512
DINO_EMBEDDING_DIM = 384

# ChromaDB distance function (cosine for normalized embeddings)
DISTANCE_FUNCTION = "cosine"


# ============================================
# Helper Functions
# ============================================

def _sanitize_value(value: Any) -> Any:
    """
    Sanitize a value for ChromaDB metadata.
    
    ChromaDB doesn't accept None values, so we convert them to empty strings.
    Also handles datetime conversion.
    
    Args:
        value: Any value to sanitize.
        
    Returns:
        Sanitized value safe for ChromaDB.
    """
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (list, dict)):
        # ChromaDB doesn't support complex types, convert to string
        import json
        return json.dumps(value)
    return value


def _item_to_metadata(item: ClothingItem) -> Dict[str, Any]:
    """
    Convert ClothingItem to ChromaDB metadata dict.
    
    All None values are converted to empty strings because
    ChromaDB doesn't accept None in metadata.
    
    Args:
        item: The clothing item to convert.
        
    Returns:
        Dictionary suitable for ChromaDB metadata.
    """
    metadata = {
        "id": _sanitize_value(item.id),
        "title": _sanitize_value(item.title),
        "price": float(item.price) if item.price is not None else 0.0,
        "currency": _sanitize_value(item.currency),
        "brand": _sanitize_value(item.brand),
        "size": _sanitize_value(item.size),
        "condition": _sanitize_value(item.condition),
        "category": _sanitize_value(item.category),
        "image_url": _sanitize_value(item.image_url),
        "local_image_path": _sanitize_value(item.local_image_path),
        "item_url": _sanitize_value(item.item_url),
        "description": _sanitize_value(item.description)[:1000] if item.description else "",  # Truncate long descriptions
        "seller_id": _sanitize_value(item.seller_id),
        "scraped_at": _sanitize_value(item.scraped_at),
    }
    
    return metadata


def _metadata_to_item(metadata: Dict[str, Any]) -> ClothingItem:
    """
    Convert ChromaDB metadata back to ClothingItem.
    
    Args:
        metadata: Dictionary from ChromaDB.
        
    Returns:
        Reconstructed ClothingItem.
    """
    # Parse scraped_at datetime
    scraped_at = datetime.utcnow()
    if metadata.get("scraped_at"):
        try:
            scraped_at = datetime.fromisoformat(metadata["scraped_at"])
        except (ValueError, TypeError):
            pass
    
    return ClothingItem(
        id=metadata.get("id", ""),
        title=metadata.get("title", ""),
        price=float(metadata.get("price", 0.0)),
        currency=metadata.get("currency", "EUR") or "EUR",
        brand=metadata.get("brand") or None,
        size=metadata.get("size") or None,
        condition=metadata.get("condition") or None,
        category=metadata.get("category") or None,
        image_url=metadata.get("image_url", ""),
        local_image_path=metadata.get("local_image_path") or None,
        item_url=metadata.get("item_url", ""),
        description=metadata.get("description") or None,
        seller_id=metadata.get("seller_id") or None,
        scraped_at=scraped_at,
    )


def _convert_filters_to_chroma(
    filters: Optional[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    """
    Convert simple filters to ChromaDB where clause format.
    
    Args:
        filters: Simple key-value filters.
        
    Returns:
        ChromaDB-compatible where clause.
    """
    if not filters:
        return None
    
    # Build ChromaDB where clause (simple equality filters)
    where_clause = {}
    for key, value in filters.items():
        if value is not None and value != "":
            where_clause[key] = value
    
    return where_clause if where_clause else None


# ============================================
# ChromaRepository Implementation
# ============================================

class ChromaRepository(RepositoryInterface):
    """
    ChromaDB-based vector storage for clothing items.
    
    Maintains two separate collections:
    - clip_collection: For CLIP embeddings (512d, semantic features)
    - dino_collection: For DINO embeddings (384d, structural features)
    
    Both collections share the same item IDs and metadata, allowing
    for hybrid search across both embedding spaces.
    
    Attributes:
        clip_collection: Collection for CLIP embeddings.
        dino_collection: Collection for DINO embeddings.
        
    Example:
        >>> repo = ChromaRepository()
        >>> 
        >>> # Add an item
        >>> item_id = repo.add_item(item, clip_emb, dino_emb)
        >>> 
        >>> # Search by CLIP similarity
        >>> results = repo.search_similar(query_emb, "clip", n_results=10)
    """
    
    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_prefix: Optional[str] = None,
    ):
        """
        Initialize ChromaDB repository.
        
        Args:
            persist_directory: Path for database persistence.
                              If None, reads from config.
            collection_prefix: Optional prefix for collection names.
                              Useful for testing with isolated collections.
                              
        Raises:
            DatabaseError: If ChromaDB initialization fails.
            ImportError: If ChromaDB is not installed.
        """
        # Check if ChromaDB is available
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed or not compatible with your Python version. "
                "ChromaDB requires Python < 3.14 due to onnxruntime dependency. "
                "Please use Python 3.12 or earlier, or install: pip install chromadb"
            )
        
        # Get settings
        settings = get_settings()
        
        # Determine persist directory
        if persist_directory is None:
            persist_directory = settings.database.chroma.persist_directory
        
        self._persist_directory = Path(persist_directory)
        self._collection_prefix = collection_prefix or ""
        
        # Ensure directory exists
        self._persist_directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {self._persist_directory}")
        
        try:
            # Initialize ChromaDB client with persistence
            self._client = chromadb.PersistentClient(
                path=str(self._persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            
            # Create or get CLIP collection
            clip_name = f"{self._collection_prefix}{CLIP_COLLECTION_NAME}"
            self._clip_collection = self._client.get_or_create_collection(
                name=clip_name,
                metadata={
                    "hnsw:space": DISTANCE_FUNCTION,
                    "embedding_dim": CLIP_EMBEDDING_DIM,
                    "description": "CLIP embeddings for semantic similarity",
                },
            )
            
            # Create or get DINO collection
            dino_name = f"{self._collection_prefix}{DINO_COLLECTION_NAME}"
            self._dino_collection = self._client.get_or_create_collection(
                name=dino_name,
                metadata={
                    "hnsw:space": DISTANCE_FUNCTION,
                    "embedding_dim": DINO_EMBEDDING_DIM,
                    "description": "DINO embeddings for structural similarity",
                },
            )
            
            logger.info(
                f"ChromaDB initialized - CLIP collection: {self._clip_collection.count()} items, "
                f"DINO collection: {self._dino_collection.count()} items"
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise DatabaseError(f"ChromaDB initialization failed: {e}")
    
    @property
    def clip_collection(self) -> chromadb.Collection:
        """Return the CLIP embeddings collection."""
        return self._clip_collection
    
    @property
    def dino_collection(self) -> chromadb.Collection:
        """Return the DINO embeddings collection."""
        return self._dino_collection
    
    # =========================================
    # Item Management (Tâche 4.4)
    # =========================================
    
    def add_item(
        self,
        item: ClothingItem,
        clip_embedding: List[float],
        dino_embedding: List[float],
    ) -> str:
        """
        Add a clothing item with its embeddings to both collections.
        
        Stores the item metadata and embeddings:
        - CLIP embedding (512d) in clip_collection
        - DINO embedding (384d) in dino_collection
        
        Args:
            item: The clothing item entity with metadata.
            clip_embedding: CLIP embedding vector (512 dimensions).
            dino_embedding: DINOv2 embedding vector (384 dimensions).
            
        Returns:
            The ID of the stored item.
            
        Raises:
            EmbeddingMismatchError: If embedding dimensions are wrong.
            DatabaseError: If storage fails.
            
        Example:
            >>> item = ClothingItem(id="123", title="Jacket", price=50.0)
            >>> clip_emb = encoder.encode_clip(image)  # 512d
            >>> dino_emb = encoder.encode_dino(image)  # 384d
            >>> item_id = repo.add_item(item, clip_emb, dino_emb)
        """
        # Validate CLIP embedding dimension
        if len(clip_embedding) != CLIP_EMBEDDING_DIM:
            raise EmbeddingMismatchError(
                f"CLIP embedding dimension mismatch: expected {CLIP_EMBEDDING_DIM}, got {len(clip_embedding)}",
                expected=CLIP_EMBEDDING_DIM,
                actual=len(clip_embedding),
            )
        
        # Validate DINO embedding dimension
        if len(dino_embedding) != DINO_EMBEDDING_DIM:
            raise EmbeddingMismatchError(
                f"DINO embedding dimension mismatch: expected {DINO_EMBEDDING_DIM}, got {len(dino_embedding)}",
                expected=DINO_EMBEDDING_DIM,
                actual=len(dino_embedding),
            )
        
        # Convert item to metadata (None values -> "")
        metadata = _item_to_metadata(item)
        
        # Document is used for full-text search (optional in ChromaDB)
        document = item.title or item.id
        
        logger.debug(f"Adding item {item.id} to ChromaDB collections...")
        
        try:
            # Add to CLIP collection
            self._clip_collection.upsert(
                ids=[item.id],
                embeddings=[clip_embedding],
                metadatas=[metadata],
                documents=[document],
            )
            logger.debug(f"Item {item.id} added to CLIP collection")
            
            # Add to DINO collection
            self._dino_collection.upsert(
                ids=[item.id],
                embeddings=[dino_embedding],
                metadatas=[metadata],
                documents=[document],
            )
            logger.debug(f"Item {item.id} added to DINO collection")
            
            logger.info(
                f"Item '{item.id}' ajouté aux 2 collections "
                f"(CLIP: {CLIP_EMBEDDING_DIM}d, DINO: {DINO_EMBEDDING_DIM}d)"
            )
            
            return item.id
            
        except Exception as e:
            logger.error(f"Failed to add item {item.id}: {e}")
            raise DatabaseError(f"Failed to add item to ChromaDB: {e}")
    
    def add_items_batch(
        self,
        items: List[ClothingItem],
        clip_embeddings: List[List[float]],
        dino_embeddings: List[List[float]],
    ) -> List[str]:
        """
        Add multiple items in a batch operation.
        
        More efficient than adding items one by one.
        
        Args:
            items: List of clothing items.
            clip_embeddings: List of CLIP embeddings (one per item).
            dino_embeddings: List of DINO embeddings (one per item).
            
        Returns:
            List of stored item IDs.
            
        Raises:
            ValueError: If list lengths don't match.
            EmbeddingMismatchError: If embedding dimensions are wrong.
            DatabaseError: If batch storage fails.
        """
        # Validate lengths
        if not (len(items) == len(clip_embeddings) == len(dino_embeddings)):
            raise ValueError(
                f"List lengths must match: items={len(items)}, "
                f"clip={len(clip_embeddings)}, dino={len(dino_embeddings)}"
            )
        
        if not items:
            return []
        
        # Validate dimensions
        for i, (clip_emb, dino_emb) in enumerate(zip(clip_embeddings, dino_embeddings)):
            if len(clip_emb) != CLIP_EMBEDDING_DIM:
                raise EmbeddingMismatchError(
                    f"CLIP embedding dimension mismatch at index {i}",
                    expected=CLIP_EMBEDDING_DIM,
                    actual=len(clip_emb),
                )
            if len(dino_emb) != DINO_EMBEDDING_DIM:
                raise EmbeddingMismatchError(
                    f"DINO embedding dimension mismatch at index {i}",
                    expected=DINO_EMBEDDING_DIM,
                    actual=len(dino_emb),
                )
        
        # Prepare batch data
        ids = [item.id for item in items]
        metadatas = [_item_to_metadata(item) for item in items]
        documents = [item.title or item.id for item in items]
        
        logger.debug(f"Adding {len(items)} items to ChromaDB in batch...")
        
        try:
            # Batch add to CLIP collection
            self._clip_collection.upsert(
                ids=ids,
                embeddings=clip_embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            
            # Batch add to DINO collection
            self._dino_collection.upsert(
                ids=ids,
                embeddings=dino_embeddings,
                metadatas=metadatas,
                documents=documents,
            )
            
            logger.info(f"{len(items)} items ajoutés aux 2 collections en batch")
            return ids
            
        except Exception as e:
            logger.error(f"Batch add failed: {e}")
            raise DatabaseError(f"Batch add failed: {e}")
    
    def get_item(self, item_id: str) -> Optional[ClothingItem]:
        """Retrieve a clothing item by ID."""
        try:
            result = self._clip_collection.get(
                ids=[item_id],
                include=["metadatas"],
            )
            
            if not result["ids"]:
                return None
            
            metadata = result["metadatas"][0]
            return _metadata_to_item(metadata)
            
        except Exception as e:
            logger.error(f"Failed to get item {item_id}: {e}")
            return None
    
    def delete_item(self, item_id: str) -> bool:
        """Delete an item by ID from both collections."""
        try:
            # Check if exists
            if not self.item_exists(item_id):
                return False
            
            # Delete from both collections
            self._clip_collection.delete(ids=[item_id])
            self._dino_collection.delete(ids=[item_id])
            
            logger.info(f"Item {item_id} supprimé des 2 collections")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete item {item_id}: {e}")
            return False
    
    def item_exists(self, item_id: str) -> bool:
        """Check if an item exists in the repository."""
        try:
            result = self._clip_collection.get(
                ids=[item_id],
                include=[],
            )
            return len(result["ids"]) > 0
        except Exception:
            return False
    
    # =========================================
    # Search Operations
    # =========================================
    
    def search_similar(
        self,
        query_embedding: List[float],
        embedding_type: str,
        n_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[ClothingItem, float]]:
        """
        Search for similar items by embedding.
        
        Args:
            query_embedding: The query embedding vector.
            embedding_type: Type of embedding ("clip" or "dino").
            n_results: Maximum number of results to return.
            filters: Optional metadata filters.
            
        Returns:
            List of (ClothingItem, similarity_score) tuples,
            sorted by similarity (highest first).
        """
        # Select collection based on embedding type
        if embedding_type.lower() == "clip":
            collection = self._clip_collection
            expected_dim = CLIP_EMBEDDING_DIM
        elif embedding_type.lower() == "dino":
            collection = self._dino_collection
            expected_dim = DINO_EMBEDDING_DIM
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}. Use 'clip' or 'dino'.")
        
        # Validate embedding dimension
        if len(query_embedding) != expected_dim:
            raise EmbeddingMismatchError(
                f"{embedding_type.upper()} query embedding dimension mismatch",
                expected=expected_dim,
                actual=len(query_embedding),
            )
        
        try:
            where_clause = _convert_filters_to_chroma(filters)
            
            result = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=where_clause,
                include=["metadatas", "distances"],
            )
            
            items_with_scores = []
            
            if result["ids"] and result["ids"][0]:
                for i, item_id in enumerate(result["ids"][0]):
                    metadata = result["metadatas"][0][i]
                    distance = result["distances"][0][i]
                    
                    # Convert distance to similarity (cosine distance to similarity)
                    similarity = 1 - distance
                    
                    item = _metadata_to_item(metadata)
                    items_with_scores.append((item, similarity))
            
            logger.debug(
                f"Search {embedding_type} returned {len(items_with_scores)} results"
            )
            return items_with_scores
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise DatabaseError(f"Search failed: {e}")
    
    def search_hybrid(
        self,
        clip_embedding: List[float],
        dino_embedding: List[float],
        clip_weight: float = 0.5,
        dino_weight: float = 0.5,
        n_results: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[ClothingItem, float]]:
        """
        Search using both CLIP and DINO embeddings with late fusion.
        
        Args:
            clip_embedding: The CLIP query embedding (512d).
            dino_embedding: The DINO query embedding (384d).
            clip_weight: Weight for CLIP similarity (0.0 to 1.0).
            dino_weight: Weight for DINO similarity (0.0 to 1.0).
            n_results: Maximum number of results to return.
            filters: Optional metadata filters.
            
        Returns:
            List of (ClothingItem, combined_score) tuples.
        """
        # Normalize weights
        total = clip_weight + dino_weight
        if total > 0:
            clip_weight = clip_weight / total
            dino_weight = dino_weight / total
        
        # Get more results from each collection for better fusion
        fetch_n = min(n_results * 3, 100)
        
        # Search both collections
        clip_results = self.search_similar(
            clip_embedding, "clip", n_results=fetch_n, filters=filters
        )
        dino_results = self.search_similar(
            dino_embedding, "dino", n_results=fetch_n, filters=filters
        )
        
        # Build score maps
        clip_scores: Dict[str, float] = {
            item.id: score for item, score in clip_results
        }
        dino_scores: Dict[str, float] = {
            item.id: score for item, score in dino_results
        }
        
        # Get all unique item IDs
        all_ids = set(clip_scores.keys()) | set(dino_scores.keys())
        
        # Build item map for metadata
        item_map: Dict[str, ClothingItem] = {}
        for item, _ in clip_results:
            item_map[item.id] = item
        for item, _ in dino_results:
            if item.id not in item_map:
                item_map[item.id] = item
        
        # Calculate combined scores (late fusion)
        combined_results = []
        for item_id in all_ids:
            clip_score = clip_scores.get(item_id, 0.0)
            dino_score = dino_scores.get(item_id, 0.0)
            
            # Weighted average
            combined_score = (clip_weight * clip_score) + (dino_weight * dino_score)
            
            item = item_map[item_id]
            combined_results.append((item, combined_score))
        
        # Sort by combined score (highest first)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"Hybrid search returned {len(combined_results[:n_results])} results")
        
        return combined_results[:n_results]
    
    # =========================================
    # Collection Management
    # =========================================
    
    def count(self) -> int:
        """Return the total number of items in the repository."""
        return self._clip_collection.count()
    
    def clear(self) -> None:
        """Remove all items from the repository."""
        try:
            # Delete and recreate collections
            clip_name = self._clip_collection.name
            dino_name = self._dino_collection.name
            
            self._client.delete_collection(clip_name)
            self._client.delete_collection(dino_name)
            
            # Recreate empty collections
            self._clip_collection = self._client.get_or_create_collection(
                name=clip_name,
                metadata={
                    "hnsw:space": DISTANCE_FUNCTION,
                    "embedding_dim": CLIP_EMBEDDING_DIM,
                },
            )
            self._dino_collection = self._client.get_or_create_collection(
                name=dino_name,
                metadata={
                    "hnsw:space": DISTANCE_FUNCTION,
                    "embedding_dim": DINO_EMBEDDING_DIM,
                },
            )
            
            logger.info("Repository cleared - both collections reset")
            
        except Exception as e:
            logger.error(f"Failed to clear repository: {e}")
            raise DatabaseError(f"Failed to clear repository: {e}")
    
    def get_all_ids(self) -> List[str]:
        """Get all item IDs in the repository."""
        try:
            result = self._clip_collection.get(include=[])
            return result["ids"]
        except Exception as e:
            logger.error(f"Failed to get all IDs: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get repository statistics.
        
        Returns:
            Dictionary with repository stats.
        """
        return {
            "total_items": self.count(),
            "clip_collection_count": self._clip_collection.count(),
            "dino_collection_count": self._dino_collection.count(),
            "persist_directory": str(self._persist_directory),
            "clip_embedding_dim": CLIP_EMBEDDING_DIM,
            "dino_embedding_dim": DINO_EMBEDDING_DIM,
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ChromaRepository("
            f"items={self.count()}, "
            f"path={self._persist_directory})"
        )

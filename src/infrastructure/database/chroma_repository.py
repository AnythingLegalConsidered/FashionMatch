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
# Data Classes
# ============================================

from dataclasses import dataclass


@dataclass
class FusionWeights:
    """
    Weights for hybrid search late fusion.
    
    Controls the relative importance of CLIP (semantic) vs DINO (structural)
    embeddings in the final ranking score.
    
    Attributes:
        clip: Weight for CLIP similarity (semantic features).
        dino: Weight for DINO similarity (structural features).
        
    Example:
        >>> weights = FusionWeights(clip=0.6, dino=0.4)
        >>> # Prioritize semantic similarity
    """
    clip: float = 0.5
    dino: float = 0.5
    
    def __post_init__(self):
        """Validate and normalize weights."""
        if self.clip < 0 or self.dino < 0:
            raise ValueError("Weights must be non-negative")
        
        total = self.clip + self.dino
        if total == 0:
            raise ValueError("At least one weight must be positive")
        
        # Normalize to sum to 1.0
        self.clip = self.clip / total
        self.dino = self.dino / total
    
    @classmethod
    def from_config(cls) -> "FusionWeights":
        """Create FusionWeights from application config."""
        settings = get_settings()
        return cls(
            clip=settings.models.fusion.weights.clip,
            dino=settings.models.fusion.weights.dino,
        )
    
    def __repr__(self) -> str:
        return f"FusionWeights(clip={self.clip:.2f}, dino={self.dino:.2f})"


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
    
    def hybrid_search(
        self,
        clip_vector: List[float],
        dino_vector: List[float],
        weights: Optional[FusionWeights] = None,
        limit: int = 20,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ClothingItem]:
        """
        Search using both CLIP and DINO embeddings with late fusion.
        
        Implements a late fusion algorithm that:
        1. Queries both collections independently
        2. Converts distances to similarity scores
        3. Combines scores using weighted average
        4. Returns top results sorted by combined score
        
        Args:
            clip_vector: The CLIP query embedding (512d).
            dino_vector: The DINO query embedding (384d).
            weights: Fusion weights for CLIP and DINO. If None, uses config.
            limit: Maximum number of results to return.
            filters: Optional metadata filters.
            
        Returns:
            List of ClothingItem objects sorted by combined score (best first).
            
        Example:
            >>> weights = FusionWeights(clip=0.6, dino=0.4)
            >>> items = repo.hybrid_search(clip_emb, dino_emb, weights, limit=20)
            >>> for item in items:
            ...     print(f"{item.title}: matched!")
        """
        # Use default weights from config if not provided
        if weights is None:
            weights = FusionWeights.from_config()
        
        logger.debug(f"Hybrid search with weights: {weights}")
        
        # Validate embedding dimensions
        if len(clip_vector) != CLIP_EMBEDDING_DIM:
            raise EmbeddingMismatchError(
                f"CLIP vector dimension mismatch",
                expected=CLIP_EMBEDDING_DIM,
                actual=len(clip_vector),
            )
        if len(dino_vector) != DINO_EMBEDDING_DIM:
            raise EmbeddingMismatchError(
                f"DINO vector dimension mismatch",
                expected=DINO_EMBEDDING_DIM,
                actual=len(dino_vector),
            )
        
        # Fetch more results than needed for better fusion coverage
        # Using limit * 2 to have margin for items found by only one model
        fetch_limit = min(limit * 2, 100)
        
        try:
            # =========================================
            # Step 1: Query both collections
            # =========================================
            
            where_clause = _convert_filters_to_chroma(filters)
            
            # Query CLIP collection
            clip_results = self._clip_collection.query(
                query_embeddings=[clip_vector],
                n_results=fetch_limit,
                where=where_clause,
                include=["metadatas", "distances"],
            )
            
            # Query DINO collection
            dino_results = self._dino_collection.query(
                query_embeddings=[dino_vector],
                n_results=fetch_limit,
                where=where_clause,
                include=["metadatas", "distances"],
            )
            
            # =========================================
            # Step 2: Build score dictionaries
            # =========================================
            
            # Convert CLIP results: distance -> similarity (1 - distance for cosine)
            clip_scores: Dict[str, float] = {}
            clip_metadata: Dict[str, Dict] = {}
            
            if clip_results["ids"] and clip_results["ids"][0]:
                for i, item_id in enumerate(clip_results["ids"][0]):
                    distance = clip_results["distances"][0][i]
                    # Convert cosine distance to similarity score
                    # Cosine distance in ChromaDB: 0 = identical, 2 = opposite
                    # Similarity: 1 - distance, clamped to [0, 1]
                    similarity = max(0.0, min(1.0, 1.0 - distance))
                    clip_scores[item_id] = similarity
                    clip_metadata[item_id] = clip_results["metadatas"][0][i]
            
            # Convert DINO results: distance -> similarity
            dino_scores: Dict[str, float] = {}
            dino_metadata: Dict[str, Dict] = {}
            
            if dino_results["ids"] and dino_results["ids"][0]:
                for i, item_id in enumerate(dino_results["ids"][0]):
                    distance = dino_results["distances"][0][i]
                    similarity = max(0.0, min(1.0, 1.0 - distance))
                    dino_scores[item_id] = similarity
                    dino_metadata[item_id] = dino_results["metadatas"][0][i]
            
            logger.debug(
                f"CLIP found {len(clip_scores)} items, DINO found {len(dino_scores)} items"
            )
            
            # =========================================
            # Step 3: Fusion - Combine scores
            # =========================================
            
            # Get all unique item IDs from both result sets
            all_item_ids = set(clip_scores.keys()) | set(dino_scores.keys())
            
            if not all_item_ids:
                logger.warning("Hybrid search found no results")
                return []
            
            # Calculate average scores for fallback when item is missing from one collection
            clip_avg = sum(clip_scores.values()) / len(clip_scores) if clip_scores else 0.0
            dino_avg = sum(dino_scores.values()) / len(dino_scores) if dino_scores else 0.0
            
            # Build fusion results with combined scores
            fusion_results: List[Tuple[str, float, Dict]] = []
            
            for item_id in all_item_ids:
                # Get scores from each model (use average as fallback if missing)
                clip_score = clip_scores.get(item_id, clip_avg)
                dino_score = dino_scores.get(item_id, dino_avg)
                
                # Apply late fusion formula: weighted average
                # ScoreFinal = (ScoreClip * weights.clip) + (ScoreDino * weights.dino)
                final_score = (clip_score * weights.clip) + (dino_score * weights.dino)
                
                # Get metadata (prefer CLIP, fallback to DINO)
                metadata = clip_metadata.get(item_id) or dino_metadata.get(item_id, {})
                
                fusion_results.append((item_id, final_score, metadata))
            
            # =========================================
            # Step 4: Sort by final score (descending)
            # =========================================
            
            fusion_results.sort(key=lambda x: x[1], reverse=True)
            
            # =========================================
            # Step 5: Convert to ClothingItem objects
            # =========================================
            
            items: List[ClothingItem] = []
            
            for item_id, score, metadata in fusion_results[:limit]:
                try:
                    item = _metadata_to_item(metadata)
                    items.append(item)
                except Exception as e:
                    logger.warning(f"Failed to convert item {item_id}: {e}")
                    continue
            
            logger.info(
                f"Hybrid search completed: {len(items)} results "
                f"(CLIP weight: {weights.clip:.2f}, DINO weight: {weights.dino:.2f})"
            )
            
            return items
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise DatabaseError(f"Hybrid search failed: {e}")
    
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
        
        This is a convenience wrapper around hybrid_search that returns
        tuples with scores for backwards compatibility.
        
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
        weights = FusionWeights(clip=clip_weight, dino=dino_weight)
        
        # Use the main hybrid_search method
        items = self.hybrid_search(
            clip_vector=clip_embedding,
            dino_vector=dino_embedding,
            weights=weights,
            limit=n_results,
            filters=filters,
        )
        
        # Note: We don't have access to the scores anymore in this wrapper
        # For full score access, use hybrid_search directly
        # Here we return a placeholder score of 0.0
        return [(item, 0.0) for item in items]
    
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

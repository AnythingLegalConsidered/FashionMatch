"""ChromaDB vector store wrapper with dual-collection architecture.

This module implements a comprehensive vector store manager that maintains
separate collections for CLIP and DINOv2 embeddings while keeping metadata
synchronized.
"""

import time
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from chromadb.config import Settings
from tqdm import tqdm

from src.utils import get_logger, log_exception, log_execution_time
from src.utils.config import DatabaseConfig, FusionWeights

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
)

logger = get_logger(__name__)


class VectorStore:
    """Dual-collection vector store for fashion item embeddings."""
    
    def __init__(
        self,
        config: DatabaseConfig,
        clip_dim: int,
        dino_dim: int
    ):
        """Initialize vector store with dual collections.
        
        Args:
            config: Database configuration
            clip_dim: Expected CLIP embedding dimension
            dino_dim: Expected DINOv2 embedding dimension
        """
        self.config = config
        self.clip_dim = clip_dim
        self.dino_dim = dino_dim
        
        # Initialize ChromaDB client
        persist_dir = Path(config.persist_directory)
        persist_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"Initialized ChromaDB client at {persist_dir}")
        except Exception as e:
            raise CollectionError(f"Failed to initialize ChromaDB client: {e}") from e
        
        # Collection names
        self.clip_collection_name = f"{config.collection_name}_clip"
        self.dino_collection_name = f"{config.collection_name}_dino"
        
        # Distance metric mapping
        distance_metric_map = {
            'cosine': 'cosine',
            'l2': 'l2',
            'ip': 'ip'
        }
        self.distance_metric = distance_metric_map.get(
            config.distance_metric.lower(),
            'cosine'
        )
        
        # Create or get collections
        try:
            self.clip_collection = self.client.get_or_create_collection(
                name=self.clip_collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            self.dino_collection = self.client.get_or_create_collection(
                name=self.dino_collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            clip_count = self.clip_collection.count()
            dino_count = self.dino_collection.count()
            
            logger.info(
                f"Collections initialized: {self.clip_collection_name} ({clip_count} items), "
                f"{self.dino_collection_name} ({dino_count} items)"
            )
        except Exception as e:
            raise CollectionError(f"Failed to create/access collections: {e}") from e
    
    def add_items(
        self,
        items: list[FashionItem],
        batch_size: Optional[int] = None
    ) -> BatchInsertResult:
        """Add fashion items to vector store with embeddings.
        
        Args:
            items: List of FashionItem instances with embeddings
            batch_size: Batch size for insertion (uses config default if None)
            
        Returns:
            BatchInsertResult with insertion statistics
            
        Raises:
            EmbeddingDimensionError: If embedding dimensions don't match
            BatchInsertError: If any items fail to insert
            VectorStoreError: If insertion fails
        """
        if not items:
            return BatchInsertResult(success_count=0, failed_ids=[], total_time=0.0)
        
        batch_size = batch_size or self.config.batch_size
        start_time = time.time()
        
        logger.info(f"Adding {len(items)} items to vector store (batch_size={batch_size})")
        
        # Validate all items have embeddings
        for item in items:
            if item.clip_embedding is None or item.dino_embedding is None:
                raise VectorStoreError(
                    f"Item {item.item_id} missing embeddings. "
                    "Both CLIP and DINOv2 embeddings required."
                )
            self._validate_embeddings(item.clip_embedding, item.dino_embedding)
        
        # Prepare data
        ids = [item.item_id for item in items]
        clip_embeddings = [item.clip_embedding for item in items]
        dino_embeddings = [item.dino_embedding for item in items]
        metadatas = [fashion_item_to_metadata(item) for item in items]
        
        # Insert into both collections
        with log_execution_time(logger, f"batch insert of {len(items)} items"):
            clip_failed = self._batch_insert_to_collection(
                self.clip_collection,
                ids,
                clip_embeddings,
                metadatas,
                batch_size
            )
            
            dino_failed = self._batch_insert_to_collection(
                self.dino_collection,
                ids,
                dino_embeddings,
                metadatas,
                batch_size
            )
            
            # Combine failed IDs
            failed_ids = list(set(clip_failed + dino_failed))
            success_count = len(items) - len(failed_ids)
        
        total_time = time.time() - start_time
        
        result = BatchInsertResult(
            success_count=success_count,
            failed_ids=failed_ids,
            total_time=total_time
        )
        
        logger.info(
            f"Batch insert complete: {success_count}/{len(items)} successful, "
            f"{len(failed_ids)} failed, {total_time:.2f}s"
        )
        
        # Raise BatchInsertError if any insertions failed
        if failed_ids:
            raise BatchInsertError(
                f"Failed to insert {len(failed_ids)}/{len(items)} items",
                failed_ids=failed_ids
            )
        
        return result
    
    def search(
        self,
        clip_query: np.ndarray,
        dino_query: np.ndarray,
        top_k: int = 10,
        fusion_weights: Optional[FusionWeights] = None
    ) -> list[SearchResult]:
        """Search for similar items using dual embeddings.
        
        Args:
            clip_query: CLIP query embedding
            dino_query: DINOv2 query embedding
            top_k: Number of results to return
            fusion_weights: Fusion weights (uses default if None)
            
        Returns:
            List of SearchResult objects ranked by similarity
            
        Raises:
            EmbeddingDimensionError: If query dimensions don't match
        """
        # Validate query dimensions
        self._validate_embeddings(clip_query, dino_query)
        
        logger.debug(f"Searching for top {top_k} similar items")
        
        with log_execution_time(logger, f"dual search with top_k={top_k}"):
            # Query CLIP collection
            clip_results = self.clip_collection.query(
                query_embeddings=[clip_query.tolist()],
                n_results=top_k,
                include=['metadatas', 'distances', 'embeddings']
            )
            
            # Query DINOv2 collection
            dino_results = self.dino_collection.query(
                query_embeddings=[dino_query.tolist()],
                n_results=top_k,
                include=['metadatas', 'distances', 'embeddings']
            )
            
            # Merge and rank results
            results = self._merge_search_results(
                clip_results,
                dino_results,
                fusion_weights
            )
        
        logger.debug(f"Search complete: found {len(results)} results")
        return results[:top_k]
    
    def get_by_ids(self, item_ids: list[str]) -> list[FashionItem]:
        """Retrieve fashion items by IDs.
        
        Args:
            item_ids: List of item IDs to retrieve
            
        Returns:
            List of FashionItem objects (only items that exist; missing IDs are filtered out)
            
        Note:
            Missing IDs are logged as warnings but do not raise errors.
            Returns empty list if no IDs match existing items.
        """
        if not item_ids:
            return []
        
        logger.debug(f"Retrieving {len(item_ids)} items by ID")
        
        try:
            # Get from CLIP collection (source of truth for metadata)
            clip_data = self.clip_collection.get(
                ids=item_ids,
                include=['metadatas', 'embeddings']
            )
            
            # Get from DINOv2 collection
            dino_data = self.dino_collection.get(
                ids=item_ids,
                include=['embeddings']
            )
            
            # Create index lookup dicts for O(1) access
            clip_idx_map = {clip_id: i for i, clip_id in enumerate(clip_data['ids'])}
            dino_idx_map = {dino_id: i for i, dino_id in enumerate(dino_data['ids'])}
            
            # Build result list, filtering out missing items
            results = []
            skipped_count = 0
            
            for item_id in item_ids:
                # Check if item exists in CLIP collection
                if item_id not in clip_idx_map:
                    logger.warning(f"Skipping missing item: {item_id} (not in CLIP collection)")
                    skipped_count += 1
                    continue
                
                # Get CLIP data
                clip_idx = clip_idx_map[item_id]
                metadata = clip_data['metadatas'][clip_idx]
                clip_emb = np.array(clip_data['embeddings'][clip_idx], dtype=np.float32)
                
                # Check if DINOv2 embedding exists
                if item_id not in dino_idx_map:
                    logger.warning(f"Skipping item {item_id}: DINO embedding missing")
                    skipped_count += 1
                    continue
                
                # Get DINOv2 data
                dino_idx = dino_idx_map[item_id]
                dino_emb = np.array(dino_data['embeddings'][dino_idx], dtype=np.float32)
                
                # Construct FashionItem
                item = metadata_to_fashion_item(item_id, metadata, clip_emb, dino_emb)
                results.append(item)
            
            # Log summary
            if skipped_count > 0:
                logger.info(
                    f"Retrieved {len(results)}/{len(item_ids)} items, "
                    f"skipped {skipped_count} missing"
                )
            else:
                logger.debug(f"Retrieved all {len(results)} requested items")
            
            return results
        
        except Exception as e:
            log_exception(logger, f"get items by IDs", e)
            raise VectorStoreError(f"Failed to retrieve items: {e}") from e
    
    def update_item(self, item_id: str, item: FashionItem) -> bool:
        """Update an existing item in the vector store.
        
        Args:
            item_id: ID of item to update
            item: Updated FashionItem
            
        Returns:
            True if successful
            
        Raises:
            ItemNotFoundError: If item is not found
        """
        logger.debug(f"Updating item {item_id}")
        
        try:
            # Check if item exists
            existing = self.clip_collection.get(ids=[item_id])
            if not existing['ids']:
                logger.error(f"Item {item_id} not found for update")
                raise ItemNotFoundError(f"Item {item_id} not found", item_id=item_id)
            
            # Prepare update data
            metadata = fashion_item_to_metadata(item)
            
            # Update CLIP collection (metadata + embedding if provided)
            if item.clip_embedding is not None:
                self.clip_collection.update(
                    ids=[item_id],
                    embeddings=[item.clip_embedding.tolist()],
                    metadatas=[metadata]
                )
            else:
                self.clip_collection.update(
                    ids=[item_id],
                    metadatas=[metadata]
                )
            
            # Update DINOv2 collection (embedding only if provided)
            if item.dino_embedding is not None:
                self.dino_collection.update(
                    ids=[item_id],
                    embeddings=[item.dino_embedding.tolist()],
                    metadatas=[metadata]
                )
            
            logger.info(f"Updated item {item_id}")
            return True
        
        except Exception as e:
            log_exception(logger, f"update item {item_id}", e)
            raise VectorStoreError(f"Failed to update item: {e}") from e
    
    def delete_items(self, item_ids: list[str]) -> int:
        """Delete items from vector store.
        
        Args:
            item_ids: List of item IDs to delete
            
        Returns:
            Number of items confirmed deleted from both collections
        """
        if not item_ids:
            return 0
        
        logger.info(f"Deleting {len(item_ids)} items")
        
        try:
            # Get initial counts
            clip_count_before = self.clip_collection.count()
            dino_count_before = self.dino_collection.count()
            
            # Delete from CLIP collection
            self.clip_collection.delete(ids=item_ids)
            
            # Delete from DINOv2 collection
            self.dino_collection.delete(ids=item_ids)
            
            # Get final counts to confirm deletions
            clip_count_after = self.clip_collection.count()
            dino_count_after = self.dino_collection.count()
            
            clip_deleted = clip_count_before - clip_count_after
            dino_deleted = dino_count_before - dino_count_after
            
            # Return minimum to ensure both collections deleted successfully
            deleted_count = min(clip_deleted, dino_deleted)
            
            logger.info(f"Deleted {deleted_count} items (CLIP: {clip_deleted}, DINO: {dino_deleted})")
            return deleted_count
        
        except Exception as e:
            log_exception(logger, f"delete {len(item_ids)} items", e)
            raise VectorStoreError(f"Failed to delete items: {e}") from e
    
    def count(self) -> int:
        """Get total number of items in vector store.
        
        Returns:
            Item count from CLIP collection (source of truth)
        """
        return self.clip_collection.count()
    
    def clear(self) -> None:
        """Clear all items from vector store.
        
        WARNING: This deletes all data from both collections!
        """
        logger.warning("Clearing all data from vector store")
        
        try:
            # Delete collections
            self.client.delete_collection(self.clip_collection_name)
            self.client.delete_collection(self.dino_collection_name)
            
            # Recreate empty collections
            self.clip_collection = self.client.create_collection(
                name=self.clip_collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            self.dino_collection = self.client.create_collection(
                name=self.dino_collection_name,
                metadata={"hnsw:space": self.distance_metric}
            )
            
            logger.info("Vector store cleared and recreated")
        
        except Exception as e:
            log_exception(logger, "clear vector store", e)
            raise VectorStoreError(f"Failed to clear vector store: {e}") from e
    
    def get_all_ids(self) -> list[str]:
        """Get all item IDs from vector store.
        
        Returns:
            List of all item IDs
        """
        try:
            # Get all IDs from CLIP collection
            data = self.clip_collection.get(include=[])
            return data['ids']
        except Exception as e:
            log_exception(logger, "get all IDs", e)
            raise VectorStoreError(f"Failed to retrieve IDs: {e}") from e
    
    def _validate_embeddings(
        self,
        clip_emb: np.ndarray,
        dino_emb: np.ndarray
    ) -> None:
        """Validate embedding dimensions and dtypes.
        
        Args:
            clip_emb: CLIP embedding
            dino_emb: DINOv2 embedding
            
        Raises:
            EmbeddingDimensionError: If dimensions or dtypes don't match
        """
        # Check CLIP embedding
        if clip_emb.shape != (self.clip_dim,):
            raise EmbeddingDimensionError(
                f"CLIP embedding shape mismatch: expected ({self.clip_dim},), got {clip_emb.shape}",
                expected_dim=self.clip_dim,
                actual_dim=clip_emb.shape[0] if clip_emb.ndim == 1 else None
            )
        
        # Check DINOv2 embedding
        if dino_emb.shape != (self.dino_dim,):
            raise EmbeddingDimensionError(
                f"DINOv2 embedding shape mismatch: expected ({self.dino_dim},), got {dino_emb.shape}",
                expected_dim=self.dino_dim,
                actual_dim=dino_emb.shape[0] if dino_emb.ndim == 1 else None
            )
        
        # Check dtypes
        if clip_emb.dtype != np.float32:
            raise EmbeddingDimensionError(
                f"CLIP embedding dtype mismatch: expected float32, got {clip_emb.dtype}"
            )
        
        if dino_emb.dtype != np.float32:
            raise EmbeddingDimensionError(
                f"DINOv2 embedding dtype mismatch: expected float32, got {dino_emb.dtype}"
            )
    
    def _batch_insert_to_collection(
        self,
        collection,
        ids: list[str],
        embeddings: list[np.ndarray],
        metadatas: list[dict],
        batch_size: int
    ) -> list[str]:
        """Insert items into collection in batches.
        
        Args:
            collection: ChromaDB collection
            ids: List of item IDs
            embeddings: List of embeddings
            metadatas: List of metadata dicts
            batch_size: Batch size for insertion
            
        Returns:
            List of failed item IDs
        """
        failed_ids = []
        num_batches = (len(ids) + batch_size - 1) // batch_size
        
        # Use progress bar if multiple batches
        iterator = range(0, len(ids), batch_size)
        if num_batches > 1:
            iterator = tqdm(iterator, total=num_batches, desc=f"Inserting to {collection.name}")
        
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, len(ids))
            
            batch_ids = ids[start_idx:end_idx]
            batch_embeddings = [emb.tolist() for emb in embeddings[start_idx:end_idx]]
            batch_metadatas = metadatas[start_idx:end_idx]
            
            try:
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas
                )
            except Exception as e:
                logger.error(f"Failed to insert batch {start_idx}-{end_idx}: {e}")
                failed_ids.extend(batch_ids)
        
        return failed_ids
    
    def _merge_search_results(
        self,
        clip_results: dict,
        dino_results: dict,
        fusion_weights: Optional[FusionWeights]
    ) -> list[SearchResult]:
        """Merge and rank search results from both collections.
        
        Args:
            clip_results: Results from CLIP collection query
            dino_results: Results from DINOv2 collection query
            fusion_weights: Fusion weights for scoring
            
        Returns:
            List of SearchResult objects ranked by fused score
        """
        # Use default fusion weights if not provided
        if fusion_weights is None:
            # Default to equal weights
            alpha = 0.5
            beta = 0.5
        else:
            alpha = fusion_weights.clip
            beta = fusion_weights.dino
        
        # Extract results (ChromaDB returns nested lists)
        clip_ids = clip_results['ids'][0] if clip_results['ids'] else []
        clip_distances = clip_results['distances'][0] if clip_results['distances'] else []
        clip_metadatas = clip_results['metadatas'][0] if clip_results['metadatas'] else []
        clip_embeddings = clip_results['embeddings'][0] if clip_results['embeddings'] else []
        
        dino_ids = dino_results['ids'][0] if dino_results['ids'] else []
        dino_distances = dino_results['distances'][0] if dino_results['distances'] else []
        dino_embeddings = dino_results['embeddings'][0] if dino_results['embeddings'] else []
        
        # Build result map by item_id with explicit presence tracking
        result_map = {}
        
        # Process CLIP results
        for i, item_id in enumerate(clip_ids):
            clip_dist = clip_distances[i]
            clip_score = self._distance_to_similarity(clip_dist)
            
            result_map[item_id] = {
                'has_clip': True,
                'has_dino': False,
                'clip_score': clip_score,
                'clip_distance': clip_dist,
                'clip_embedding': np.array(clip_embeddings[i], dtype=np.float32) if i < len(clip_embeddings) else None,
                'metadata': clip_metadatas[i] if i < len(clip_metadatas) else {},
                'dino_score': 0.0,
                'dino_embedding': None,
            }
        
        # Process DINOv2 results
        for i, item_id in enumerate(dino_ids):
            dino_dist = dino_distances[i]
            dino_score = self._distance_to_similarity(dino_dist)
            
            if item_id in result_map:
                # Item in both collections
                result_map[item_id]['has_dino'] = True
                result_map[item_id]['dino_score'] = dino_score
                result_map[item_id]['dino_embedding'] = np.array(dino_embeddings[i], dtype=np.float32) if i < len(dino_embeddings) else None
            else:
                # Item only in DINOv2 results - fetch metadata from CLIP collection
                logger.warning(f"Item {item_id} found in DINO but not CLIP results, fetching metadata")
                try:
                    clip_data = self.clip_collection.get(
                        ids=[item_id],
                        include=['metadatas', 'embeddings']
                    )
                    if clip_data['ids']:
                        metadata = clip_data['metadatas'][0]
                        clip_emb = np.array(clip_data['embeddings'][0], dtype=np.float32) if clip_data['embeddings'] else None
                        result_map[item_id] = {
                            'has_clip': True,
                            'has_dino': True,
                            'clip_score': 0.0,  # Not in search results, assign zero
                            'clip_distance': float('inf'),
                            'clip_embedding': clip_emb,
                            'metadata': metadata,
                            'dino_score': dino_score,
                            'dino_embedding': np.array(dino_embeddings[i], dtype=np.float32) if i < len(dino_embeddings) else None,
                        }
                    else:
                        logger.warning(f"Item {item_id} not found in CLIP collection, collections are out of sync, skipping")
                        continue
                except Exception as e:
                    logger.error(f"Failed to fetch metadata for {item_id}: {e}, skipping")
                    continue
        
        # Compute fused scores and create SearchResult objects
        search_results = []
        
        for item_id, data in result_map.items():
            # Enforce strict dual-presence: both embeddings must exist
            if not (data['has_clip'] and data['has_dino']):
                logger.warning(f"Item {item_id} missing from one collection (CLIP: {data['has_clip']}, DINO: {data['has_dino']}), skipping")
                continue
            
            # Verify embeddings are not None
            if data['clip_embedding'] is None or data['dino_embedding'] is None:
                logger.warning(f"Item {item_id} has None embedding (CLIP: {data['clip_embedding'] is not None}, DINO: {data['dino_embedding'] is not None}), skipping")
                continue
            
            # Compute fused score
            fused_score = alpha * data['clip_score'] + beta * data['dino_score']
            
            # Reconstruct FashionItem
            fashion_item = metadata_to_fashion_item(
                item_id,
                data['metadata'],
                data['clip_embedding'],
                data['dino_embedding']
            )
            
            # Create SearchResult
            result = SearchResult(
                item_id=item_id,
                similarity_score=fused_score,
                clip_score=data['clip_score'],
                dino_score=data['dino_score'],
                item=fashion_item
            )
            
            search_results.append(result)
        
        # Sort by fused score descending
        search_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return search_results
    
    def _distance_to_similarity(self, distance: float) -> float:
        """Convert distance to similarity score.
        
        Args:
            distance: Distance value from ChromaDB
            
        Returns:
            Similarity score (higher is better)
        """
        if self.distance_metric == 'cosine':
            # Cosine distance in [0, 2], convert to similarity in [0, 1]
            return 1.0 - distance
        elif self.distance_metric == 'l2':
            # L2 distance, convert to similarity (negative distance)
            return -distance
        elif self.distance_metric == 'ip':
            # Inner product (already a similarity)
            return distance
        else:
            return -distance


# Singleton cache for vector store instances
_vector_store_cache: dict[str, VectorStore] = {}


def get_vector_store(
    config: DatabaseConfig,
    clip_dim: int,
    dino_dim: int
) -> VectorStore:
    """Get or create a VectorStore instance (singleton pattern).
    
    Args:
        config: Database configuration
        clip_dim: CLIP embedding dimension
        dino_dim: DINOv2 embedding dimension
        
    Returns:
        Cached or newly created VectorStore instance
    """
    # Create cache key
    cache_key = f"{config.persist_directory}_{config.collection_name}_{clip_dim}_{dino_dim}"
    
    if cache_key not in _vector_store_cache:
        logger.debug(f"Creating new VectorStore for {cache_key}")
        _vector_store_cache[cache_key] = VectorStore(config, clip_dim, dino_dim)
    else:
        logger.debug(f"Reusing cached VectorStore for {cache_key}")
    
    return _vector_store_cache[cache_key]

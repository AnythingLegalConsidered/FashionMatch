# Get Recommendations Use Case
"""
Use case for retrieving personalized clothing recommendations.

Computes user preference vectors from reference images and performs
hybrid search to find matching items.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import logging

from src.domain.entities.clothing_item import ClothingItem
from src.infrastructure.database.chroma_repository import (
    ChromaRepository,
    FusionWeights,
)

logger = logging.getLogger(__name__)


@dataclass
class RecommendationResult:
    """Result of a recommendation query."""
    items: List[Tuple[ClothingItem, float]]  # (item, score) pairs
    clip_vector_used: bool = False
    dino_vector_used: bool = False
    reference_count: int = 0
    total_items_searched: int = 0
    message: str = ""


@dataclass
class RecommendationFilters:
    """Filters for recommendations."""
    category: Optional[str] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    brand: Optional[str] = None
    size: Optional[str] = None
    ignored_ids: List[str] = field(default_factory=list)


class GetRecommendationsUseCase:
    """
    Use case for getting personalized recommendations.
    
    This use case:
    1. Retrieves all user reference embeddings
    2. Computes mean preference vectors (CLIP and DINO)
    3. Performs hybrid search with late fusion
    4. Returns ranked items with similarity scores
    """
    
    def __init__(self, repository: ChromaRepository):
        """
        Initialize the use case.
        
        Args:
            repository: ChromaDB repository for data access.
        """
        self.repository = repository
    
    def execute(
        self,
        limit: int = 20,
        weights: Optional[FusionWeights] = None,
        filters: Optional[RecommendationFilters] = None,
    ) -> RecommendationResult:
        """
        Get personalized recommendations based on user references.
        
        Args:
            limit: Maximum number of items to return.
            weights: Fusion weights for CLIP/DINO balance.
            filters: Optional filters to apply.
            
        Returns:
            RecommendationResult with ranked items and metadata.
        """
        # Default weights
        if weights is None:
            weights = FusionWeights(clip=0.5, dino=0.5)
        
        # Default filters
        if filters is None:
            filters = RecommendationFilters()
        
        # Step 1: Get reference embeddings
        logger.info("Fetching user reference embeddings...")
        clip_embeddings, dino_embeddings = self.repository.get_reference_embeddings()
        
        reference_count = len(clip_embeddings)
        
        if reference_count == 0:
            logger.warning("No reference images found")
            return RecommendationResult(
                items=[],
                reference_count=0,
                message="Aucune image de référence trouvée. Veuillez d'abord uploader des photos."
            )
        
        logger.info(f"Found {reference_count} reference images")
        
        # Step 2: Compute mean preference vectors
        clip_mean_vector = self._compute_mean_vector(clip_embeddings)
        dino_mean_vector = self._compute_mean_vector(dino_embeddings)
        
        clip_vector_used = clip_mean_vector is not None
        dino_vector_used = dino_mean_vector is not None
        
        if not clip_vector_used and not dino_vector_used:
            return RecommendationResult(
                items=[],
                reference_count=reference_count,
                message="Impossible de calculer les vecteurs de préférence."
            )
        
        logger.info(
            f"Computed mean vectors - CLIP: {clip_vector_used}, DINO: {dino_vector_used}"
        )
        
        # Step 3: Perform hybrid search
        logger.info(f"Performing hybrid search with weights: {weights}")
        
        try:
            search_results = self.repository.hybrid_search(
                clip_query=clip_mean_vector.tolist() if clip_vector_used else None,
                dino_query=dino_mean_vector.tolist() if dino_vector_used else None,
                weights=weights,
                limit=limit + len(filters.ignored_ids),  # Request extra to account for filtering
                filters=self._build_chroma_filters(filters),
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return RecommendationResult(
                items=[],
                reference_count=reference_count,
                message=f"Erreur lors de la recherche: {str(e)}"
            )
        
        # Step 4: Filter out ignored items
        filtered_results = [
            (item, score) for item, score in search_results
            if item.id not in filters.ignored_ids
        ][:limit]
        
        # Step 5: Apply price filters (if not handled by ChromaDB)
        if filters.min_price is not None or filters.max_price is not None:
            filtered_results = self._apply_price_filter(
                filtered_results,
                filters.min_price,
                filters.max_price
            )
        
        total_items = self.repository.count()
        
        logger.info(f"Returning {len(filtered_results)} recommendations")
        
        return RecommendationResult(
            items=filtered_results,
            clip_vector_used=clip_vector_used,
            dino_vector_used=dino_vector_used,
            reference_count=reference_count,
            total_items_searched=total_items,
            message=f"Trouvé {len(filtered_results)} articles correspondant à votre style."
        )
    
    def _compute_mean_vector(
        self,
        embeddings: List[List[float]]
    ) -> Optional[np.ndarray]:
        """
        Compute the mean vector from a list of embeddings.
        
        Args:
            embeddings: List of embedding vectors.
            
        Returns:
            Mean vector as numpy array, or None if empty.
        """
        if not embeddings:
            return None
        
        # Convert to numpy and compute mean
        embeddings_array = np.array(embeddings)
        mean_vector = np.mean(embeddings_array, axis=0)
        
        # Normalize the mean vector (for cosine similarity)
        norm = np.linalg.norm(mean_vector)
        if norm > 0:
            mean_vector = mean_vector / norm
        
        return mean_vector
    
    def _build_chroma_filters(
        self,
        filters: RecommendationFilters
    ) -> Optional[dict]:
        """Build ChromaDB where clause from filters."""
        where_clause = {}
        
        if filters.category:
            where_clause["category"] = filters.category
        
        if filters.brand:
            where_clause["brand"] = filters.brand
        
        if filters.size:
            where_clause["size"] = filters.size
        
        return where_clause if where_clause else None
    
    def _apply_price_filter(
        self,
        results: List[Tuple[ClothingItem, float]],
        min_price: Optional[float],
        max_price: Optional[float]
    ) -> List[Tuple[ClothingItem, float]]:
        """Apply price range filter to results."""
        filtered = []
        
        for item, score in results:
            price = item.price or 0
            
            if min_price is not None and price < min_price:
                continue
            
            if max_price is not None and price > max_price:
                continue
            
            filtered.append((item, score))
        
        return filtered


class AddItemToReferencesUseCase:
    """
    Use case for adding a liked item to user references.
    
    This allows the recommendation system to learn from user feedback
    by treating liked items as additional reference images.
    """
    
    def __init__(self, repository: ChromaRepository):
        """
        Initialize the use case.
        
        Args:
            repository: ChromaDB repository.
        """
        self.repository = repository
    
    def execute(self, item_id: str) -> bool:
        """
        Add an item's embeddings to user references.
        
        Args:
            item_id: The ID of the liked item.
            
        Returns:
            True if successful, False otherwise.
        """
        try:
            # Get the item's embeddings from the main collections
            clip_result = self.repository.clip_collection.get(
                ids=[item_id],
                include=["embeddings", "metadatas"]
            )
            
            dino_result = self.repository.dino_collection.get(
                ids=[item_id],
                include=["embeddings"]
            )
            
            if not clip_result["ids"] or not dino_result["ids"]:
                logger.warning(f"Item {item_id} not found in database")
                return False
            
            clip_embedding = clip_result["embeddings"][0]
            dino_embedding = dino_result["embeddings"][0]
            metadata = clip_result["metadatas"][0]
            
            # Create reference metadata
            ref_metadata = {
                "type": "liked_item",
                "original_id": item_id,
                "name": metadata.get("title", f"Liked: {item_id}"),
                "category": metadata.get("category", ""),
                "tags": "liked,feedback",
            }
            
            # Add to references with a unique ID
            reference_id = f"liked_{item_id}"
            
            self.repository.add_reference(
                reference_id=reference_id,
                clip_embedding=clip_embedding,
                dino_embedding=dino_embedding,
                metadata=ref_metadata,
            )
            
            logger.info(f"Added item {item_id} to user references")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add item to references: {e}")
            return False

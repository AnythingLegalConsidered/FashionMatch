"""
Abstract interface for data repositories (vector storage).
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.domain.entities.clothing_item import ClothingItem


class RepositoryInterface(ABC):
    """
    Abstract base class for data repositories.
    
    Defines the contract for storing and retrieving clothing items
    with their embeddings.
    """

    @abstractmethod
    def add_item(
        self,
        item: ClothingItem,
        clip_embedding: List[float],
        dino_embedding: List[float],
    ) -> str:
        """
        Add a clothing item with its embeddings.
        
        Args:
            item: The clothing item entity.
            clip_embedding: CLIP embedding vector.
            dino_embedding: DINOv2 embedding vector.
            
        Returns:
            The ID of the stored item.
        """
        pass

    @abstractmethod
    def get_item(self, item_id: str) -> Optional[ClothingItem]:
        """
        Retrieve a clothing item by ID.
        
        Args:
            item_id: The unique identifier of the item.
            
        Returns:
            The ClothingItem if found, None otherwise.
        """
        pass

    @abstractmethod
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
            List of (ClothingItem, similarity_score) tuples.
        """
        pass

    @abstractmethod
    def delete_item(self, item_id: str) -> bool:
        """
        Delete an item by ID.
        
        Args:
            item_id: The unique identifier of the item.
            
        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the total number of items in the repository."""
        pass

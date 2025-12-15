"""Unit tests for vector store operations."""

import numpy as np
import pytest

from src.database.exceptions import (
    BatchInsertError,
    EmbeddingDimensionError,
    ItemNotFoundError,
)
from src.database.vector_store import VectorStore
from src.utils.config import FusionWeights


class TestVectorStoreInitialization:
    """Test vector store initialization."""
    
    def test_create_vector_store(self, temp_data_dir, test_config):
        """Test creating new vector store."""
        # Update config to use temp directory
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        assert store.clip_collection is not None
        assert store.dino_collection is not None
    
    def test_in_memory_store(self, test_config):
        """Test in-memory vector store."""
        # Use in-memory config
        test_config.database.persist_directory = ":memory:"
        test_config.database.collection_name = "test_memory"
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        assert store.clip_collection is not None


class TestVectorStoreCRUD:
    """Test CRUD operations."""
    
    def test_add_single_item(self, temp_data_dir, test_config, mock_fashion_item):
        """Test adding single item."""
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        test_config.database.collection_name = "test_add"
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        result = store.add_items([mock_fashion_item])
        
        assert result.total_items == 1
        assert result.successful == 1
        assert result.failed == 0
    
    def test_add_batch(self, temp_data_dir, test_config, mock_fashion_items):
        """Test adding batch of items."""
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        test_config.database.collection_name = "test_batch"
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        result = store.add_items(mock_fashion_items, batch_size=5)
        
        assert result.successful == len(mock_fashion_items)
        assert result.failed == 0
    
    def test_get_by_ids(self, mock_vector_store, mock_fashion_items):
        """Test retrieving items by IDs."""
        item_ids = [item.item_id for item in mock_fashion_items[:3]]
        
        results = mock_vector_store.get_by_ids(item_ids)
        
        assert len(results) == 3
        for result in results:
            assert result.item_id in item_ids
    
    def test_get_by_ids_mixed(self, mock_vector_store, mock_fashion_items):
        """Test retrieving mix of existing and nonexistent IDs."""
        existing_ids = [item.item_id for item in mock_fashion_items[:2]]
        mixed_ids = existing_ids + ["nonexistent_1", "nonexistent_2"]
        
        results = mock_vector_store.get_by_ids(mixed_ids)
        
        # Should return only existing items, filter out nonexistent
        assert len(results) == 2
        result_ids = [r.item_id for r in results]
        assert all(rid in existing_ids for rid in result_ids)
    
    def test_update_item(self, mock_vector_store, mock_fashion_item):
        """Test updating existing item."""
        # Add item first
        mock_vector_store.add_items([mock_fashion_item])
        
        # Update price
        mock_fashion_item.price = 39.99
        mock_vector_store.update_item(mock_fashion_item)
        
        # Retrieve and verify
        updated = mock_vector_store.get_by_ids([mock_fashion_item.item_id])[0]
        assert updated.price == 39.99
    
    def test_delete_items(self, mock_vector_store, mock_fashion_items):
        """Test deleting items."""
        item_ids = [item.item_id for item in mock_fashion_items[:2]]
        
        deleted_count = mock_vector_store.delete_items(item_ids)
        
        assert deleted_count == 2
        
        # Verify items are gone
        remaining = mock_vector_store.get_by_ids(item_ids)
        assert len(remaining) == 0


class TestVectorStoreSearch:
    """Test search functionality."""
    
    def test_basic_search(self, mock_vector_store, sample_embeddings):
        """Test basic similarity search."""
        clip_query, dino_query = sample_embeddings
        
        results = mock_vector_store.search(
            clip_query=clip_query,
            dino_query=dino_query,
            top_k=5
        )
        
        assert len(results) <= 5
        for result in results:
            assert 0.0 <= result.similarity_score <= 1.0
            assert result.item is not None
    
    def test_search_with_fusion_weights(self, mock_vector_store, sample_embeddings):
        """Test search with custom fusion weights."""
        clip_query, dino_query = sample_embeddings
        
        weights = FusionWeights(clip=0.8, dino=0.2)
        
        results = mock_vector_store.search(
            clip_query=clip_query,
            dino_query=dino_query,
            top_k=5,
            fusion_weights=weights
        )
        
        assert len(results) <= 5
    
    def test_search_top_k(self, mock_vector_store, sample_embeddings):
        """Test top_k parameter."""
        clip_query, dino_query = sample_embeddings
        
        results = mock_vector_store.search(
            clip_query=clip_query,
            dino_query=dino_query,
            top_k=3
        )
        
        assert len(results) <= 3
    
    def test_search_result_ordering(self, mock_vector_store, sample_embeddings):
        """Test results are ordered by similarity."""
        clip_query, dino_query = sample_embeddings
        
        results = mock_vector_store.search(
            clip_query=clip_query,
            dino_query=dino_query,
            top_k=10
        )
        
        # Check descending order
        for i in range(len(results) - 1):
            assert results[i].similarity_score >= results[i+1].similarity_score


class TestVectorStoreValidation:
    """Test validation and error handling."""
    
    def test_dimension_mismatch_clip(self, temp_data_dir, test_config, mock_fashion_item):
        """Test error on CLIP dimension mismatch."""
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        test_config.database.collection_name = "test_dim"
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        # Create item with wrong CLIP dimension
        wrong_clip = np.random.randn(256).astype(np.float32)
        mock_fashion_item.clip_embedding = wrong_clip
        
        with pytest.raises(EmbeddingDimensionError):
            store.add_items([mock_fashion_item])
    
    def test_missing_embeddings(self, temp_data_dir, test_config, mock_fashion_item):
        """Test error on missing embeddings causes store-level error."""
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        test_config.database.collection_name = "test_missing"
        
        from src.database.exceptions import VectorStoreError
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        mock_fashion_item.clip_embedding = None
        
        with pytest.raises(VectorStoreError, match="missing embeddings"):
            store.add_items([mock_fashion_item])
    
    def test_item_not_found(self, mock_vector_store):
        """Test error when item not found."""
        results = mock_vector_store.get_by_ids(["nonexistent_id"])
        
        # Should return empty list, not raise exception
        assert len(results) == 0
    
    def test_empty_search(self, temp_data_dir, test_config, sample_embeddings):
        """Test search on empty database."""
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        test_config.database.collection_name = "test_empty"
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        clip_query, dino_query = sample_embeddings
        
        results = store.search(clip_query, dino_query, top_k=5)
        
        # Should return empty list
        assert len(results) == 0


class TestVectorStoreUtilities:
    """Test utility methods."""
    
    def test_count_items(self, mock_vector_store):
        """Test counting items in database."""
        count = mock_vector_store.count()
        
        assert count > 0
    
    def test_list_all_ids(self, mock_vector_store):
        """Test listing all item IDs."""
        ids = mock_vector_store.list_ids()
        
        assert len(ids) > 0
        assert all(isinstance(id, str) for id in ids)
    
    def test_clear_database(self, temp_data_dir, test_config, mock_fashion_items):
        """Test clearing all items."""
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        test_config.database.collection_name = "test_clear"
        
        store = VectorStore(
            config=test_config.database,
            clip_dim=512,
            dino_dim=384
        )
        
        store.add_items(mock_fashion_items)
        assert store.count() > 0
        
        store.clear()
        assert store.count() == 0

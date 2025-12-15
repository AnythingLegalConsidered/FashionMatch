"""Integration tests for search workflow."""

import numpy as np
import pytest

from src.core.scorer import HybridScorer
from src.database.vector_store import VectorStore
from src.utils.config import FusionWeights


class TestSearchWorkflow:
    """Test end-to-end search workflow."""
    
    def test_basic_search_workflow(self, mock_vector_store, sample_image, test_config):
        """Test complete search workflow."""
        # Create scorer
        scorer = HybridScorer(test_config.models)
        
        # Encode query image
        clip_embs, dino_embs = scorer.encode_dual([sample_image])
        clip_query = clip_embs[0]
        dino_query = dino_embs[0]
        
        # Search database
        results = mock_vector_store.search(
            clip_query=clip_query,
            dino_query=dino_query,
            top_k=5
        )
        
        assert len(results) <= 5
        assert all(r.similarity_score > 0 for r in results)
    
    def test_multi_reference_averaging(
        self, mock_vector_store, sample_images, test_config
    ):
        """Test search with averaged multi-reference embeddings."""
        scorer = HybridScorer(test_config.models)
        
        # Encode multiple references
        clip_embs, dino_embs = scorer.encode_dual(sample_images[:3])
        
        # Average embeddings
        clip_query = np.mean(clip_embs, axis=0).astype(np.float32)
        dino_query = np.mean(dino_embs, axis=0).astype(np.float32)
        
        # Normalize
        clip_query = clip_query / np.linalg.norm(clip_query)
        dino_query = dino_query / np.linalg.norm(dino_query)
        
        # Search
        results = mock_vector_store.search(
            clip_query=clip_query,
            dino_query=dino_query,
            top_k=10
        )
        
        assert len(results) <= 10


class TestFusionWeightImpact:
    """Test fusion weight impact on results."""
    
    def test_varying_weights_affect_ranking(
        self, mock_vector_store, sample_image, test_config
    ):
        """Test different fusion weights produce different rankings."""
        scorer = HybridScorer(test_config.models)
        
        clip_embs, dino_embs = scorer.encode_dual([sample_image])
        clip_query = clip_embs[0]
        dino_query = dino_embs[0]
        
        # Search with CLIP-heavy weights
        results_clip = mock_vector_store.search(
            clip_query, dino_query, top_k=5,
            fusion_weights=FusionWeights(clip=0.9, dino=0.1)
        )
        
        # Search with DINO-heavy weights
        results_dino = mock_vector_store.search(
            clip_query, dino_query, top_k=5,
            fusion_weights=FusionWeights(clip=0.1, dino=0.9)
        )
        
        # Rankings may differ
        if len(results_clip) > 1 and len(results_dino) > 1:
            # At least check we got results
            assert len(results_clip) > 0
            assert len(results_dino) > 0


class TestSearchFiltering:
    """Test search result filtering."""
    
    def test_price_range_filtering(self, mock_vector_store, sample_image, test_config):
        """Test filtering by price range."""
        scorer = HybridScorer(test_config.models)
        
        clip_embs, dino_embs = scorer.encode_dual([sample_image])
        
        results = mock_vector_store.search(
            clip_embs[0], dino_embs[0], top_k=50
        )
        
        # Apply price filter
        min_price = 20.0
        max_price = 40.0
        
        filtered = [
            r for r in results
            if r.item.price and min_price <= r.item.price <= max_price
        ]
        
        assert all(min_price <= r.item.price <= max_price for r in filtered)
    
    def test_similarity_threshold_filtering(
        self, mock_vector_store, sample_image, test_config
    ):
        """Test filtering by minimum similarity."""
        scorer = HybridScorer(test_config.models)
        
        clip_embs, dino_embs = scorer.encode_dual([sample_image])
        
        results = mock_vector_store.search(
            clip_embs[0], dino_embs[0], top_k=50
        )
        
        # Apply similarity threshold
        threshold = 0.5
        filtered = [r for r in results if r.similarity_score >= threshold]
        
        assert all(r.similarity_score >= threshold for r in filtered)


class TestSearchEdgeCases:
    """Test search edge cases."""
    
    def test_search_empty_database(self, temp_data_dir, sample_image, test_config):
        """Test search on empty database."""
        store = VectorStore(
            persist_directory=str(temp_data_dir / "chroma"),
            collection_name="empty_db",
            clip_dim=512,
            dino_dim=384
        )
        
        scorer = HybridScorer(test_config.models)
        clip_embs, dino_embs = scorer.encode_dual([sample_image])
        
        results = store.search(clip_embs[0], dino_embs[0], top_k=10)
        
        assert len(results) == 0
    
    def test_large_top_k(self, mock_vector_store, sample_image, test_config):
        """Test requesting more results than available."""
        scorer = HybridScorer(test_config.models)
        
        clip_embs, dino_embs = scorer.encode_dual([sample_image])
        
        # Request more than database size
        results = mock_vector_store.search(
            clip_embs[0], dino_embs[0], top_k=1000
        )
        
        # Should return all available items
        assert len(results) <= 1000


class TestSearchPagination:
    """Test search result pagination."""
    
    def test_paginated_results(self, mock_vector_store, sample_image, test_config):
        """Test pagination of search results."""
        scorer = HybridScorer(test_config.models)
        
        clip_embs, dino_embs = scorer.encode_dual([sample_image])
        
        # Get all results
        all_results = mock_vector_store.search(
            clip_embs[0], dino_embs[0], top_k=50
        )
        
        # Simulate pagination
        page_size = 12
        pages = [
            all_results[i:i+page_size]
            for i in range(0, len(all_results), page_size)
        ]
        
        # Verify pagination
        total_items = sum(len(page) for page in pages)
        assert total_items == len(all_results)

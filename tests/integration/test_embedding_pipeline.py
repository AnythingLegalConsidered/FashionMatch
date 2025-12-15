"""Integration tests for embedding pipeline."""

import json
from pathlib import Path

import pytest

from src.core.embedding_pipeline import EmbeddingPipeline
from src.database.vector_store import get_vector_store


class TestEmbeddingPipelineReferences:
    """Test pipeline processing of reference images."""
    
    def test_process_references(self, test_config, temp_data_dir, sample_images):
        """Test processing reference images."""
        # Create reference directory with images
        ref_dir = temp_data_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(sample_images):
            img.save(ref_dir / f"ref_{i}.jpg")
        
        # Update config paths
        test_config.references_dir = str(ref_dir)
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        # Create pipeline
        pipeline = EmbeddingPipeline(test_config)
        
        # Process references
        stats = pipeline.process_references()
        
        assert stats.processed_items == len(sample_images)
        assert stats.skipped_items == 0
        assert stats.failed_items == 0
    
    def test_incremental_reference_processing(
        self, test_config, temp_data_dir, sample_images
    ):
        """Test incremental updates skip already-processed references."""
        ref_dir = temp_data_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        
        # Save first batch
        for i, img in enumerate(sample_images[:2]):
            img.save(ref_dir / f"ref_{i}.jpg")
        
        test_config.references_dir = str(ref_dir)
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        pipeline = EmbeddingPipeline(test_config)
        
        # First run
        stats1 = pipeline.process_references()
        assert stats1.processed_items == 2
        
        # Add more images
        for i, img in enumerate(sample_images[2:], start=2):
            img.save(ref_dir / f"ref_{i}.jpg")
        
        # Second run (should only process new images)
        stats2 = pipeline.process_references()
        assert stats2.processed_items == len(sample_images) - 2
        assert stats2.skipped_items == 2


class TestEmbeddingPipelineScraped:
    """Test pipeline processing of scraped items."""
    
    def test_process_scraped_batch(
        self, test_config, temp_data_dir, mock_scraped_batch, sample_images
    ):
        """Test processing scraped batch."""
        scraped_dir = temp_data_dir / "scraped"
        scraped_dir.mkdir(exist_ok=True)
        
        # Create batch JSON
        batch_data = {
            "items": [item.dict() for item in mock_scraped_batch],
            "category": "chemises",
            "timestamp": "2025-01-01T00:00:00"
        }
        
        batch_file = scraped_dir / "batch_001.json"
        with open(batch_file, "w") as f:
            json.dump(batch_data, f)
        
        # Save sample images
        for i, img in enumerate(sample_images[:len(mock_scraped_batch)]):
            img.save(scraped_dir / f"vinted_{i:03d}.jpg")
        
        test_config.scraped_dir = str(scraped_dir)
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        pipeline = EmbeddingPipeline(test_config)
        
        # Process scraped items
        stats = pipeline.process_scraped()
        
        assert stats.processed_items > 0


class TestEmbeddingPipelineFullWorkflow:
    """Test full pipeline workflow."""
    
    def test_process_all(self, test_config, temp_data_dir, sample_images):
        """Test processing all sources."""
        # Setup both reference and scraped directories
        ref_dir = temp_data_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        
        scraped_dir = temp_data_dir / "scraped"
        scraped_dir.mkdir(exist_ok=True)
        
        # Add reference images
        for i in range(2):
            sample_images[i].save(ref_dir / f"ref_{i}.jpg")
        
        test_config.references_dir = str(ref_dir)
        test_config.scraped_dir = str(scraped_dir)
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        pipeline = EmbeddingPipeline(test_config)
        
        # Process all
        stats = pipeline.process_all()
        
        # process_all returns dict, need to check both sources
        total_processed = stats["references"].processed_items + stats["scraped"].processed_items
        assert total_processed >= 2
    
    def test_force_reprocess(self, test_config, temp_data_dir, sample_images):
        """Test --force flag clears database."""
        ref_dir = temp_data_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        
        sample_images[0].save(ref_dir / "ref_0.jpg")
        
        test_config.references_dir = str(ref_dir)
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        pipeline = EmbeddingPipeline(test_config)
        
        # First run
        stats1 = pipeline.process_all(force=False)
        
        # Force reprocess
        stats2 = pipeline.process_all(force=True)
        
        # Should process same items again
        total1 = stats1["references"].processed_items + stats1["scraped"].processed_items
        total2 = stats2["references"].processed_items + stats2["scraped"].processed_items
        assert total2 == total1


class TestEmbeddingPipelineErrorHandling:
    """Test pipeline error handling."""
    
    def test_corrupted_image_handling(self, test_config, temp_data_dir):
        """Test handling of corrupted images."""
        ref_dir = temp_data_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        
        # Create corrupted file
        corrupt_file = ref_dir / "corrupt.jpg"
        corrupt_file.write_bytes(b"not an image")
        
        test_config.references_dir = str(ref_dir)
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        pipeline = EmbeddingPipeline(test_config)
        
        # Should not crash
        stats = pipeline.process_references()
        assert stats.failed_items >= 1
    
    def test_empty_directory(self, test_config, temp_data_dir):
        """Test processing empty directory."""
        ref_dir = temp_data_dir / "references"
        ref_dir.mkdir(exist_ok=True)
        
        test_config.references_dir = str(ref_dir)
        test_config.database.persist_directory = str(temp_data_dir / "chroma")
        
        pipeline = EmbeddingPipeline(test_config)
        
        stats = pipeline.process_references()
        
        assert stats.processed_items == 0
        assert stats.failed_items == 0

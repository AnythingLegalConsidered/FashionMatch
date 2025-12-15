"""Example usage of the EmbeddingPipeline.

This script demonstrates how to use the pipeline to process
reference images and scraped batches.
"""

from pathlib import Path

from src.core import EmbeddingPipeline


def main():
    """Run example pipeline processing."""
    
    # Initialize pipeline with default config
    print("Initializing EmbeddingPipeline...")
    pipeline = EmbeddingPipeline()
    
    print(f"Vector store initialized with {pipeline.vector_store.count()} existing items")
    print(f"CLIP encoder: {pipeline.scorer.clip_encoder.model_name}")
    print(f"DINOv2 encoder: {pipeline.scorer.dino_encoder.model_name}")
    print()
    
    # Define directories
    reference_dir = Path("data/references")
    scraped_dir = Path("data/scraped")
    
    # Example 1: Process reference images only
    print("=" * 60)
    print("Example 1: Processing reference images")
    print("=" * 60)
    
    if reference_dir.exists():
        stats = pipeline.process_references(
            reference_dir=reference_dir,
            batch_size=32
        )
        
        print(f"\nResults:")
        print(f"  Total items:      {stats.total_items}")
        print(f"  Processed:        {stats.processed_items}")
        print(f"  Skipped:          {stats.skipped_items}")
        print(f"  Failed:           {stats.failed_items}")
        print(f"  Duration:         {stats.duration_seconds:.2f}s")
    else:
        print(f"Reference directory not found: {reference_dir}")
        print("Skipping reference processing")
    
    print()
    
    # Example 2: Process scraped batches only
    print("=" * 60)
    print("Example 2: Processing scraped batches")
    print("=" * 60)
    
    if scraped_dir.exists():
        stats = pipeline.process_scraped_batches(
            scraped_dir=scraped_dir,
            batch_size=32
        )
        
        print(f"\nResults:")
        print(f"  Total items:      {stats.total_items}")
        print(f"  Processed:        {stats.processed_items}")
        print(f"  Skipped:          {stats.skipped_items}")
        print(f"  Failed:           {stats.failed_items}")
        print(f"  Duration:         {stats.duration_seconds:.2f}s")
        
        if stats.failed_ids:
            print(f"  Failed IDs:       {', '.join(stats.failed_ids[:5])}")
            if len(stats.failed_ids) > 5:
                print(f"                    ... and {len(stats.failed_ids) - 5} more")
    else:
        print(f"Scraped directory not found: {scraped_dir}")
        print("Skipping scraped batch processing")
    
    print()
    
    # Example 3: Process everything at once
    print("=" * 60)
    print("Example 3: Processing all sources")
    print("=" * 60)
    
    if reference_dir.exists() and scraped_dir.exists():
        results = pipeline.process_all(
            reference_dir=reference_dir,
            scraped_dir=scraped_dir,
            batch_size=32
        )
        
        print(f"\nCombined Results:")
        for source, stats in results.items():
            print(f"\n{source.upper()}:")
            print(f"  Processed:        {stats.processed_items}")
            print(f"  Skipped:          {stats.skipped_items}")
            print(f"  Failed:           {stats.failed_items}")
    else:
        print("One or more directories not found")
        print("Skipping combined processing")
    
    print()
    print("=" * 60)
    print(f"Final vector store count: {pipeline.vector_store.count()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

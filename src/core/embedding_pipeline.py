"""End-to-end embedding pipeline for fashion items.

This module orchestrates the complete flow from images to ChromaDB storage:
reference images and scraped batches → dual encoding → vector store persistence.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Optional

from PIL import Image
from pydantic import BaseModel, Field
from tqdm import tqdm

from src.database import (
    BatchInsertError,
    FashionItem,
    get_vector_store,
    vinted_item_to_fashion_item,
)
from src.scraper.storage import list_batches, load_batch
from src.utils import get_config, get_logger, load_image, log_exception, log_execution_time
from src.utils.config import AppConfig
from src.utils.performance import get_performance_monitor

from .scorer import get_hybrid_scorer

logger = get_logger(__name__)


class PipelineStats(BaseModel):
    """Statistics container for pipeline processing results."""
    
    total_items: int = Field(..., ge=0, description="Total items discovered")
    processed_items: int = Field(..., ge=0, description="Successfully encoded and stored")
    skipped_items: int = Field(..., ge=0, description="Already in vector store")
    failed_items: int = Field(..., ge=0, description="Errors during processing")
    failed_ids: list[str] = Field(default_factory=list, description="IDs of failed items")
    duration_seconds: float = Field(..., ge=0.0, description="Total processing time")
    source: str = Field(..., description="'references' or 'scraped'")


class EmbeddingPipeline:
    """Stateful orchestrator for end-to-end embedding generation and storage."""
    
    def __init__(self, config: Optional[AppConfig] = None, config_path: Optional[Path] = None):
        """Initialize pipeline with encoders and vector store.
        
        Args:
            config: AppConfig instance (takes precedence over config_path)
            config_path: Optional path to config YAML (uses default if both None)
        """
        logger.info("Initializing EmbeddingPipeline")
        
        # Load configuration - prefer explicit config over config_path
        if config is not None:
            self.config = config
        else:
            self.config = get_config(config_path)
        
        # Initialize hybrid scorer (contains both encoders)
        self.scorer = get_hybrid_scorer(self.config.models)
        
        # Get embedding dimensions from encoders
        clip_dim = self.scorer.clip_encoder.embedding_dim
        dino_dim = self.scorer.dino_encoder.embedding_dim
        
        # Initialize vector store
        self.vector_store = get_vector_store(
            config=self.config.database,
            clip_dim=clip_dim,
            dino_dim=dino_dim
        )
        
        logger.info(
            f"Pipeline initialized: CLIP dim={clip_dim}, DINO dim={dino_dim}, "
            f"store count={self.vector_store.count()}"
        )
    
    def process_references(
        self,
        reference_dir: Optional[Path] = None,
        batch_size: int = 32
    ) -> PipelineStats:
        """Process reference images from directory.
        
        If reference_dir is not provided, uses self.config.references_dir.
        
        Args:
            reference_dir: Directory containing reference images
            batch_size: Number of images to encode per batch
            
        Returns:
            PipelineStats with processing results
            
        Raises:
            FileNotFoundError: If reference_dir doesn't exist
        """
        if not reference_dir.exists():
            raise FileNotFoundError(f"Reference directory not found: {reference_dir}")
        
        logger.info(f"Processing reference images from {reference_dir}")
        start_time = time.time()
        
        # Scan for image files
        image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(reference_dir.glob(f"**/{ext}"))
        
        total_items = len(image_paths)
        logger.info(f"Found {total_items} image files")
        
        if total_items == 0:
            return PipelineStats(
                total_items=0,
                processed_items=0,
                skipped_items=0,
                failed_items=0,
                failed_ids=[],
                duration_seconds=time.time() - start_time,
                source="references"
            )
        
        # Get existing IDs for incremental updates
        existing_ids = set(self.vector_store.get_all_ids())
        logger.info(f"Found {len(existing_ids)} existing items in vector store")
        
        # Prepare items with deterministic IDs
        items_to_process = []
        
        for path in image_paths:
            # Use deterministic ID based on path for stable incremental updates
            # Use SHA-1 for stable hash across runs and Python versions
            path_str = str(path.absolute())
            path_hash = hashlib.sha1(path_str.encode('utf-8')).hexdigest()[:16]
            item_id = f"ref_{path.stem}_{path_hash}"
            if item_id not in existing_ids:
                items_to_process.append((item_id, path))
        
        skipped_items = total_items - len(items_to_process)
        logger.info(f"Processing {len(items_to_process)} new items, skipping {skipped_items} existing")
        
        # Process in batches
        processed_items = 0
        failed_items = 0
        failed_ids = []
        
        with log_execution_time(logger, f"reference image processing"):
            for i in tqdm(
                range(0, len(items_to_process), batch_size),
                desc="Processing references",
                unit="batch"
            ):
                batch = items_to_process[i:i + batch_size]
                
                # Load images
                batch_images = []
                batch_items = []
                batch_failed = []
                
                for item_id, path in batch:
                    try:
                        image = load_image(path)
                        batch_images.append(image)
                        
                        # Create FashionItem with metadata
                        fashion_item = FashionItem(
                            item_id=item_id,
                            title=path.name,
                            price=0.0,
                            url=f"file://{path.absolute()}",
                            image_url=f"file://{path.absolute()}",
                            local_image_path=str(path),
                            additional_metadata={"source": "reference"}
                        )
                        batch_items.append(fashion_item)
                    
                    except Exception as e:
                        log_exception(logger, f"load image {path.name}", e)
                        batch_failed.append(item_id)
                        failed_items += 1
                
                # Encode and store batch
                if batch_images:
                    success_count = self._encode_and_store_batch(batch_items, batch_images)
                    processed_items += success_count
                    
                    # Track batch-level failures
                    batch_encode_failures = len(batch_items) - success_count
                    if batch_encode_failures > 0:
                        failed_items += batch_encode_failures
                        batch_failed.extend([
                            item.item_id for item in batch_items[success_count:]
                        ])
                
                failed_ids.extend(batch_failed)
        
        duration = time.time() - start_time
        
        stats = PipelineStats(
            total_items=total_items,
            processed_items=processed_items,
            skipped_items=skipped_items,
            failed_items=failed_items,
            failed_ids=failed_ids,
            duration_seconds=duration,
            source="references"
        )
        
        logger.info(
            f"Reference processing complete: {processed_items} processed, "
            f"{skipped_items} skipped, {failed_items} failed in {duration:.2f}s"
        )
        
        return stats
    
    def process_scraped_batches(
        self,
        scraped_dir: Optional[Path] = None,
        batch_size: int = 32
    ) -> PipelineStats:
        """Process scraped batches from storage.
        
        If scraped_dir is not provided, uses self.config.scraped_dir.
        
        Args:
            scraped_dir: Directory containing scraped batch JSON files (uses config if None)
            batch_size: Number of items to encode per batch
            
        Returns:
            PipelineStats with aggregated results
            
        Raises:
            FileNotFoundError: If scraped_dir doesn't exist
        """
        if scraped_dir is None:
            scraped_dir = Path(self.config.scraped_dir)
        
        if not scraped_dir.exists():
            raise FileNotFoundError(f"Scraped directory not found: {scraped_dir}")
        
        logger.info(f"Processing scraped batches from {scraped_dir}")
        start_time = time.time()
        
        # Get existing IDs once for all batches
        existing_ids = set(self.vector_store.get_all_ids())
        logger.info(f"Found {len(existing_ids)} existing items in vector store")
        
        # List all batches
        try:
            batch_ids = list_batches(scraped_dir)
        except Exception as e:
            log_exception(logger, "list scraped batches", e)
            batch_ids = []
        
        logger.info(f"Found {len(batch_ids)} scraped batches")
        
        if not batch_ids:
            return PipelineStats(
                total_items=0,
                processed_items=0,
                skipped_items=0,
                failed_items=0,
                failed_ids=[],
                duration_seconds=time.time() - start_time,
                source="scraped"
            )
        
        # Aggregate stats
        total_items = 0
        processed_items = 0
        skipped_items = 0
        failed_items = 0
        failed_ids = []
        
        with log_execution_time(logger, f"scraped batch processing"):
            for batch_id in tqdm(batch_ids, desc="Processing batches", unit="batch"):
                try:
                    # Load batch
                    batch = load_batch(scraped_dir / batch_id)
                    logger.debug(f"Loaded batch {batch_id} with {len(batch.items)} items")
                    
                    # Filter new items
                    new_items = [
                        item for item in batch.items
                        if item.item_id not in existing_ids
                    ]
                    
                    total_items += len(batch.items)
                    batch_skipped = len(batch.items) - len(new_items)
                    skipped_items += batch_skipped
                    
                    if not new_items:
                        logger.debug(f"Batch {batch_id}: all {len(batch.items)} items already processed")
                        continue
                    
                    logger.info(
                        f"Batch {batch_id}: processing {len(new_items)} new items, "
                        f"skipping {batch_skipped} existing"
                    )
                    
                    # Process items in sub-batches
                    for i in range(0, len(new_items), batch_size):
                        sub_batch = new_items[i:i + batch_size]
                        
                        # Load images and convert to FashionItems
                        batch_images = []
                        fashion_items = []
                        batch_failed = []
                        
                        for vinted_item in sub_batch:
                            try:
                                # Check for local image
                                if not vinted_item.local_image_paths:
                                    logger.warning(
                                        f"Item {vinted_item.item_id} has no local images, skipping"
                                    )
                                    batch_failed.append(vinted_item.item_id)
                                    failed_items += 1
                                    continue
                                
                                # Load primary image
                                image_path = Path(vinted_item.local_image_paths[0])
                                if not image_path.exists():
                                    logger.warning(
                                        f"Image not found for {vinted_item.item_id}: {image_path}"
                                    )
                                    batch_failed.append(vinted_item.item_id)
                                    failed_items += 1
                                    continue
                                
                                image = load_image(image_path)
                                batch_images.append(image)
                                
                                # Convert to FashionItem
                                fashion_item = vinted_item_to_fashion_item(vinted_item)
                                fashion_items.append(fashion_item)
                            
                            except Exception as e:
                                log_exception(
                                    logger,
                                    f"process item {vinted_item.item_id}",
                                    e
                                )
                                batch_failed.append(vinted_item.item_id)
                                failed_items += 1
                        
                        # Encode and store
                        if batch_images:
                            success_count = self._encode_and_store_batch(
                                fashion_items,
                                batch_images
                            )
                            processed_items += success_count
                            
                            # Track encoding/storage failures
                            batch_encode_failures = len(fashion_items) - success_count
                            if batch_encode_failures > 0:
                                failed_items += batch_encode_failures
                                batch_failed.extend([
                                    item.item_id for item in fashion_items[success_count:]
                                ])
                        
                        failed_ids.extend(batch_failed)
                
                except Exception as e:
                    log_exception(logger, f"process batch {batch_id}", e)
                    continue
        
        duration = time.time() - start_time
        
        stats = PipelineStats(
            total_items=total_items,
            processed_items=processed_items,
            skipped_items=skipped_items,
            failed_items=failed_items,
            failed_ids=failed_ids,
            duration_seconds=duration,
            source="scraped"
        )
        
        logger.info(
            f"Scraped batch processing complete: {processed_items} processed, "
            f"{skipped_items} skipped, {failed_items} failed in {duration:.2f}s"
        )
        
        return stats
    
    def process_scraped(self, batch_size: int = 32) -> PipelineStats:
        """Convenience method to process scraped batches using config directory.
        
        Args:
            batch_size: Number of items to encode per batch
            
        Returns:
            PipelineStats with aggregated results
        """
        return self.process_scraped_batches(scraped_dir=None, batch_size=batch_size)
    
    def process_all(
        self,
        reference_dir: Optional[Path] = None,
        scraped_dir: Optional[Path] = None,
        batch_size: int = 32,
        force: bool = False
    ) -> dict[str, PipelineStats]:
        """Process both reference images and scraped batches.
        
        If directories are not provided, uses self.config paths.
        
        Args:
            reference_dir: Directory containing reference images
            scraped_dir: Directory containing scraped batches
            batch_size: Encoding batch size
            
        Returns:
            Dictionary with 'references' and 'scraped' stats
        """
        logger.info("Processing all sources (references + scraped)")
        
        results = {}
        
        # Process references
        ref_start_time = time.time()
        try:
            results["references"] = self.process_references(reference_dir, batch_size)
        except Exception as e:
            ref_duration = time.time() - ref_start_time
            log_exception(logger, "process references", e)
            results["references"] = PipelineStats(
                total_items=0,
                processed_items=0,
                skipped_items=0,
                failed_items=0,
                failed_ids=[],
                duration_seconds=ref_duration,
                source="references"
            )
        
        # Process scraped batches
        scraped_start_time = time.time()
        try:
            results["scraped"] = self.process_scraped_batches(scraped_dir, batch_size)
        except Exception as e:
            scraped_duration = time.time() - scraped_start_time
            log_exception(logger, "process scraped batches", e)
            results["scraped"] = PipelineStats(
                total_items=0,
                processed_items=0,
                skipped_items=0,
                failed_items=0,
                failed_ids=[],
                duration_seconds=scraped_duration,
                source="scraped"
            )
        
        # Log combined summary
        total_processed = (
            results["references"].processed_items +
            results["scraped"].processed_items
        )
        total_failed = (
            results["references"].failed_items +
            results["scraped"].failed_items
        )
        total_duration = (
            results["references"].duration_seconds +
            results["scraped"].duration_seconds
        )
        
        logger.info(
            f"All processing complete: {total_processed} total processed, "
            f"{total_failed} total failed in {total_duration:.2f}s"
        )
        
        return results
    
    def _encode_and_store_batch(
        self,
        items: list[FashionItem],
        images: list[Image.Image]
    ) -> int:
        """Encode images and store items with embeddings.
        
        Args:
            items: FashionItem objects without embeddings
            images: Corresponding PIL images
            
        Returns:
            Number of successfully stored items
        """
        if len(items) != len(images):
            logger.error(
                f"Item/image count mismatch: {len(items)} items, {len(images)} images"
            )
            return 0
        
        try:
            # Encode with dual encoders
            monitor = get_performance_monitor()
            with monitor.measure("encode_batch", items_count=len(images)):
                clip_embeddings, dino_embeddings = self.scorer.encode_dual(images)
            
            # Assign embeddings to items
            for item, clip_emb, dino_emb in zip(items, clip_embeddings, dino_embeddings):
                item.clip_embedding = clip_emb
                item.dino_embedding = dino_emb
            
            # Insert to vector store
            try:
                monitor = get_performance_monitor()
                with monitor.measure("insert_batch", items_count=len(items)):
                    result = self.vector_store.add_items(items)
                logger.debug(
                    f"Stored {result.success_count} items in {result.total_time:.2f}s"
                )
                return result.success_count
            
            except BatchInsertError as e:
                logger.error(
                    f"Batch insert failed for {len(e.failed_ids)} items: {e.failed_ids}"
                )
                # Return number of successful inserts before error
                return len(items) - len(e.failed_ids)
        
        except Exception as e:
            log_exception(logger, "encode and store batch", e)
            return 0


def _print_stats_table(stats: dict[str, PipelineStats]) -> None:
    """Print formatted statistics table.
    
    Args:
        stats: Dictionary of PipelineStats by source
    """
    print("\n" + "=" * 80)
    print("EMBEDDING PIPELINE RESULTS")
    print("=" * 80)
    
    for source, stat in stats.items():
        print(f"\n{source.upper()}:")
        print(f"  Total items:      {stat.total_items}")
        print(f"  Processed:        {stat.processed_items}")
        print(f"  Skipped:          {stat.skipped_items}")
        print(f"  Failed:           {stat.failed_items}")
        print(f"  Duration:         {stat.duration_seconds:.2f}s")
        
        if stat.failed_ids:
            print(f"  Failed IDs:       {', '.join(stat.failed_ids[:10])}")
            if len(stat.failed_ids) > 10:
                print(f"                    ... and {len(stat.failed_ids) - 10} more")
    
    # Combined totals
    total_processed = sum(s.processed_items for s in stats.values())
    total_failed = sum(s.failed_items for s in stats.values())
    total_duration = sum(s.duration_seconds for s in stats.values())
    
    print(f"\nTOTAL:")
    print(f"  Processed:        {total_processed}")
    print(f"  Failed:           {total_failed}")
    print(f"  Total Duration:   {total_duration:.2f}s")
    print("=" * 80 + "\n")


def main():
    """CLI entry point for embedding pipeline."""
    parser = argparse.ArgumentParser(
        description="FashionMatch Embedding Pipeline - Generate and store dual embeddings"
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config YAML (default: config/config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["references", "scraped", "all"],
        default="all",
        help="Processing mode (default: all)"
    )
    
    parser.add_argument(
        "--references-dir",
        type=Path,
        default=None,
        help="Path to reference images directory (overrides config)"
    )
    
    parser.add_argument(
        "--scraped-dir",
        type=Path,
        default=None,
        help="Path to scraped batches directory (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (default: 32)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear vector store before processing (DANGEROUS)"
    )
    
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling and print detailed metrics"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = EmbeddingPipeline(config_path=args.config)
        
        # Enable performance monitoring if requested
        if args.profile:
            monitor = get_performance_monitor()
            monitor.enable()
            logger.info("Performance profiling enabled")
        
        # Handle force clear
        if args.force:
            response = input(
                "WARNING: This will delete ALL embeddings from the vector store. "
                "Type 'yes' to confirm: "
            )
            if response.lower() == "yes":
                logger.warning("Clearing vector store")
                pipeline.vector_store.clear()
                logger.info("Vector store cleared")
            else:
                logger.info("Clear cancelled")
                return
        
        # Determine directories
        references_dir = args.references_dir
        if references_dir is None:
            references_dir = Path(pipeline.config.references_dir)
        
        scraped_dir = args.scraped_dir
        if scraped_dir is None:
            scraped_dir = Path(pipeline.config.scraped_dir)
        
        # Process based on mode
        results = {}
        
        if args.mode == "references":
            stats = pipeline.process_references(references_dir, args.batch_size)
            results["references"] = stats
        
        elif args.mode == "scraped":
            stats = pipeline.process_scraped_batches(scraped_dir, args.batch_size)
            results["scraped"] = stats
        
        elif args.mode == "all":
            results = pipeline.process_all(
                references_dir,
                scraped_dir,
                args.batch_size
            )
        
        # Print results
        _print_stats_table(results)
        
        # Print performance report if profiling was enabled
        if args.profile:
            monitor = get_performance_monitor()
            monitor.print_report()
        
        # Exit with appropriate code
        total_failed = sum(s.failed_items for s in results.values())
        sys.exit(1 if total_failed > 0 else 0)
    
    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        log_exception(logger, "pipeline execution", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

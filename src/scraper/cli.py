"""Command-line interface for Vinted scraper.

Usage:
    python -m src.scraper.cli --category "robes" --max-pages 5
"""

import argparse
import asyncio
import random
import sys
from pathlib import Path

from src.utils import get_config, get_logger, set_log_level, log_execution_time

from .storage import save_batch, export_batch_to_csv
from .vinted_scraper import VintedScraper

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Scrape fashion items from Vinted marketplace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape dresses with default settings
  python -m src.scraper.cli --category "robes"
  
  # Scrape shirts, 10 pages, with debug logging
  python -m src.scraper.cli --category "chemises" --max-pages 10 --log-level DEBUG
  
  # Scrape without downloading images
  python -m src.scraper.cli --category "jeans" --no-download-images
  
  # Use custom config and output directory
  python -m src.scraper.cli --category "vestes" --config config/custom.yaml --output-dir data/custom/
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--category',
        type=str,
        required=True,
        help='Search category or keyword (e.g., "robes", "chemises", "jeans")'
    )
    
    # Optional arguments
    parser.add_argument(
        '--max-pages',
        type=int,
        default=None,
        help='Maximum number of pages to scrape (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/scraped'),
        help='Output directory for scraped data (default: data/scraped)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to configuration file (default: auto-detect)'
    )
    
    parser.add_argument(
        '--no-download-images',
        action='store_true',
        help='Skip downloading product images'
    )
    
    parser.add_argument(
        '--enrich-details',
        action='store_true',
        help='Fetch detailed item pages for description, brand, condition, etc.'
    )
    
    parser.add_argument(
        '--export-csv',
        action='store_true',
        help='Export results to CSV file'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default=None,
        help='Override log level'
    )
    
    return parser.parse_args()


async def main() -> int:
    """Main CLI entry point.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    args = parse_args()
    
    try:
        # Load configuration
        config = get_config(args.config)
        
        # Set log level if provided
        if args.log_level:
            set_log_level(logger, args.log_level)
        
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize scraper
        logger.info("=" * 60)
        logger.info("Vinted Scraper - FashionMatch")
        logger.info("=" * 60)
        logger.info(f"Category: {args.category}")
        logger.info(f"Max pages: {args.max_pages or config.scraper.max_pages}")
        logger.info(f"Output directory: {args.output_dir}")
        logger.info(f"Download images: {not args.no_download_images}")
        logger.info(f"Enrich details: {args.enrich_details}")
        logger.info("=" * 60)
        
        # Create scraper
        scraper = VintedScraper(config_path=args.config)
        
        # Scrape category
        with log_execution_time(logger, "entire scraping operation"):
            batch = await scraper.scrape_category(
                category=args.category,
                max_pages=args.max_pages
            )
            
            # Enrich with detailed item pages if requested
            if args.enrich_details and batch.items:
                logger.info(f"Enriching {len(batch.items)} items with detailed information...")
                enriched_items = []
                successful_enrichments = 0
                
                for i, item in enumerate(batch.items, 1):
                    try:
                        logger.debug(f"Enriching item {i}/{len(batch.items)}: {item.url}")
                        
                        # Fetch detailed item page
                        detailed_item = await scraper.scrape_item_details(item.url)
                        
                        # Preserve original scraped_at timestamp and local_image_paths
                        detailed_item.scraped_at = item.scraped_at
                        detailed_item.local_image_paths = item.local_image_paths
                        
                        enriched_items.append(detailed_item)
                        successful_enrichments += 1
                        
                        # Apply rate limiting between detail requests
                        if i < len(batch.items):
                            min_delay, max_delay = config.scraper.delay_range
                            delay = random.uniform(min_delay, max_delay)
                            await asyncio.sleep(delay)
                    
                    except Exception as e:
                        logger.warning(f"Failed to enrich item {item.url}: {e}")
                        # Keep original item if enrichment fails
                        enriched_items.append(item)
                
                # Replace items with enriched versions
                batch.items = enriched_items
                logger.info(f"Enrichment complete: {successful_enrichments}/{len(batch.items)} items successfully enriched")
            
            # Download images if requested
            if not args.no_download_images and batch.items:
                logger.info("Starting image download...")
                batch_dir = args.output_dir / batch.batch_id
                await scraper.download_images(batch.items, batch_dir)
            
            # Save batch
            logger.info("Saving batch data...")
            batch_path = save_batch(batch, args.output_dir)
            
            # Export to CSV if requested
            if args.export_csv:
                csv_path = batch_path / f"{batch.batch_id}.csv"
                export_batch_to_csv(batch, csv_path)
                logger.info(f"Exported to CSV: {csv_path}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("SCRAPING SUMMARY")
        print("=" * 60)
        print(f"Batch ID: {batch.batch_id}")
        print(f"Category: {batch.category}")
        print(f"Total Items: {batch.total_items}")
        print(f"Pages Scraped: {batch.pages_scraped}")
        if args.enrich_details:
            enriched_count = sum(1 for item in batch.items if item.description or item.brand or item.condition)
            print(f"Items Enriched: {enriched_count}/{batch.total_items}")
        print(f"Output Path: {batch_path}")
        print("=" * 60)
        
        # Per-item summary
        if batch.items:
            print(f"\nFirst 5 items:")
            for i, item in enumerate(batch.items[:5], 1):
                print(f"  {i}. {item.title} - {item.price}€")
                print(f"     URL: {item.url}")
        
        print(f"\n✓ Scraping completed successfully!")
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Scraping interrupted by user")
        print("\n✗ Scraping cancelled by user")
        return 1
    
    except Exception as e:
        logger.error(f"Scraping failed: {e}", exc_info=True)
        print(f"\n✗ Scraping failed: {e}")
        return 1


if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

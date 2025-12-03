#!/usr/bin/env python3
"""
Scraping test script for FashionMatch.

This script tests the VintedScraper by:
1. Loading configuration
2. Scraping a sample Vinted search page
3. Downloading images for found items
4. Saving metadata to a JSON file
5. Displaying results in the console

Usage:
    python scripts/scrape_test.py
    python scripts/scrape_test.py --url "https://www.vinted.fr/catalog?search_text=veste"
    python scripts/scrape_test.py --max-items 10 --no-headless

Author: FashionMatch Team
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import get_settings
from src.utils.logger import get_logger, configure_logging
from src.infrastructure.scraper import VintedScraper


# Configure logging
configure_logging()
logger = get_logger(__name__)


# ============================================
# Default Test Configuration
# ============================================

# Sample Vinted search URLs for testing
SAMPLE_URLS = {
    "t-shirts": "https://www.vinted.fr/catalog?search_text=t-shirt&order=newest_first",
    "vestes": "https://www.vinted.fr/catalog?search_text=veste&order=newest_first",
    "jeans": "https://www.vinted.fr/catalog?search_text=jean&order=newest_first",
    "robes": "https://www.vinted.fr/catalog?search_text=robe&order=newest_first",
    "sneakers": "https://www.vinted.fr/catalog?search_text=sneakers&order=newest_first",
}

DEFAULT_URL = SAMPLE_URLS["t-shirts"]
DEFAULT_MAX_ITEMS = 5
OUTPUT_FILE = PROJECT_ROOT / "data" / "debug_items.json"


# ============================================
# Helper Functions
# ============================================

def item_to_dict(item) -> dict:
    """Convert a ClothingItem to a JSON-serializable dictionary."""
    return {
        "id": item.id,
        "title": item.title,
        "price": item.price,
        "brand": item.brand,
        "size": item.size,
        "condition": item.condition,
        "image_url": item.image_url,
        "local_image_path": item.local_image_path,
        "item_url": item.item_url,
        "description": item.description,
        "seller_id": item.seller_id,
    }


def print_item_summary(item, index: int) -> None:
    """Print a formatted summary of an item."""
    print(f"\n{'='*60}")
    print(f"📦 Item #{index + 1}: {item.title or 'Sans titre'}")
    print(f"{'='*60}")
    print(f"   💰 Prix: {item.price or 'N/A'}€")
    print(f"   🏷️  Marque: {item.brand or 'N/A'}")
    print(f"   📏 Taille: {item.size or 'N/A'}")
    print(f"   ⭐ État: {item.condition or 'N/A'}")
    print(f"   🔗 URL: {item.item_url or 'N/A'}")
    if item.local_image_path:
        print(f"   🖼️  Image locale: {item.local_image_path}")


def print_banner():
    """Print a nice banner."""
    print("\n" + "="*60)
    print("🛍️  FashionMatch - Scraping Test")
    print("="*60)


def print_stats(scraper: VintedScraper):
    """Print scraping statistics."""
    stats = scraper.stats
    print(f"\n📊 Statistiques de scraping:")
    print(f"   • Articles trouvés: {stats.items_scraped}")
    print(f"   • Images téléchargées: {stats.images_downloaded}")
    print(f"   • Pages visitées: {stats.pages_visited}")
    print(f"   • Erreurs: {stats.errors}")
    print(f"   • Durée: {stats.duration_seconds:.1f}s")


# ============================================
# Main Scraping Function
# ============================================

async def run_scraping_test(
    url: str,
    max_items: int,
    headless: bool,
    download_images: bool,
) -> list:
    """
    Run the scraping test.
    
    Args:
        url: Vinted search/category URL to scrape.
        max_items: Maximum number of items to scrape.
        headless: Whether to run browser in headless mode.
        download_images: Whether to download item images.
        
    Returns:
        List of scraped items.
    """
    print(f"\n🔧 Configuration:")
    print(f"   • URL: {url}")
    print(f"   • Max items: {max_items}")
    print(f"   • Mode headless: {headless}")
    print(f"   • Télécharger images: {download_images}")
    
    items = []
    
    async with VintedScraper(headless=headless) as scraper:
        print(f"\n🌐 Navigateur lancé (User-Agent: {scraper.user_agent[:50]}...)")
        
        # Scrape the category page
        print(f"\n🔍 Scraping de la page...")
        items = await scraper.scrape_category(
            url=url,
            max_items=max_items,
            scroll_count=3,  # Scroll 3 times for lazy loading
            download_images=download_images,
        )
        
        print(f"\n✅ {len(items)} articles trouvés!")
        
        # Print each item
        for i, item in enumerate(items):
            print_item_summary(item, i)
        
        # Print statistics
        print_stats(scraper)
    
    return items


def save_results(items: list, output_file: Path) -> None:
    """Save scraped items to a JSON file."""
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert items to dictionaries
    data = {
        "scraped_at": datetime.now().isoformat(),
        "count": len(items),
        "items": [item_to_dict(item) for item in items],
    }
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Résultats sauvegardés dans: {output_file}")


# ============================================
# CLI Interface
# ============================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test the FashionMatch Vinted scraper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/scrape_test.py
  python scripts/scrape_test.py --url "https://www.vinted.fr/catalog?search_text=veste"
  python scripts/scrape_test.py --max-items 10 --no-headless
  python scripts/scrape_test.py --category jeans
        """,
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Vinted search URL to scrape",
    )
    
    parser.add_argument(
        "--category",
        type=str,
        choices=list(SAMPLE_URLS.keys()),
        default="t-shirts",
        help="Predefined category to search (default: t-shirts)",
    )
    
    parser.add_argument(
        "--max-items",
        type=int,
        default=DEFAULT_MAX_ITEMS,
        help=f"Maximum items to scrape (default: {DEFAULT_MAX_ITEMS})",
    )
    
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser in visible mode (not headless)",
    )
    
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip downloading images",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output JSON file (default: {OUTPUT_FILE})",
    )
    
    return parser.parse_args()


# ============================================
# Main Entry Point
# ============================================

def main():
    """Main entry point."""
    print_banner()
    
    # Parse arguments
    args = parse_args()
    
    # Determine URL
    if args.url:
        url = args.url
    else:
        url = SAMPLE_URLS.get(args.category, DEFAULT_URL)
    
    # Configuration
    headless = not args.no_headless
    download_images = not args.no_images
    output_file = Path(args.output)
    
    try:
        # Load settings to verify configuration
        settings = get_settings()
        logger.info(f"Configuration loaded: {settings.app.name} v{settings.app.version}")
        
        # Run the scraping test
        items = asyncio.run(
            run_scraping_test(
                url=url,
                max_items=args.max_items,
                headless=headless,
                download_images=download_images,
            )
        )
        
        # Save results
        if items:
            save_results(items, output_file)
        else:
            print("\n⚠️  Aucun article trouvé. Vérifiez l'URL ou les sélecteurs CSS.")
        
        print("\n✅ Test terminé avec succès!")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrompu par l'utilisateur")
        sys.exit(1)
        
    except Exception as e:
        logger.exception(f"Erreur lors du test: {e}")
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

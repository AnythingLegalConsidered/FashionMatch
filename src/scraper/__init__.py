"""Web scraping components for Vinted.

This module provides a complete scraping solution for Vinted marketplace:
- VintedScraper: Async Playwright-based scraper with rate limiting
- VintedParser: Multi-strategy HTML/JSON parser
- Storage utilities: Save/load scraped data
- Data models: Type-safe Pydantic models for items and batches

Usage:
    from src.scraper import VintedScraper, save_batch
    
    scraper = VintedScraper()
    batch = await scraper.scrape_category("robes", max_pages=5)
    save_batch(batch, Path("data/scraped"))
"""

from .models import ScrapedBatch, VintedItem
from .parsers import VintedParser
from .storage import load_batch, list_batches, save_batch
from .vinted_scraper import VintedScraper

__all__ = [
    "VintedItem",
    "ScrapedBatch",
    "VintedParser",
    "VintedScraper",
    "save_batch",
    "load_batch",
    "list_batches",
]

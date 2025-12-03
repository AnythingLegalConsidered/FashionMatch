# Scraper Package
"""
Web scraping implementations for Vinted.

This module provides:
- VintedScraper: Main Playwright-based scraper
- RateLimiter: Request rate limiting with backoff
- Parsers: HTML parsing for listings and item details
"""

from src.infrastructure.scraper.rate_limiter import (
    RateLimiter,
    AdaptiveRateLimiter,
    RateLimiterConfig,
    create_rate_limiter_from_config,
)
from src.infrastructure.scraper.vinted_scraper import (
    VintedScraper,
    ScraperStats,
    create_scraper,
)
from src.infrastructure.scraper.parsers import (
    ListingParser,
    ItemParser,
    ListingSelectors,
    ItemSelectors,
)

__all__ = [
    # Main scraper
    "VintedScraper",
    "ScraperStats",
    "create_scraper",
    # Rate limiting
    "RateLimiter",
    "AdaptiveRateLimiter",
    "RateLimiterConfig",
    "create_rate_limiter_from_config",
    # Parsers
    "ListingParser",
    "ItemParser",
    "ListingSelectors",
    "ItemSelectors",
]

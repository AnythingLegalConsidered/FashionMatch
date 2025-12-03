"""
Vinted scraper using Playwright.

Main scraper implementation that handles:
- Browser session management with Playwright
- Cookie consent handling
- Category/search page scraping with lazy loading
- Image downloading
- Rate limiting for respectful scraping

Example:
    >>> async with VintedScraper() as scraper:
    ...     items = await scraper.scrape_category(
    ...         "https://www.vinted.fr/catalog?catalog[]=5"
    ...     )
    ...     for item in items:
    ...         print(f"{item.title}: {item.price}€")
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, List, Optional, TYPE_CHECKING

import aiohttp

from src.domain.entities.clothing_item import ClothingItem
from src.infrastructure.scraper.rate_limiter import (
    AdaptiveRateLimiter,
    create_rate_limiter_from_config,
)
from src.infrastructure.scraper.parsers import ListingParser, ItemParser
from src.utils.config import get_settings
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

logger = get_logger(__name__)


# ============================================
# Constants
# ============================================

# Realistic User-Agent strings (rotate to avoid detection)
USER_AGENTS = [
    # Chrome on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    # Chrome on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # Firefox on Windows
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    # Safari on Mac
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
]

# Cookie consent button selectors (try in order)
COOKIE_CONSENT_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "button[id*='accept']",
    "button[data-testid='cookie-consent-accept']",
    "[class*='cookie'] button[class*='accept']",
    "button:has-text('Accepter')",
    "button:has-text('Accept')",
    "button:has-text('Tout accepter')",
    "button:has-text('Accept All')",
]


@dataclass
class ScraperStats:
    """
    Statistics for a scraping session.
    
    Tracks items scraped, images downloaded, errors, etc.
    """
    items_scraped: int = 0
    images_downloaded: int = 0
    pages_visited: int = 0
    errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def start(self) -> None:
        """Mark the start of the session."""
        self.start_time = datetime.now()
    
    def stop(self) -> None:
        """Mark the end of the session."""
        self.end_time = datetime.now()
    
    @property
    def duration_seconds(self) -> float:
        """Get session duration in seconds."""
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()
    
    def __str__(self) -> str:
        return (
            f"ScraperStats(items={self.items_scraped}, "
            f"images={self.images_downloaded}, "
            f"pages={self.pages_visited}, "
            f"errors={self.errors}, "
            f"duration={self.duration_seconds:.1f}s)"
        )


class VintedScraper:
    """
    Main Vinted scraper using Playwright.
    
    Provides async methods for scraping Vinted category pages,
    item details, and downloading images.
    
    Uses an adaptive rate limiter to respect the website and
    avoid detection.
    
    Attributes:
        base_url: Base URL for Vinted.
        headless: Whether to run browser in headless mode.
        rate_limiter: Rate limiter for requests.
        stats: Scraping statistics.
        
    Example:
        >>> async with VintedScraper() as scraper:
        ...     items = await scraper.scrape_category(url, max_items=50)
        ...     for item in items:
        ...         local_path = await scraper.download_image(item.image_url)
    """
    
    def __init__(
        self,
        headless: Optional[bool] = None,
        rate_limiter: Optional[AdaptiveRateLimiter] = None,
        user_agent: Optional[str] = None,
        images_dir: Optional[Path] = None,
    ):
        """
        Initialize the Vinted scraper.
        
        Args:
            headless: Run browser in headless mode. If None, uses config.
            rate_limiter: Custom rate limiter. If None, creates from config.
            user_agent: Custom user agent. If None, uses random from list.
            images_dir: Directory for downloaded images. If None, uses config.
        """
        self.settings = get_settings()
        
        # Configuration
        self.base_url = self.settings.scraper.base_url
        self.headless = headless if headless is not None else self.settings.scraper.headless
        self.timeout = self.settings.scraper.timeout_seconds * 1000  # Convert to ms
        self.max_pages = self.settings.scraper.max_pages_per_category
        
        # User agent
        if user_agent:
            self.user_agent = user_agent
        elif self.settings.scraper.user_agent:
            self.user_agent = self.settings.scraper.user_agent
        else:
            import random
            self.user_agent = random.choice(USER_AGENTS)
        
        # Rate limiter
        self.rate_limiter = rate_limiter or create_rate_limiter_from_config()
        
        # Images directory
        if images_dir:
            self.images_dir = images_dir
        else:
            self.images_dir = Path(self.settings.images.cache_dir)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        
        # Parsers
        self.listing_parser = ListingParser()
        self.item_parser = ItemParser()
        
        # Session state
        self._playwright: Optional["Playwright"] = None
        self._browser: Optional["Browser"] = None
        self._context: Optional["BrowserContext"] = None
        self._page: Optional["Page"] = None
        self._session_active = False
        
        # Statistics
        self.stats = ScraperStats()
        
        logger.info(
            f"VintedScraper initialized: base_url={self.base_url}, "
            f"headless={self.headless}"
        )
    
    # =========================================
    # Context Manager
    # =========================================
    
    async def __aenter__(self) -> "VintedScraper":
        """Enter async context manager."""
        await self.start_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close_session()
    
    # =========================================
    # Session Management
    # =========================================
    
    async def start_session(self) -> None:
        """
        Start the browser session.
        
        Launches Playwright, creates a browser instance with
        anti-detection settings, and opens a new page.
        
        Raises:
            RuntimeError: If session is already active.
        """
        if self._session_active:
            logger.warning("Session already active")
            return
        
        logger.info("Starting browser session...")
        
        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise ImportError(
                "Playwright is not installed. "
                "Run: pip install playwright && playwright install chromium"
            )
        
        # Start Playwright
        self._playwright = await async_playwright().start()
        
        # Launch browser with anti-detection settings
        self._browser = await self._playwright.chromium.launch(
            headless=self.headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )
        
        # Create context with realistic settings
        self._context = await self._browser.new_context(
            user_agent=self.user_agent,
            viewport={"width": 1920, "height": 1080},
            locale="fr-FR",
            timezone_id="Europe/Paris",
            permissions=["geolocation"],
            geolocation={"latitude": 48.8566, "longitude": 2.3522},  # Paris
        )
        
        # Add anti-detection script
        await self._context.add_init_script("""
            // Override webdriver property
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
            
            // Override plugins
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
            
            // Override languages
            Object.defineProperty(navigator, 'languages', {
                get: () => ['fr-FR', 'fr', 'en-US', 'en']
            });
        """)
        
        # Create page
        self._page = await self._context.new_page()
        self._page.set_default_timeout(self.timeout)
        
        self._session_active = True
        self.stats.start()
        
        logger.info(
            f"Browser session started (headless={self.headless}, "
            f"user_agent={self.user_agent[:50]}...)"
        )
    
    async def close_session(self) -> None:
        """
        Close the browser session.
        
        Closes page, context, browser, and stops Playwright.
        """
        if not self._session_active:
            return
        
        logger.info("Closing browser session...")
        
        try:
            if self._page:
                await self._page.close()
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
        except Exception as e:
            logger.error(f"Error closing session: {e}")
        finally:
            self._page = None
            self._context = None
            self._browser = None
            self._playwright = None
            self._session_active = False
            self.stats.stop()
        
        logger.info(f"Session closed. {self.stats}")
    
    def _ensure_session(self) -> None:
        """Ensure a session is active."""
        if not self._session_active or not self._page:
            raise RuntimeError(
                "No active session. Use 'async with VintedScraper()' "
                "or call start_session() first."
            )
    
    # =========================================
    # Cookie Consent
    # =========================================
    
    async def handle_cookie_consent(self) -> bool:
        """
        Handle the cookie consent popup.
        
        Tries multiple selectors to find and click the accept button.
        
        Returns:
            True if consent was handled, False if no popup found.
        """
        self._ensure_session()
        
        logger.debug("Looking for cookie consent popup...")
        
        for selector in COOKIE_CONSENT_SELECTORS:
            try:
                # Wait briefly for the element
                button = self._page.locator(selector).first
                
                # Check if visible
                if await button.is_visible(timeout=2000):
                    logger.info(f"Cookie consent found with selector: {selector}")
                    await button.click()
                    await asyncio.sleep(0.5)  # Wait for popup to close
                    logger.info("Cookie consent accepted")
                    return True
                    
            except Exception:
                # Selector not found, try next
                continue
        
        logger.debug("No cookie consent popup found")
        return False
    
    # =========================================
    # Scrolling
    # =========================================
    
    async def scroll_for_lazy_loading(
        self,
        scroll_count: int = 5,
        scroll_delay: float = 1.0,
    ) -> int:
        """
        Scroll down the page to trigger lazy loading.
        
        Args:
            scroll_count: Number of times to scroll.
            scroll_delay: Delay between scrolls in seconds.
            
        Returns:
            Final scroll position in pixels.
        """
        self._ensure_session()
        
        logger.debug(f"Scrolling {scroll_count} times for lazy loading...")
        
        last_height = 0
        
        for i in range(scroll_count):
            # Scroll down
            await self._page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(scroll_delay)
            
            # Check new height
            new_height = await self._page.evaluate("document.body.scrollHeight")
            
            if new_height == last_height:
                logger.debug(f"Reached end of page after {i + 1} scrolls")
                break
            
            last_height = new_height
        
        # Scroll back to top
        await self._page.evaluate("window.scrollTo(0, 0)")
        
        logger.debug(f"Scrolling complete, page height: {last_height}px")
        return last_height
    
    # =========================================
    # Category Scraping
    # =========================================
    
    async def scrape_category(
        self,
        url: str,
        max_items: Optional[int] = None,
        scroll_count: int = 5,
        download_images: bool = False,
    ) -> List[ClothingItem]:
        """
        Scrape items from a category or search results page.
        
        Args:
            url: URL of the category or search page.
            max_items: Maximum number of items to scrape.
            scroll_count: Number of scroll operations for lazy loading.
            download_images: Whether to download item images.
            
        Returns:
            List of ClothingItem objects.
            
        Example:
            >>> items = await scraper.scrape_category(
            ...     "https://www.vinted.fr/catalog?catalog[]=5",
            ...     max_items=50,
            ...     download_images=True
            ... )
        """
        self._ensure_session()
        
        logger.info(f"Navigating to: {url}")
        
        # Wait for rate limiter
        await self.rate_limiter.wait()
        
        try:
            # Navigate to the page
            response = await self._page.goto(url, wait_until="domcontentloaded")
            
            if not response:
                logger.error("Navigation failed: no response")
                self.stats.errors += 1
                return []
            
            if response.status >= 400:
                logger.error(f"Navigation failed: HTTP {response.status}")
                self.stats.errors += 1
                self.rate_limiter.on_error()
                return []
            
            self.stats.pages_visited += 1
            self.rate_limiter.on_success()
            
            logger.info(f"Page loaded successfully (HTTP {response.status})")
            
            # Handle cookie consent on first visit
            await self.handle_cookie_consent()
            
            # Scroll to load more items
            await self.scroll_for_lazy_loading(scroll_count=scroll_count)
            
            # Parse the listing page
            items = await self.listing_parser.parse_listing_page(
                self._page,
                max_items=max_items,
            )
            
            logger.info(f"Parsed {len(items)} items from listing")
            
            # Download images if requested
            if download_images:
                for item in items:
                    if item.image_url:
                        local_path = await self.download_image(item.image_url)
                        if local_path:
                            item.local_image_path = local_path
            
            self.stats.items_scraped += len(items)
            
            return items
            
        except Exception as e:
            logger.error(f"Error scraping category: {e}")
            self.stats.errors += 1
            self.rate_limiter.on_error()
            return []
    
    async def scrape_item_details(
        self,
        item_url: str,
        download_images: bool = False,
    ) -> Optional[ClothingItem]:
        """
        Scrape detailed information from an item page.
        
        Args:
            item_url: URL of the item detail page.
            download_images: Whether to download item images.
            
        Returns:
            ClothingItem with full details, or None on error.
        """
        self._ensure_session()
        
        logger.info(f"Navigating to item: {item_url}")
        
        # Wait for rate limiter
        await self.rate_limiter.wait()
        
        try:
            # Navigate to the page
            response = await self._page.goto(item_url, wait_until="domcontentloaded")
            
            if not response or response.status >= 400:
                logger.error(f"Navigation failed: HTTP {response.status if response else 'no response'}")
                self.stats.errors += 1
                self.rate_limiter.on_error()
                return None
            
            self.stats.pages_visited += 1
            self.rate_limiter.on_success()
            
            # Handle cookie consent
            await self.handle_cookie_consent()
            
            # Extract item ID from URL
            item_id = self._extract_item_id(item_url)
            
            # Parse the item page
            item = await self.item_parser.parse_item_page(
                self._page,
                item_id=item_id,
            )
            
            if item and download_images and item.image_url:
                local_path = await self.download_image(item.image_url)
                if local_path:
                    item.local_image_path = local_path
            
            if item:
                self.stats.items_scraped += 1
            
            return item
            
        except Exception as e:
            logger.error(f"Error scraping item: {e}")
            self.stats.errors += 1
            self.rate_limiter.on_error()
            return None
    
    def _extract_item_id(self, url: str) -> str:
        """Extract item ID from Vinted URL."""
        # Pattern: /items/12345-item-name
        match = re.search(r"/items/(\d+)", url)
        if match:
            return match.group(1)
        return str(uuid.uuid4())[:8]
    
    # =========================================
    # Image Download
    # =========================================
    
    async def download_image(
        self,
        image_url: str,
        filename: Optional[str] = None,
    ) -> Optional[str]:
        """
        Download an image to the local cache.
        
        Uses aiohttp for efficient async downloading.
        
        Args:
            image_url: URL of the image to download.
            filename: Custom filename. If None, generates from URL hash.
            
        Returns:
            Local file path of the downloaded image, or None on error.
            
        Example:
            >>> path = await scraper.download_image(
            ...     "https://example.com/image.jpg"
            ... )
            >>> print(f"Downloaded to: {path}")
        """
        if not image_url:
            return None
        
        # Generate filename if not provided
        if not filename:
            # Create hash-based filename to avoid duplicates
            url_hash = hashlib.md5(image_url.encode()).hexdigest()[:12]
            # Try to get extension from URL
            ext = self._get_image_extension(image_url)
            filename = f"{url_hash}{ext}"
        
        filepath = self.images_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            logger.debug(f"Image already cached: {filename}")
            return str(filepath)
        
        logger.debug(f"Downloading image: {image_url[:80]}...")
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    image_url,
                    headers={"User-Agent": self.user_agent},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Failed to download image: HTTP {response.status}"
                        )
                        return None
                    
                    # Read and save image
                    content = await response.read()
                    
                    # Verify it's actually an image
                    content_type = response.headers.get("Content-Type", "")
                    if not content_type.startswith("image/"):
                        logger.warning(f"Not an image: {content_type}")
                        return None
                    
                    # Save to file
                    with open(filepath, "wb") as f:
                        f.write(content)
                    
                    self.stats.images_downloaded += 1
                    logger.info(f"Image downloaded: {filename}")
                    
                    return str(filepath)
                    
        except asyncio.TimeoutError:
            logger.warning(f"Timeout downloading image: {image_url[:50]}...")
            return None
        except Exception as e:
            logger.error(f"Error downloading image: {e}")
            return None
    
    def _get_image_extension(self, url: str) -> str:
        """Extract image extension from URL."""
        # Common patterns
        patterns = [
            r"\.jpg",
            r"\.jpeg", 
            r"\.png",
            r"\.webp",
            r"\.gif",
        ]
        
        url_lower = url.lower()
        for pattern in patterns:
            if re.search(pattern, url_lower):
                return pattern.replace("\\", "")
        
        # Default to jpg
        return ".jpg"
    
    # =========================================
    # Multi-page Scraping
    # =========================================
    
    async def scrape_category_pages(
        self,
        base_url: str,
        max_pages: Optional[int] = None,
        max_items_per_page: int = 50,
        download_images: bool = False,
    ) -> List[ClothingItem]:
        """
        Scrape multiple pages of a category.
        
        Args:
            base_url: Base category URL.
            max_pages: Maximum pages to scrape. If None, uses config.
            max_items_per_page: Max items to extract per page.
            download_images: Whether to download images.
            
        Returns:
            Combined list of all scraped items.
        """
        self._ensure_session()
        
        max_pages = max_pages or self.max_pages
        all_items: List[ClothingItem] = []
        
        logger.info(f"Scraping up to {max_pages} pages from: {base_url}")
        
        for page_num in range(1, max_pages + 1):
            # Build page URL
            if "?" in base_url:
                page_url = f"{base_url}&page={page_num}"
            else:
                page_url = f"{base_url}?page={page_num}"
            
            logger.info(f"Scraping page {page_num}/{max_pages}")
            
            # Scrape the page
            items = await self.scrape_category(
                url=page_url,
                max_items=max_items_per_page,
                download_images=download_images,
            )
            
            if not items:
                logger.info(f"No items found on page {page_num}, stopping")
                break
            
            all_items.extend(items)
            logger.info(
                f"Page {page_num}: {len(items)} items, total: {len(all_items)}"
            )
            
            # Check for duplicates (might indicate we've reached the end)
            if page_num > 1:
                ids = {item.id for item in all_items}
                if len(ids) < len(all_items) * 0.9:
                    logger.info("Detected duplicate items, likely reached end")
                    break
        
        logger.info(f"Category scraping complete: {len(all_items)} total items")
        return all_items


# =========================================
# Factory Function
# =========================================

def create_scraper(**kwargs) -> VintedScraper:
    """
    Factory function to create a VintedScraper.
    
    Args:
        **kwargs: Arguments passed to VintedScraper.__init__
        
    Returns:
        Configured VintedScraper instance.
    """
    return VintedScraper(**kwargs)

"""Playwright-based web scraper for Vinted.

This module implements asynchronous web scraping with rate limiting,
user-agent rotation, retry logic, and image downloading.
"""

import asyncio
import random
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import aiohttp
from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from tqdm.asyncio import tqdm

from src.utils import get_config, get_logger, log_exception, log_execution_time

from .exceptions import NavigationError, DownloadError
from .models import ScrapedBatch, VintedItem
from .parsers import VintedParser
from .utils import generate_batch_id, sanitize_filename

logger = get_logger(__name__)


class VintedScraper:
    """Asynchronous web scraper for Vinted fashion marketplace."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize Vinted scraper.
        
        Args:
            config_path: Optional path to config file
        """
        # Load configuration
        self.config = get_config(config_path)
        self.scraper_config = self.config.scraper
        
        # Initialize parser
        self.parser = VintedParser()
        
        # State tracking
        self.current_user_agent_index = 0
        self.scraped_urls = set()  # For deduplication
        
        # Browser instances (initialized in scrape methods)
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        
        logger.info("VintedScraper initialized")
    
    async def _create_context(self, browser: Browser) -> BrowserContext:
        """Create browser context with rotated user-agent.
        
        Args:
            browser: Browser instance
            
        Returns:
            BrowserContext with configured settings
        """
        user_agent = self._get_next_user_agent()
        
        context = await browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='fr-FR',
            timezone_id='Europe/Paris',
            extra_http_headers={
                'Accept-Language': 'fr-FR,fr;q=0.9',
            }
        )
        
        logger.debug(f"Created context with user-agent: {user_agent[:50]}...")
        return context
    
    def _get_next_user_agent(self) -> str:
        """Get next user-agent from rotation list.
        
        Returns:
            User-agent string
            
        Raises:
            ValueError: If user_agents list is empty (should be caught by config validation)
        """
        if not self.scraper_config.user_agents:
            raise ValueError("user_agents list is empty - this should be prevented by config validation")
        
        user_agent = self.scraper_config.user_agents[self.current_user_agent_index]
        self.current_user_agent_index = (self.current_user_agent_index + 1) % len(self.scraper_config.user_agents)
        
        return user_agent
    
    async def _random_delay(self) -> None:
        """Apply random delay for rate limiting."""
        min_delay, max_delay = self.scraper_config.delay_range
        delay = random.uniform(min_delay, max_delay)
        logger.debug(f"Applying delay: {delay:.2f}s")
        await asyncio.sleep(delay)
    
    async def _navigate_with_retry(
        self, page: Page, url: str, max_retries: int = 3
    ) -> str:
        """Navigate to URL with retry logic.
        
        Args:
            page: Page instance
            url: URL to navigate to
            max_retries: Maximum number of retry attempts
            
        Returns:
            Page HTML content
            
        Raises:
            NavigationError: If navigation fails after all retries
        """
        for attempt in range(max_retries):
            try:
                logger.debug(f"Navigating to {url} (attempt {attempt + 1}/{max_retries})")
                
                response = await page.goto(
                    url,
                    timeout=self.scraper_config.timeout * 1000,
                    wait_until='domcontentloaded'
                )
                
                if response and response.status >= 400:
                    raise NavigationError(f"HTTP {response.status} error")
                
                # Wait a bit for dynamic content
                await page.wait_for_timeout(1000)
                
                html = await page.content()
                return html
            
            except Exception as e:
                logger.warning(f"Navigation attempt {attempt + 1} failed: {e}")
                
                if attempt < max_retries - 1:
                    # Exponential backoff
                    backoff = 2 ** attempt
                    logger.debug(f"Retrying after {backoff}s backoff")
                    await asyncio.sleep(backoff)
                else:
                    log_exception(logger, f"navigate to {url}", e)
                    raise NavigationError(f"Failed to navigate after {max_retries} attempts") from e
    
    async def scrape_category(
        self, category: str, max_pages: Optional[int] = None
    ) -> ScrapedBatch:
        """Scrape items from a category search.
        
        Args:
            category: Search category/keyword
            max_pages: Maximum pages to scrape (overrides config)
            
        Returns:
            ScrapedBatch with all scraped items
        """
        max_pages = max_pages or self.scraper_config.max_pages
        batch_id = generate_batch_id()
        
        logger.info(f"Starting category scrape: '{category}' (max {max_pages} pages)")
        
        all_items = []
        pages_scraped = 0
        
        async with async_playwright() as playwright:
            # Launch browser
            self._browser = await playwright.chromium.launch(
                headless=self.scraper_config.headless
            )
            
            try:
                with log_execution_time(logger, f"scraping category '{category}'"):
                    # Create context
                    self._context = await self._create_context(self._browser)
                    page = await self._context.new_page()
                    
                    # Scrape pages with progress bar
                    for page_num in tqdm(range(1, max_pages + 1), desc="Scraping pages"):
                        try:
                            # Construct search URL
                            search_url = f"{self.scraper_config.base_url}/catalog?search_text={category}&page={page_num}"
                            
                            # Check if already scraped
                            if search_url in self.scraped_urls:
                                logger.debug(f"Skipping already scraped URL: {search_url}")
                                continue
                            
                            # Navigate and extract HTML
                            html = await self._navigate_with_retry(page, search_url)
                            self.scraped_urls.add(search_url)
                            
                            # Parse items
                            items = self.parser.parse_catalog_page(html, self.scraper_config.base_url)
                            
                            if not items:
                                logger.warning(f"No items found on page {page_num}, stopping")
                                break
                            
                            # Filter duplicates
                            new_items = [
                                item for item in items
                                if item.url not in self.scraped_urls
                            ]
                            
                            # Mark as scraped
                            for item in new_items:
                                self.scraped_urls.add(item.url)
                            
                            all_items.extend(new_items)
                            pages_scraped += 1
                            
                            logger.info(f"Page {page_num}: Found {len(new_items)} new items")
                            
                            # Rate limiting
                            await self._random_delay()
                        
                        except NavigationError as e:
                            logger.error(f"Failed to scrape page {page_num}: {e}")
                            continue
                        except Exception as e:
                            log_exception(logger, f"scrape page {page_num}", e)
                            continue
                    
                    await page.close()
            
            finally:
                # Cleanup
                if self._context:
                    await self._context.close()
                if self._browser:
                    await self._browser.close()
        
        # Create batch
        batch = ScrapedBatch(
            batch_id=batch_id,
            category=category,
            total_items=len(all_items),
            pages_scraped=pages_scraped,
            items=all_items
        )
        
        logger.info(f"Scraping complete: {len(all_items)} items from {pages_scraped} pages")
        return batch
    
    async def scrape_item_details(self, item_url: str) -> VintedItem:
        """Scrape detailed information from a product page.
        
        Args:
            item_url: URL of product page
            
        Returns:
            VintedItem with detailed information
        """
        logger.debug(f"Scraping item details: {item_url}")
        
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=self.scraper_config.headless)
            
            try:
                context = await self._create_context(browser)
                page = await context.new_page()
                
                # Navigate and extract
                html = await self._navigate_with_retry(page, item_url)
                item = self.parser.parse_product_page(html, item_url)
                
                await page.close()
                await context.close()
                
                return item
            
            finally:
                await browser.close()
    
    async def download_images(
        self, items: list[VintedItem], output_dir: Path, concurrency: int = 10
    ) -> None:
        """Download product images for scraped items with concurrent requests.
        
        Args:
            items: List of VintedItems to download images for
            output_dir: Base directory for images
            concurrency: Maximum concurrent downloads (default: 10)
        """
        logger.info(f"Downloading images for {len(items)} items (concurrency: {concurrency})")
        
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Create download tasks metadata
        download_tasks = []
        for item in items:
            for idx, image_url in enumerate(item.image_urls):
                if image_url:
                    download_tasks.append((item, idx, image_url))
        
        if not download_tasks:
            logger.info("No images to download")
            return
        
        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(concurrency)
        
        # Get user-agent for headers
        user_agent = self._get_next_user_agent()
        headers = {
            'User-Agent': user_agent,
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'fr-FR,fr;q=0.9',
        }
        
        # Create aiohttp session with headers
        async with aiohttp.ClientSession(headers=headers) as session:
            # Create coroutine wrapper for each download
            async def download_with_semaphore(item: VintedItem, idx: int, url: str):
                async with semaphore:
                    try:
                        await self._download_image(session, url, item.item_id, idx, images_dir)
                        
                        # Update item's local paths
                        local_path = str(images_dir / item.item_id / f"{idx}.jpg")
                        if local_path not in item.local_image_paths:
                            item.local_image_paths.append(local_path)
                        
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to download {url}: {e}")
                        return False
            
            # Create all download tasks
            tasks = [
                asyncio.create_task(download_with_semaphore(item, idx, url))
                for item, idx, url in download_tasks
            ]
            
            # Execute with progress bar
            successful = 0
            failed = 0
            
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Downloading images"
            ):
                result = await coro
                if result:
                    successful += 1
                else:
                    failed += 1
                
                # Apply light throttling every 20 images
                if (successful + failed) % 20 == 0:
                    await asyncio.sleep(0.1)
        
        logger.info(f"Image download complete: {successful} successful, {failed} failed")
    
    async def _download_image(
        self,
        session: aiohttp.ClientSession,
        url: str,
        item_id: str,
        index: int,
        output_dir: Path
    ) -> None:
        """Download a single image.
        
        Args:
            session: aiohttp session
            url: Image URL
            item_id: Item identifier
            index: Image index
            output_dir: Output directory
        """
        # Create item directory
        item_dir = output_dir / sanitize_filename(item_id)
        item_dir.mkdir(parents=True, exist_ok=True)
        
        # Download image
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                content = await response.read()
                
                # Save image
                image_path = item_dir / f"{index}.jpg"
                with open(image_path, 'wb') as f:
                    f.write(content)
                
                logger.debug(f"Downloaded image: {image_path}")
            else:
                raise DownloadError(f"HTTP {response.status}")
    
    async def close(self) -> None:
        """Close browser resources."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()

"""
Abstract interface for web scrapers.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Optional

from src.domain.entities.clothing_item import ClothingItem


class ScraperInterface(ABC):
    """
    Abstract base class for web scrapers.
    
    Defines the contract for scraping clothing items from marketplaces.
    """

    @abstractmethod
    async def scrape_category(
        self,
        category: str,
        max_pages: int = 10,
    ) -> AsyncGenerator[ClothingItem, None]:
        """
        Scrape items from a category.
        
        Args:
            category: Category identifier or URL path.
            max_pages: Maximum number of pages to scrape.
            
        Yields:
            ClothingItem objects as they are scraped.
        """
        pass

    @abstractmethod
    async def scrape_item_details(
        self,
        item_url: str,
    ) -> Optional[ClothingItem]:
        """
        Scrape detailed information from an item page.
        
        Args:
            item_url: Full URL to the item page.
            
        Returns:
            ClothingItem with full details, or None if failed.
        """
        pass

    @abstractmethod
    async def download_image(
        self,
        image_url: str,
        save_path: str,
    ) -> Optional[str]:
        """
        Download and save an image.
        
        Args:
            image_url: URL of the image to download.
            save_path: Local path to save the image.
            
        Returns:
            Local path if successful, None otherwise.
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources (browser, connections)."""
        pass

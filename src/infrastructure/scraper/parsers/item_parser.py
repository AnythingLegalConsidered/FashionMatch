"""
Item detail page parser for Vinted product pages.

Extracts complete item information from individual product pages,
including description, seller info, condition, and photos.

Example:
    >>> from playwright.async_api import Page
    >>> parser = ItemParser()
    >>> item = await parser.parse_item_page(page, item_id="12345")
    >>> print(f"{item.title} by {item.seller_id}")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional, TYPE_CHECKING

from src.domain.entities.clothing_item import ClothingItem
from src.utils.logger import get_logger

if TYPE_CHECKING:
    from playwright.async_api import Locator, Page

logger = get_logger(__name__)


# ============================================
# CSS Selectors Configuration
# ============================================
# TODO: Update selectors based on current Vinted DOM
# Vinted frequently changes their class names and structure.
# Use browser DevTools to inspect the current DOM and update these selectors.

@dataclass
class ItemSelectors:
    """
    CSS selectors for Vinted item detail pages.
    
    These selectors target elements on individual product pages
    where full item details are displayed.
    
    TODO: Update selectors based on current Vinted DOM
    Last verified: [DATE]
    """
    
    # Main content container
    main_content: str = "main.item-page"
    main_content_alt: str = "div[data-testid='item-page']"
    
    # Product title
    title: str = "h1.item-page__title"
    title_alt: str = "h1[itemprop='name']"
    
    # Product price
    price: str = "span.item-page__price"
    price_alt: str = "span[data-testid='item-price']"
    
    # Product description
    description: str = "div.item-page__description"
    description_alt: str = "div[itemprop='description']"
    
    # Brand name
    brand: str = "a.item-page__brand"
    brand_alt: str = "span[itemprop='brand']"
    
    # Size
    size: str = "span.item-page__size"
    size_alt: str = "div[data-testid='item-size'] span"
    
    # Condition/State
    condition: str = "span.item-page__condition"
    condition_alt: str = "div[data-testid='item-condition'] span"
    
    # Category
    category: str = "nav.breadcrumbs a"
    category_alt: str = "span[itemprop='category']"
    
    # Color
    color: str = "span.item-page__color"
    color_alt: str = "div[data-testid='item-color'] span"
    
    # Main product image
    main_image: str = "img.item-page__image"
    main_image_alt: str = "img[itemprop='image']"
    
    # All product images (gallery)
    gallery_images: str = "div.item-photos img"
    gallery_images_alt: str = "div[data-testid='item-gallery'] img"
    
    # Seller username
    seller_name: str = "a.item-page__seller-name"
    seller_name_alt: str = "span[data-testid='seller-username']"
    
    # Seller rating
    seller_rating: str = "span.item-page__seller-rating"
    
    # Number of views
    views_count: str = "span.item-page__views"
    
    # Number of favorites
    favorites_count: str = "span.item-page__favorites"
    
    # Posted date
    posted_date: str = "time.item-page__date"
    posted_date_alt: str = "span[data-testid='item-date']"
    
    # Item status (available, reserved, sold)
    status: str = "span.item-page__status"
    status_alt: str = "div[data-testid='item-status']"


# Default selectors instance
DEFAULT_SELECTORS = ItemSelectors()


@dataclass
class ParsedItemDetails:
    """
    Container for all parsed item details.
    
    Used internally before converting to ClothingItem.
    Allows for partial data when some fields fail to parse.
    """
    
    id: str
    title: Optional[str] = None
    price: Optional[float] = None
    currency: str = "EUR"
    description: Optional[str] = None
    brand: Optional[str] = None
    size: Optional[str] = None
    condition: Optional[str] = None
    category: Optional[str] = None
    color: Optional[str] = None
    image_url: Optional[str] = None
    all_image_urls: List[str] = None
    item_url: str = ""
    seller_id: Optional[str] = None
    seller_name: Optional[str] = None
    seller_rating: Optional[float] = None
    views: Optional[int] = None
    favorites: Optional[int] = None
    posted_date: Optional[str] = None
    status: Optional[str] = None
    
    def __post_init__(self):
        if self.all_image_urls is None:
            self.all_image_urls = []
    
    def to_clothing_item(self) -> ClothingItem:
        """Convert to ClothingItem entity."""
        return ClothingItem(
            id=self.id,
            title=self.title or f"Item {self.id}",
            price=self.price or 0.0,
            currency=self.currency,
            brand=self.brand,
            size=self.size,
            condition=self.condition,
            category=self.category,
            image_url=self.image_url or "",
            item_url=self.item_url,
            description=self.description,
            seller_id=self.seller_id or self.seller_name,
            scraped_at=datetime.utcnow(),
        )


class ItemParser:
    """
    Parser for Vinted item detail pages.
    
    Extracts complete item information including description,
    seller info, condition, and all photos.
    
    The parser is designed to be resilient - it will return partial
    data rather than failing if some elements are not found.
    
    Attributes:
        selectors: CSS selectors configuration.
        
    Example:
        >>> parser = ItemParser()
        >>> item = await parser.parse_item_page(page, "12345")
        >>> if item:
        ...     print(item.description)
    """
    
    def __init__(self, selectors: Optional[ItemSelectors] = None):
        """
        Initialize the item parser.
        
        Args:
            selectors: Custom CSS selectors. Uses defaults if None.
        """
        self.selectors = selectors or DEFAULT_SELECTORS
        logger.debug("ItemParser initialized")
    
    async def parse_item_page(
        self,
        page: "Page",
        item_id: str,
        item_url: Optional[str] = None,
    ) -> Optional[ClothingItem]:
        """
        Parse complete item details from an item page.
        
        Args:
            page: Playwright Page object with loaded item page.
            item_id: The item ID (for fallback data).
            item_url: The item URL (optional, extracted from page if None).
            
        Returns:
            ClothingItem with full details, or None if parsing failed.
            
        Example:
            >>> item = await parser.parse_item_page(page, "12345678")
            >>> print(f"Description: {item.description[:100]}...")
        """
        try:
            # Wait for main content to load
            await self._wait_for_content(page)
            
            # Extract all details
            details = ParsedItemDetails(id=item_id)
            details.item_url = item_url or page.url
            
            # Parse each field
            details.title = await self._extract_title(page)
            details.price = await self._extract_price(page)
            details.description = await self._extract_description(page)
            details.brand = await self._extract_brand(page)
            details.size = await self._extract_size(page)
            details.condition = await self._extract_condition(page)
            details.category = await self._extract_category(page)
            details.image_url = await self._extract_main_image(page)
            details.all_image_urls = await self._extract_all_images(page)
            details.seller_name = await self._extract_seller_name(page)
            details.seller_id = details.seller_name  # Use name as ID for now
            details.status = await self._extract_status(page)
            
            logger.info(
                f"Parsed item {item_id}: {details.title or 'No title'} "
                f"- {details.price or 0}€"
            )
            
            return details.to_clothing_item()
            
        except Exception as e:
            logger.error(f"Failed to parse item page {item_id}: {e}")
            return None
    
    async def _wait_for_content(self, page: "Page") -> None:
        """Wait for main content to load."""
        try:
            await page.wait_for_selector(
                self.selectors.main_content,
                timeout=10000,
            )
        except Exception:
            # Try alternative selector
            try:
                await page.wait_for_selector(
                    self.selectors.title,
                    timeout=5000,
                )
            except Exception:
                logger.warning("Could not detect item page content loaded")
    
    async def _extract_text(
        self,
        page: "Page",
        *selectors: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """
        Extract text content using multiple selectors.
        
        Tries each selector in order until one succeeds.
        Returns None or default if all selectors fail.
        """
        for selector in selectors:
            try:
                element = page.locator(selector)
                if await element.count() > 0:
                    text = await element.first.text_content()
                    if text:
                        return text.strip()
            except Exception:
                continue
        
        return default
    
    async def _extract_attribute(
        self,
        page: "Page",
        selector: str,
        attribute: str,
        default: Optional[str] = None,
    ) -> Optional[str]:
        """Extract an attribute from an element."""
        try:
            element = page.locator(selector)
            if await element.count() > 0:
                value = await element.first.get_attribute(attribute)
                return value if value else default
        except Exception:
            pass
        return default
    
    async def _extract_title(self, page: "Page") -> Optional[str]:
        """Extract product title."""
        return await self._extract_text(
            page,
            self.selectors.title,
            self.selectors.title_alt,
            "h1",  # Generic fallback
        )
    
    async def _extract_price(self, page: "Page") -> Optional[float]:
        """Extract and parse product price."""
        text = await self._extract_text(
            page,
            self.selectors.price,
            self.selectors.price_alt,
        )
        
        if text:
            return self._parse_price(text)
        return None
    
    async def _extract_description(self, page: "Page") -> Optional[str]:
        """Extract product description."""
        description = await self._extract_text(
            page,
            self.selectors.description,
            self.selectors.description_alt,
        )
        
        if description:
            # Clean up whitespace
            import re
            description = re.sub(r"\s+", " ", description).strip()
        
        return description
    
    async def _extract_brand(self, page: "Page") -> Optional[str]:
        """Extract brand name."""
        return await self._extract_text(
            page,
            self.selectors.brand,
            self.selectors.brand_alt,
        )
    
    async def _extract_size(self, page: "Page") -> Optional[str]:
        """Extract size information."""
        return await self._extract_text(
            page,
            self.selectors.size,
            self.selectors.size_alt,
        )
    
    async def _extract_condition(self, page: "Page") -> Optional[str]:
        """Extract item condition/state."""
        return await self._extract_text(
            page,
            self.selectors.condition,
            self.selectors.condition_alt,
        )
    
    async def _extract_category(self, page: "Page") -> Optional[str]:
        """Extract category from breadcrumbs."""
        try:
            # Get all breadcrumb links
            breadcrumbs = page.locator(self.selectors.category)
            count = await breadcrumbs.count()
            
            if count > 0:
                # Get the last meaningful category (usually second-to-last)
                categories = []
                for i in range(count):
                    text = await breadcrumbs.nth(i).text_content()
                    if text:
                        categories.append(text.strip())
                
                # Return last category (most specific)
                if categories:
                    return categories[-1]
        except Exception:
            pass
        
        return await self._extract_text(
            page,
            self.selectors.category_alt,
        )
    
    async def _extract_main_image(self, page: "Page") -> Optional[str]:
        """Extract main product image URL."""
        for selector in [
            self.selectors.main_image,
            self.selectors.main_image_alt,
            "img[itemprop='image']",
            "div.item-photos img",
        ]:
            try:
                img = page.locator(selector)
                if await img.count() > 0:
                    # Try src
                    src = await img.first.get_attribute("src")
                    if src and not src.startswith("data:"):
                        return self._get_high_res_url(src)
                    
                    # Try data-src
                    data_src = await img.first.get_attribute("data-src")
                    if data_src:
                        return self._get_high_res_url(data_src)
            except Exception:
                continue
        
        return None
    
    async def _extract_all_images(self, page: "Page") -> List[str]:
        """Extract all product image URLs from gallery."""
        images: List[str] = []
        
        for selector in [
            self.selectors.gallery_images,
            self.selectors.gallery_images_alt,
            "div.item-photos img",
            "div[data-testid='gallery'] img",
        ]:
            try:
                img_elements = page.locator(selector)
                count = await img_elements.count()
                
                for i in range(count):
                    img = img_elements.nth(i)
                    
                    # Try different attributes
                    for attr in ["src", "data-src", "data-original"]:
                        url = await img.get_attribute(attr)
                        if url and not url.startswith("data:"):
                            high_res = self._get_high_res_url(url)
                            if high_res not in images:
                                images.append(high_res)
                            break
                
                if images:
                    break
                    
            except Exception:
                continue
        
        logger.debug(f"Found {len(images)} gallery images")
        return images
    
    async def _extract_seller_name(self, page: "Page") -> Optional[str]:
        """Extract seller username."""
        return await self._extract_text(
            page,
            self.selectors.seller_name,
            self.selectors.seller_name_alt,
        )
    
    async def _extract_status(self, page: "Page") -> Optional[str]:
        """Extract item status (available/sold/reserved)."""
        status = await self._extract_text(
            page,
            self.selectors.status,
            self.selectors.status_alt,
        )
        
        if status:
            status_lower = status.lower()
            if "vendu" in status_lower or "sold" in status_lower:
                return "sold"
            elif "réservé" in status_lower or "reserved" in status_lower:
                return "reserved"
        
        return "available"
    
    def _parse_price(self, text: str) -> float:
        """Parse price string to float."""
        import re
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r"[^\d,.]", "", text)
        
        # Handle European format
        if "," in cleaned and "." in cleaned:
            if cleaned.index(",") > cleaned.index("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            parts = cleaned.split(",")
            if len(parts[-1]) <= 2:
                cleaned = cleaned.replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Failed to parse price: {text}")
            return 0.0
    
    def _get_high_res_url(self, url: str) -> str:
        """
        Convert image URL to high resolution version.
        
        Vinted uses URL parameters for image sizing.
        """
        import re
        
        # Remove size parameters to get original image
        # Pattern: /f800/ or ?w=400 etc.
        url = re.sub(r"/f\d+/", "/f800/", url)
        url = re.sub(r"\?.*$", "", url)
        
        return url

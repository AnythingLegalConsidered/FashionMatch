"""
Listing page parser for Vinted product grids.

Extracts basic item information from category/search result pages.
CSS selectors are isolated here for easy maintenance when Vinted updates their DOM.

Example:
    >>> from playwright.async_api import Page
    >>> parser = ListingParser()
    >>> items = await parser.parse_listing_page(page)
    >>> for item in items:
    ...     print(f"{item.title}: {item.price}€")
"""

from __future__ import annotations

from dataclasses import dataclass
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
class ListingSelectors:
    """
    CSS selectors for Vinted listing pages.
    
    These selectors target elements in the product grid on category
    and search result pages.
    
    TODO: Update selectors based on current Vinted DOM
    Last verified: [DATE]
    """
    
    # Container for all product items in the grid
    # Look for: div containing multiple product cards
    item_grid: str = "div.feed-grid"
    
    # Individual product card in the listing
    # Look for: article or div wrapping each product
    item_card: str = "div.feed-grid__item"
    
    # Alternative selectors (try these if main ones fail)
    item_card_alt: str = "article.product-item"
    
    # Link to product detail page (usually wraps the whole card or image)
    item_link: str = "a.new-item-box__overlay"
    item_link_alt: str = "a[href*='/items/']"
    
    # Product title
    title: str = "h3.new-item-box__title"
    title_alt: str = "span.web_ui__Text__text"
    
    # Product price
    price: str = "p.new-item-box__price"
    price_alt: str = "span[data-testid='price-text']"
    
    # Product image
    image: str = "img.new-item-box__image"
    image_alt: str = "img[src*='images']"
    
    # Size information
    size: str = "span.new-item-box__size"
    size_alt: str = "span[data-testid='size-text']"
    
    # Brand name
    brand: str = "span.new-item-box__brand"
    brand_alt: str = "span[data-testid='brand-text']"
    
    # Favorite/like button (useful to detect item cards)
    favorite_button: str = "button[data-testid='favourite-button']"


# Default selectors instance
DEFAULT_SELECTORS = ListingSelectors()


class ListingParser:
    """
    Parser for Vinted listing/search result pages.
    
    Extracts basic item information (title, price, image, URL) from
    product cards displayed in category listings or search results.
    
    The parser is designed to be resilient to missing elements -
    it will return partial data rather than failing completely.
    
    Attributes:
        selectors: CSS selectors configuration.
        
    Example:
        >>> parser = ListingParser()
        >>> items = await parser.parse_listing_page(page)
        >>> print(f"Found {len(items)} items")
    """
    
    def __init__(self, selectors: Optional[ListingSelectors] = None):
        """
        Initialize the listing parser.
        
        Args:
            selectors: Custom CSS selectors. Uses defaults if None.
        """
        self.selectors = selectors or DEFAULT_SELECTORS
        logger.debug("ListingParser initialized")
    
    async def parse_listing_page(
        self,
        page: "Page",
        max_items: Optional[int] = None,
    ) -> List[ClothingItem]:
        """
        Parse all product items from a listing page.
        
        Args:
            page: Playwright Page object with loaded listing.
            max_items: Maximum number of items to parse. None = all.
            
        Returns:
            List of ClothingItem objects with basic info.
            
        Example:
            >>> items = await parser.parse_listing_page(page, max_items=50)
        """
        items: List[ClothingItem] = []
        
        try:
            # Wait for grid to load
            await page.wait_for_selector(
                self.selectors.item_grid,
                timeout=10000,
            )
        except Exception:
            logger.warning("Item grid not found, trying alternative selectors")
            try:
                await page.wait_for_selector(
                    self.selectors.item_card,
                    timeout=5000,
                )
            except Exception:
                logger.error("No product items found on page")
                return items
        
        # Find all item cards
        item_cards = await self._find_item_cards(page)
        logger.info(f"Found {len(item_cards)} item cards on page")
        
        # Limit items if specified
        if max_items:
            item_cards = item_cards[:max_items]
        
        # Parse each card
        for i, card in enumerate(item_cards):
            try:
                item = await self.parse_item_card(card, index=i)
                if item:
                    items.append(item)
            except Exception as e:
                logger.warning(f"Failed to parse item card {i}: {e}")
                continue
        
        logger.info(f"Successfully parsed {len(items)} items")
        return items
    
    async def _find_item_cards(self, page: "Page") -> List["Locator"]:
        """
        Find all product cards on the page.
        
        Tries multiple selectors for resilience.
        """
        # Try primary selector
        cards = page.locator(self.selectors.item_card)
        count = await cards.count()
        
        if count > 0:
            return [cards.nth(i) for i in range(count)]
        
        # Try alternative selector
        logger.debug("Primary selector found 0 items, trying alternative")
        cards = page.locator(self.selectors.item_card_alt)
        count = await cards.count()
        
        if count > 0:
            return [cards.nth(i) for i in range(count)]
        
        # Try finding by link pattern
        logger.debug("Trying link pattern selector")
        cards = page.locator(self.selectors.item_link_alt)
        count = await cards.count()
        
        return [cards.nth(i) for i in range(count)]
    
    async def parse_item_card(
        self,
        card: "Locator",
        index: int = 0,
    ) -> Optional[ClothingItem]:
        """
        Parse a single product card element.
        
        Args:
            card: Playwright Locator for the item card.
            index: Index of the card (for logging/ID generation).
            
        Returns:
            ClothingItem with extracted data, or None if parsing failed.
        """
        try:
            # Extract item URL (required)
            item_url = await self._extract_url(card)
            if not item_url:
                logger.debug(f"Card {index}: No URL found, skipping")
                return None
            
            # Extract item ID from URL
            item_id = self._extract_id_from_url(item_url)
            if not item_id:
                item_id = f"temp_{index}_{hash(item_url) % 10000}"
            
            # Extract other fields (optional)
            title = await self._extract_title(card)
            price = await self._extract_price(card)
            image_url = await self._extract_image_url(card)
            size = await self._extract_size(card)
            brand = await self._extract_brand(card)
            
            # Create ClothingItem
            return ClothingItem(
                id=item_id,
                title=title or f"Item {item_id}",
                price=price or 0.0,
                image_url=image_url or "",
                item_url=item_url,
                size=size,
                brand=brand,
            )
            
        except Exception as e:
            logger.error(f"Failed to parse card {index}: {e}")
            return None
    
    async def _extract_url(self, card: "Locator") -> Optional[str]:
        """Extract item URL from card."""
        # Try primary selector
        link = card.locator(self.selectors.item_link)
        if await link.count() > 0:
            href = await link.first.get_attribute("href")
            if href:
                return self._normalize_url(href)
        
        # Try alternative selector
        link = card.locator(self.selectors.item_link_alt)
        if await link.count() > 0:
            href = await link.first.get_attribute("href")
            if href:
                return self._normalize_url(href)
        
        # Try finding any link with /items/
        link = card.locator("a[href*='/items/']")
        if await link.count() > 0:
            href = await link.first.get_attribute("href")
            if href:
                return self._normalize_url(href)
        
        return None
    
    async def _extract_title(self, card: "Locator") -> Optional[str]:
        """Extract product title from card."""
        for selector in [self.selectors.title, self.selectors.title_alt]:
            element = card.locator(selector)
            if await element.count() > 0:
                text = await element.first.text_content()
                if text:
                    return text.strip()
        
        # Fallback: try img alt text
        img = card.locator("img")
        if await img.count() > 0:
            alt = await img.first.get_attribute("alt")
            if alt:
                return alt.strip()
        
        return None
    
    async def _extract_price(self, card: "Locator") -> Optional[float]:
        """Extract and parse price from card."""
        for selector in [self.selectors.price, self.selectors.price_alt]:
            element = card.locator(selector)
            if await element.count() > 0:
                text = await element.first.text_content()
                if text:
                    return self._parse_price(text)
        
        return None
    
    async def _extract_image_url(self, card: "Locator") -> Optional[str]:
        """Extract product image URL from card."""
        for selector in [self.selectors.image, self.selectors.image_alt, "img"]:
            img = card.locator(selector)
            if await img.count() > 0:
                # Try src first
                src = await img.first.get_attribute("src")
                if src and not src.startswith("data:"):
                    return src
                
                # Try data-src (lazy loading)
                data_src = await img.first.get_attribute("data-src")
                if data_src:
                    return data_src
                
                # Try srcset
                srcset = await img.first.get_attribute("srcset")
                if srcset:
                    # Get first URL from srcset
                    first_src = srcset.split(",")[0].split()[0]
                    if first_src:
                        return first_src
        
        return None
    
    async def _extract_size(self, card: "Locator") -> Optional[str]:
        """Extract size information from card."""
        for selector in [self.selectors.size, self.selectors.size_alt]:
            element = card.locator(selector)
            if await element.count() > 0:
                text = await element.first.text_content()
                if text:
                    return text.strip()
        
        return None
    
    async def _extract_brand(self, card: "Locator") -> Optional[str]:
        """Extract brand name from card."""
        for selector in [self.selectors.brand, self.selectors.brand_alt]:
            element = card.locator(selector)
            if await element.count() > 0:
                text = await element.first.text_content()
                if text:
                    return text.strip()
        
        return None
    
    def _normalize_url(self, url: str) -> str:
        """Normalize URL to absolute path."""
        if url.startswith("//"):
            return f"https:{url}"
        elif url.startswith("/"):
            return f"https://www.vinted.fr{url}"
        elif not url.startswith("http"):
            return f"https://www.vinted.fr/{url}"
        return url
    
    def _extract_id_from_url(self, url: str) -> Optional[str]:
        """Extract item ID from URL."""
        # Pattern: /items/123456-title-slug
        import re
        match = re.search(r"/items/(\d+)", url)
        if match:
            return match.group(1)
        return None
    
    def _parse_price(self, text: str) -> float:
        """Parse price string to float."""
        import re
        
        # Remove currency symbols and whitespace
        cleaned = re.sub(r"[^\d,.]", "", text)
        
        # Handle European format (1.234,56 -> 1234.56)
        if "," in cleaned and "." in cleaned:
            if cleaned.index(",") > cleaned.index("."):
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            # Could be 1,50 (decimal) or 1,500 (thousands)
            parts = cleaned.split(",")
            if len(parts[-1]) == 2:  # Decimal separator
                cleaned = cleaned.replace(",", ".")
            else:  # Thousands separator
                cleaned = cleaned.replace(",", "")
        
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Failed to parse price: {text}")
            return 0.0

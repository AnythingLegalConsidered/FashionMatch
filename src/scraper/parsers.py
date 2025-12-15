"""HTML/JSON parsing utilities for Vinted pages.

This module implements multiple parsing strategies for extracting product data:
1. JSON-LD structured data extraction
2. Embedded JavaScript JSON extraction
3. CSS selector-based HTML parsing
"""

import json
import re
from typing import Optional

from bs4 import BeautifulSoup

from src.utils import get_logger

from .exceptions import ParsingError
from .models import VintedItem
from .utils import extract_item_id_from_url, format_price

logger = get_logger(__name__)


class VintedParser:
    """Parser for extracting structured data from Vinted HTML pages."""
    
    # CSS selectors for fallback parsing
    SELECTORS = {
        'catalog': {
            'items': ['.feed-grid__item', '[data-testid="item-box"]', '.item-box'],
            'title': ['.item-title', '[data-testid="item-title"]', '.item__title'],
            'price': ['.item-price', '[data-testid="item-price"]', '.item__price'],
            'image': ['.item-photo img', '[data-testid="item-photo"] img', 'img.item__image'],
            'url': ['a.item-box', 'a[data-testid="item-box"]', 'a.item__link'],
        },
        'product': {
            'title': ['.item-title', '[data-testid="item-title"]', 'h1.item-title'],
            'price': ['.item-price', '[data-testid="item-price"]', '.price-text'],
            'description': ['.item-description', '[data-testid="item-description"]', '.description-text'],
            'brand': ['.item-brand', '[data-testid="item-brand"]', '.brand-name'],
            'size': ['.item-size', '[data-testid="item-size"]', '.size-value'],
            'condition': ['.item-condition', '[data-testid="item-condition"]', '.condition-value'],
            'images': ['.item-photos img', '[data-testid="item-photo"] img', '.photo-item img'],
        }
    }
    
    def parse_catalog_page(self, html: str, base_url: str = "https://www.vinted.fr") -> list[VintedItem]:
        """Extract items from catalog listing page.
        
        Args:
            html: HTML content of catalog page
            base_url: Base URL for constructing absolute URLs
            
        Returns:
            List of VintedItem objects
        """
        soup = BeautifulSoup(html, 'lxml')
        items = []
        
        # Try JSON-LD extraction first
        try:
            json_ld_items = self._extract_from_json_ld(soup)
            if json_ld_items:
                items.extend(json_ld_items)
                logger.debug(f"Extracted {len(json_ld_items)} items from JSON-LD")
                return items
        except Exception as e:
            logger.warning(f"JSON-LD extraction failed: {e}")
        
        # Try embedded JSON extraction
        try:
            embedded_items = self._extract_from_embedded_json(html)
            if embedded_items:
                items.extend(embedded_items)
                logger.debug(f"Extracted {len(embedded_items)} items from embedded JSON")
                return items
        except Exception as e:
            logger.warning(f"Embedded JSON extraction failed: {e}")
        
        # Fallback to CSS selectors
        try:
            selector_items = self._extract_from_selectors(soup, 'catalog', base_url)
            items.extend(selector_items)
            logger.debug(f"Extracted {len(selector_items)} items from CSS selectors")
        except Exception as e:
            logger.error(f"CSS selector extraction failed: {e}")
            raise ParsingError(f"All parsing strategies failed: {e}")
        
        return items
    
    def parse_product_page(self, html: str, url: str) -> VintedItem:
        """Extract detailed item from product page.
        
        Args:
            html: HTML content of product page
            url: Product page URL
            
        Returns:
            VintedItem with detailed information
        """
        soup = BeautifulSoup(html, 'lxml')
        
        # Try JSON-LD first
        try:
            item_data = self._extract_product_json_ld(soup)
            if item_data:
                return self._create_item_from_dict(item_data, url)
        except Exception as e:
            logger.warning(f"Product JSON-LD extraction failed: {e}")
        
        # Try embedded JSON
        try:
            item_data = self._extract_product_embedded_json(html)
            if item_data:
                return self._create_item_from_dict(item_data, url)
        except Exception as e:
            logger.warning(f"Product embedded JSON extraction failed: {e}")
        
        # Fallback to selectors
        try:
            item_data = self._extract_product_from_selectors(soup)
            return self._create_item_from_dict(item_data, url)
        except Exception as e:
            raise ParsingError(f"Failed to parse product page: {e}")
    
    def _extract_from_json_ld(self, soup: BeautifulSoup) -> list[VintedItem]:
        """Extract items from JSON-LD structured data."""
        items = []
        
        # Find all JSON-LD script tags
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                
                # Handle ItemList
                if isinstance(data, dict) and data.get('@type') == 'ItemList':
                    for item in data.get('itemListElement', []):
                        item_data = self._parse_json_ld_item(item)
                        if item_data:
                            items.append(item_data)
                
                # Handle single Product
                elif isinstance(data, dict) and data.get('@type') == 'Product':
                    item_data = self._parse_json_ld_product(data)
                    if item_data:
                        items.append(item_data)
            
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON-LD: {e}")
                continue
        
        return items
    
    def _extract_from_embedded_json(self, html: str) -> list[VintedItem]:
        """Extract items from embedded JavaScript JSON."""
        items = []
        
        # Search for common patterns
        patterns = [
            r'window\.__INITIAL_STATE__\s*=\s*({.+?});',
            r'window\.__PRELOADED_STATE__\s*=\s*({.+?});',
            r'var\s+initialData\s*=\s*({.+?});',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, html, re.DOTALL)
            for match in matches:
                try:
                    data = json.loads(match.group(1))
                    # Navigate nested structure to find items
                    extracted = self._extract_items_from_nested_json(data)
                    items.extend(extracted)
                except json.JSONDecodeError:
                    continue
        
        return items
    
    def _extract_from_selectors(self, soup: BeautifulSoup, page_type: str, base_url: str) -> list[VintedItem]:
        """Extract items using CSS selectors (fallback method)."""
        items = []
        selectors = self.SELECTORS[page_type]
        
        # Find item containers
        item_elements = []
        for selector in selectors['items']:
            item_elements = soup.select(selector)
            if item_elements:
                break
        
        if not item_elements:
            logger.warning("No item containers found with any selector")
            return items
        
        for elem in item_elements:
            try:
                # Extract title
                title = self._extract_with_fallback(elem, selectors['title'])
                if not title:
                    continue
                
                # Extract price
                price_str = self._extract_with_fallback(elem, selectors['price'])
                try:
                    price = format_price(price_str) if price_str else 0.0
                except ValueError:
                    price = 0.0
                
                # Extract image
                image_url = None
                for img_selector in selectors['image']:
                    img_elem = elem.select_one(img_selector)
                    if img_elem:
                        image_url = img_elem.get('src') or img_elem.get('data-src')
                        break
                
                if not image_url:
                    continue
                
                # Extract URL
                url = None
                for url_selector in selectors['url']:
                    url_elem = elem.select_one(url_selector)
                    if url_elem:
                        url = url_elem.get('href')
                        if url and not url.startswith('http'):
                            url = base_url + url
                        break
                
                if not url:
                    continue
                
                # Extract item ID from URL
                try:
                    item_id = extract_item_id_from_url(url)
                except ValueError:
                    logger.warning(f"Could not extract item ID from URL: {url}")
                    continue
                
                item = VintedItem(
                    item_id=item_id,
                    title=title,
                    price=price,
                    image_urls=[image_url],
                    url=url
                )
                items.append(item)
            
            except Exception as e:
                logger.warning(f"Failed to parse item element: {e}")
                continue
        
        return items
    
    def _extract_with_fallback(self, elem, selectors: list[str]) -> Optional[str]:
        """Try multiple selectors and return first match."""
        for selector in selectors:
            found = elem.select_one(selector)
            if found:
                return found.get_text(strip=True)
        return None
    
    def _parse_json_ld_item(self, item: dict) -> Optional[VintedItem]:
        """Parse JSON-LD ItemList element."""
        try:
            # Handle nested 'item' key
            if 'item' in item:
                item = item['item']
            
            url = item.get('url', '')
            if not url:
                return None
            
            item_id = extract_item_id_from_url(url)
            
            # Extract offers data
            offers = item.get('offers', {})
            price = float(offers.get('price', 0.0))
            currency = offers.get('priceCurrency', 'EUR')
            
            return VintedItem(
                item_id=item_id,
                title=item.get('name', 'Untitled'),
                price=price,
                currency=currency,
                description=item.get('description'),
                image_urls=[item.get('image', '')],
                url=url,
                brand=item.get('brand', {}).get('name') if isinstance(item.get('brand'), dict) else None
            )
        except Exception as e:
            logger.debug(f"Failed to parse JSON-LD item: {e}")
            return None
    
    def _parse_json_ld_product(self, data: dict) -> Optional[VintedItem]:
        """Parse JSON-LD Product data."""
        return self._parse_json_ld_item(data)
    
    def _extract_product_json_ld(self, soup: BeautifulSoup) -> Optional[dict]:
        """Extract product data from JSON-LD."""
        json_ld_scripts = soup.find_all('script', type='application/ld+json')
        
        for script in json_ld_scripts:
            try:
                data = json.loads(script.string)
                if isinstance(data, dict) and data.get('@type') == 'Product':
                    return data
            except json.JSONDecodeError:
                continue
        
        return None
    
    def _extract_product_embedded_json(self, html: str) -> Optional[dict]:
        """Extract product data from embedded JSON."""
        # Similar to catalog, but look for single product data
        patterns = [
            r'window\.__INITIAL_STATE__\s*=\s*({.+?});',
            r'window\.__PRELOADED_STATE__\s*=\s*({.+?});',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    # Navigate to product data
                    if 'item' in data:
                        return data['item']
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _extract_product_from_selectors(self, soup: BeautifulSoup) -> dict:
        """Extract product data using CSS selectors."""
        selectors = self.SELECTORS['product']
        
        title = self._extract_with_fallback(soup, selectors['title'])
        price_str = self._extract_with_fallback(soup, selectors['price'])
        description = self._extract_with_fallback(soup, selectors['description'])
        brand = self._extract_with_fallback(soup, selectors['brand'])
        size = self._extract_with_fallback(soup, selectors['size'])
        condition = self._extract_with_fallback(soup, selectors['condition'])
        
        # Extract all images
        image_urls = []
        for img_selector in selectors['images']:
            img_elems = soup.select(img_selector)
            for img in img_elems:
                img_url = img.get('src') or img.get('data-src')
                if img_url and img_url not in image_urls:
                    image_urls.append(img_url)
        
        return {
            'title': title or 'Untitled',
            'price': format_price(price_str) if price_str else 0.0,
            'description': description,
            'brand': brand,
            'size': size,
            'condition': condition,
            'image_urls': image_urls if image_urls else [''],
        }
    
    def _extract_items_from_nested_json(self, data: dict, max_depth: int = 10) -> list[VintedItem]:
        """Recursively search for item arrays in nested JSON."""
        items = []
        
        if max_depth <= 0:
            return items
        
        # Look for common keys that might contain item lists
        item_keys = ['items', 'products', 'results', 'data', 'catalogItems']
        
        for key in item_keys:
            if key in data and isinstance(data[key], list):
                for item_data in data[key]:
                    if isinstance(item_data, dict):
                        try:
                            # Try to create item from this data
                            if 'id' in item_data or 'item_id' in item_data:
                                item = self._create_item_from_dict(item_data)
                                if item:
                                    items.append(item)
                        except Exception:
                            continue
        
        # Recurse into nested dictionaries
        for value in data.values():
            if isinstance(value, dict):
                nested_items = self._extract_items_from_nested_json(value, max_depth - 1)
                items.extend(nested_items)
        
        return items
    
    def _create_item_from_dict(self, data: dict, url: str = None) -> VintedItem:
        """Create VintedItem from dictionary data."""
        # Extract item ID
        item_id = str(data.get('id') or data.get('item_id') or data.get('itemId', ''))
        if not item_id and url:
            item_id = extract_item_id_from_url(url)
        
        # Extract URL
        item_url = url or data.get('url') or data.get('link') or ''
        
        # Extract image URLs
        image_urls = data.get('image_urls') or data.get('images') or data.get('photos') or []
        if isinstance(image_urls, str):
            image_urls = [image_urls]
        elif isinstance(image_urls, list):
            # Handle nested image objects
            image_urls = [
                img.get('url') if isinstance(img, dict) else str(img)
                for img in image_urls
            ]
        
        if not image_urls:
            image_urls = ['']
        
        return VintedItem(
            item_id=item_id,
            title=data.get('title') or data.get('name') or 'Untitled',
            price=float(data.get('price', 0.0)),
            currency=data.get('currency', 'EUR'),
            description=data.get('description'),
            brand=data.get('brand'),
            size=data.get('size'),
            condition=data.get('condition'),
            image_urls=image_urls,
            url=item_url,
            seller_id=str(data.get('seller_id') or data.get('user_id') or '') or None,
            category=data.get('category')
        )

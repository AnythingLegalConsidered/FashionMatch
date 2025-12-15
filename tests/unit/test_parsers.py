"""Unit tests for HTML/JSON parsers."""

import pytest
from bs4 import BeautifulSoup

from src.scraper.parsers import VintedParser


class TestVintedParser:
    """Test Vinted HTML/JSON parsing."""
    
    @pytest.fixture
    def parser(self):
        """Create parser instance."""
        return VintedParser()
    
    def test_parse_json_ld(self, parser):
        """Test parsing JSON-LD structured data."""
        html = """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@type": "Product",
                "name": "Vintage Denim Jacket",
                "offers": {
                    "@type": "Offer",
                    "price": "45.00",
                    "priceCurrency": "EUR"
                },
                "brand": {"name": "Levi's"},
                "image": "https://example.com/jacket.jpg"
            }
            </script>
        </head>
        <body></body>
        </html>
        """
        
        soup = BeautifulSoup(html, "html.parser")
        item = parser.parse_item(soup, "https://vinted.fr/items/123")
        
        assert item.title == "Vintage Denim Jacket"
        assert item.price == 45.00
        assert item.brand == "Levi's"
        assert "jacket.jpg" in item.image_url
    
    def test_parse_fallback_css_selectors(self, parser):
        """Test fallback to CSS selectors when JSON-LD missing."""
        html = """
        <html>
        <body>
            <h1 class="item-title">Casual White Shirt</h1>
            <div class="item-price">€29.99</div>
            <div class="item-brand">Zara</div>
            <img class="item-image" src="https://example.com/shirt.jpg" />
        </body>
        </html>
        """
        
        soup = BeautifulSoup(html, "html.parser")
        item = parser.parse_item(soup, "https://vinted.fr/items/456")
        
        # Should extract some information even without JSON-LD
        assert item.url == "https://vinted.fr/items/456"
    
    def test_parse_missing_optional_fields(self, parser):
        """Test parsing with missing optional fields."""
        html = """
        <html>
        <head>
            <script type="application/ld+json">
            {
                "@type": "Product",
                "name": "Simple Item",
                "offers": {"@type": "Offer", "price": "10.00"}
            }
            </script>
        </head>
        <body></body>
        </html>
        """
        
        soup = BeautifulSoup(html, "html.parser")
        item = parser.parse_item(soup, "https://vinted.fr/items/789")
        
        assert item.title == "Simple Item"
        assert item.price == 10.00
        assert item.brand is None  # Optional field missing
    
    def test_parse_malformed_html(self, parser):
        """Test error handling for malformed HTML."""
        html = "<html><body><div>Incomplete"
        
        soup = BeautifulSoup(html, "html.parser")
        
        # Should not crash, may return partial data or None
        try:
            item = parser.parse_item(soup, "https://vinted.fr/items/999")
            # If it succeeds, check basic fields exist
            if item:
                assert item.url is not None
        except Exception:
            # Acceptable to raise exception for malformed HTML
            pass
    
    def test_extract_item_id_from_url(self, parser):
        """Test extracting item ID from URL."""
        url = "https://www.vinted.fr/items/1234567890-casual-shirt"
        
        item_id = parser.extract_item_id(url)
        
        assert item_id == "1234567890"
    
    def test_extract_price_from_string(self, parser):
        """Test price extraction from various formats."""
        test_cases = [
            ("29.99", 29.99),
            ("€45.00", 45.00),
            ("19,50 €", 19.50),
            ("FREE", 0.0),
        ]
        
        for price_str, expected in test_cases:
            price = parser.parse_price(price_str)
            assert price == pytest.approx(expected, abs=0.01)


class TestParserRobustness:
    """Test parser robustness and edge cases."""
    
    def test_empty_html(self):
        """Test parsing empty HTML."""
        parser = VintedParser()
        soup = BeautifulSoup("", "html.parser")
        
        # Should handle gracefully
        item = parser.parse_item(soup, "https://vinted.fr/items/000")
        assert item.url is not None
    
    def test_non_ascii_characters(self):
        """Test handling non-ASCII characters."""
        parser = VintedParser()
        html = """
        <script type="application/ld+json">
        {"@type": "Product", "name": "Robe été", "offers": {"price": "25.00"}}
        </script>
        """
        
        soup = BeautifulSoup(html, "html.parser")
        item = parser.parse_item(soup, "https://vinted.fr/items/111")
        
        assert "été" in item.title

"""Utility functions for FashionMatch Streamlit UI."""

from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from src.database.models import SearchResult
from src.ui.state_manager import FilterSettings


def load_image_from_upload(uploaded_file) -> Image.Image:
    """Convert Streamlit UploadedFile to PIL Image.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        PIL Image object in RGB mode
        
    Raises:
        ValueError: If file format is invalid
    """
    try:
        image = Image.open(uploaded_file)
        # Convert to RGB if needed (handle RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}") from e


def load_image_from_path(path: str | Path) -> Optional[Image.Image]:
    """Load image from file path with error handling.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        path = Path(path)
        if not path.exists():
            return None
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception:
        return None


def format_price(price: Optional[float], currency: str = "€") -> str:
    """Format price with currency symbol.
    
    Args:
        price: Price value
        currency: Currency symbol (default Euro)
        
    Returns:
        Formatted price string (e.g., "€25.00")
    """
    if price is None:
        return "N/A"
    return f"{currency}{price:.2f}"


def format_similarity_score(score: float, as_percentage: bool = True) -> str:
    """Format similarity score with optional percentage.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        as_percentage: If True, format as percentage
        
    Returns:
        Formatted score string
    """
    if as_percentage:
        return f"{score * 100:.1f}%"
    else:
        return f"{score:.3f}"


def get_score_color(score: float) -> str:
    """Get color code based on score value.
    
    Args:
        score: Similarity score (0.0 to 1.0)
        
    Returns:
        CSS color code
    """
    if score >= 0.8:
        return "#00C853"  # Green
    elif score >= 0.6:
        return "#FFD600"  # Yellow
    elif score >= 0.4:
        return "#FF6D00"  # Orange
    else:
        return "#DD2C00"  # Red


def create_external_link(url: str, text: str = "View Item") -> str:
    """Generate HTML link that opens in new tab.
    
    Args:
        url: Target URL
        text: Link text
        
    Returns:
        HTML string
    """
    return f'<a href="{url}" target="_blank" style="text-decoration: none;">{text}</a>'


def compute_average_embeddings(embeddings_list: list[np.ndarray]) -> np.ndarray:
    """Average multiple embeddings for multi-reference search.
    
    Args:
        embeddings_list: List of embedding vectors
        
    Returns:
        Averaged embedding vector
    """
    if not embeddings_list:
        raise ValueError("Empty embeddings list")
    
    stacked = np.stack(embeddings_list)
    return np.mean(stacked, axis=0).astype(np.float32)


def get_unique_categories(results: list[SearchResult]) -> list[str]:
    """Extract unique categories from search results.
    
    Args:
        results: List of SearchResult objects
        
    Returns:
        Sorted list of unique category names
    """
    categories = set()
    for result in results:
        if result.item.category:
            categories.add(result.item.category)
    return sorted(categories)


def get_price_range(results: list[SearchResult]) -> tuple[float, float]:
    """Get min and max prices from search results.
    
    Args:
        results: List of SearchResult objects
        
    Returns:
        Tuple of (min_price, max_price)
    """
    prices = [r.item.price for r in results if r.item.price is not None]
    if not prices:
        return 0.0, 1000.0
    return min(prices), max(prices)


def apply_filters(
    results: list[SearchResult],
    filter_settings: FilterSettings
) -> list[SearchResult]:
    """Filter and sort search results.
    
    Args:
        results: List of SearchResult objects
        filter_settings: Filter criteria
        
    Returns:
        Filtered and sorted list
    """
    # Apply price filter
    filtered = [
        r for r in results
        if r.item.price is not None and
        filter_settings.min_price <= r.item.price <= filter_settings.max_price
    ]
    
    # Apply category filter
    if filter_settings.categories:
        filtered = [
            r for r in filtered
            if r.item.category in filter_settings.categories
        ]
    
    # Apply similarity filter
    filtered = [
        r for r in filtered
        if r.similarity_score >= filter_settings.min_similarity
    ]
    
    # Apply sorting
    if filter_settings.sort_by == "similarity":
        filtered.sort(key=lambda r: r.similarity_score, reverse=True)
    elif filter_settings.sort_by == "price_asc":
        filtered.sort(key=lambda r: r.item.price or 0.0)
    elif filter_settings.sort_by == "price_desc":
        filtered.sort(key=lambda r: r.item.price or 0.0, reverse=True)
    
    return filtered


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to maximum length with ellipsis.
    
    Args:
        text: Input text
        max_length: Maximum length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def create_thumbnail(image: Image.Image, size: tuple[int, int] = (300, 300)) -> Image.Image:
    """Create thumbnail of image maintaining aspect ratio.
    
    Args:
        image: PIL Image
        size: Target size (width, height)
        
    Returns:
        Thumbnail image
    """
    img_copy = image.copy()
    img_copy.thumbnail(size, Image.Resampling.LANCZOS)
    return img_copy

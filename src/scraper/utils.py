"""Helper utilities for scraping operations."""

import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse


def sanitize_filename(name: str, max_length: int = 255) -> str:
    """Remove invalid characters from filename and truncate.
    
    Args:
        name: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename safe for filesystem
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    
    # Remove multiple consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Truncate to max length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    # Strip leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized if sanitized else 'unnamed'


def generate_batch_id() -> str:
    """Generate a unique timestamp-based batch ID.
    
    Returns:
        Batch ID in format: batch_YYYYMMDD_HHMMSS_ffffff
    """
    now = datetime.now()
    return now.strftime("batch_%Y%m%d_%H%M%S_%f")


def validate_url(url: str) -> bool:
    """Check if URL is well-formed and from Vinted domain.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid Vinted URL, False otherwise
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ('http', 'https'):
            return False
        
        # Check if domain contains 'vinted'
        if not parsed.netloc or 'vinted' not in parsed.netloc.lower():
            return False
        
        return True
    except Exception:
        return False


def extract_item_id_from_url(url: str) -> str:
    """Extract Vinted item ID from URL.
    
    Handles various URL formats:
    - https://www.vinted.fr/items/1234567890
    - https://www.vinted.fr/items/1234567890-product-title
    
    Args:
        url: Vinted product URL
        
    Returns:
        Item ID as string
        
    Raises:
        ValueError: If item ID cannot be extracted
    """
    try:
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        
        # Look for 'items' segment
        if 'items' in path_parts:
            items_index = path_parts.index('items')
            if items_index + 1 < len(path_parts):
                # Extract ID from next segment (may have format: ID-title)
                id_segment = path_parts[items_index + 1]
                # Split on hyphen and take first part (the ID)
                item_id = id_segment.split('-')[0]
                
                # Validate it's numeric
                if item_id.isdigit():
                    return item_id
        
        raise ValueError(f"Could not extract item ID from URL: {url}")
    
    except Exception as e:
        raise ValueError(f"Invalid Vinted URL format: {url}") from e


def format_price(price_str: str) -> float:
    """Parse price string to float.
    
    Handles various formats:
    - "12,50 €" → 12.50
    - "12.50€" → 12.50
    - "€12.50" → 12.50
    - "12,50" → 12.50
    
    Args:
        price_str: Price string from webpage
        
    Returns:
        Price as float
        
    Raises:
        ValueError: If price cannot be parsed
    """
    try:
        # Remove currency symbols and whitespace
        cleaned = re.sub(r'[€$£\s]', '', price_str)
        
        # Replace comma with dot for decimal
        cleaned = cleaned.replace(',', '.')
        
        # Extract numeric value
        match = re.search(r'\d+\.?\d*', cleaned)
        if match:
            return float(match.group())
        
        raise ValueError(f"No numeric value found in price string: {price_str}")
    
    except Exception as e:
        raise ValueError(f"Failed to parse price: {price_str}") from e


def create_batch_directory(base_dir: Path, batch_id: str) -> Path:
    """Create directory structure for a scraping batch.
    
    Creates:
    - {base_dir}/{batch_id}/
    - {base_dir}/{batch_id}/items/
    - {base_dir}/{batch_id}/images/
    
    Args:
        base_dir: Base output directory
        batch_id: Unique batch identifier
        
    Returns:
        Path to batch directory
    """
    batch_dir = base_dir / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (batch_dir / 'items').mkdir(exist_ok=True)
    (batch_dir / 'images').mkdir(exist_ok=True)
    
    return batch_dir

"""
Input validation utilities.
"""

import re
from typing import Any, List, Optional
from urllib.parse import urlparse


def validate_url(url: str) -> bool:
    """
    Validate that a string is a valid URL.
    
    Args:
        url: URL string to validate.
        
    Returns:
        True if valid URL, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_vinted_url(url: str) -> bool:
    """
    Validate that URL is a Vinted URL.
    
    Args:
        url: URL string to validate.
        
    Returns:
        True if valid Vinted URL, False otherwise.
    """
    if not validate_url(url):
        return False
    
    parsed = urlparse(url)
    return "vinted" in parsed.netloc.lower()


def validate_price(price: Any) -> float:
    """
    Validate and convert price to float.
    
    Args:
        price: Price value (string or number).
        
    Returns:
        Price as float.
        
    Raises:
        ValueError: If price is invalid or negative.
    """
    try:
        # Handle string prices like "25,50 €" or "25.50"
        if isinstance(price, str):
            # Remove currency symbols and spaces
            price = re.sub(r"[€$£\s]", "", price)
            # Replace comma with dot for European format
            price = price.replace(",", ".")
        
        price_float = float(price)
        
        if price_float < 0:
            raise ValueError("Price cannot be negative")
        
        return price_float
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid price value: {price}") from e


def validate_embedding_dimension(
    embedding: List[float], 
    expected_dim: int
) -> None:
    """
    Validate embedding has correct dimension.
    
    Args:
        embedding: Embedding vector to validate.
        expected_dim: Expected dimension.
        
    Raises:
        ValueError: If dimension doesn't match.
    """
    if len(embedding) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: "
            f"got {len(embedding)}, expected {expected_dim}"
        )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a string to be safe for use as a filename.
    
    Args:
        filename: String to sanitize.
        
    Returns:
        Sanitized filename.
    """
    # Remove or replace invalid characters
    invalid_chars = r'[<>:"/\\|?*]'
    sanitized = re.sub(invalid_chars, "_", filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(". ")
    
    # Limit length
    max_length = 200
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    
    return sanitized or "unnamed"

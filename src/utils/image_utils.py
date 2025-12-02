"""
Image utility functions for preprocessing.

TODO: Implement full preprocessing in Phase 2
"""

from pathlib import Path
from typing import Tuple, Union

from PIL import Image


def load_image(path: Union[str, Path]) -> Image.Image:
    """
    Load an image from path.
    
    Args:
        path: Path to the image file.
        
    Returns:
        PIL Image in RGB format.
        
    Raises:
        FileNotFoundError: If image doesn't exist.
        ValueError: If image format is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image {path}: {e}")


def resize_image(
    image: Image.Image, 
    size: Tuple[int, int] = (224, 224)
) -> Image.Image:
    """
    Resize image to target size.
    
    Args:
        image: PIL Image to resize.
        size: Target size (width, height).
        
    Returns:
        Resized PIL Image.
    """
    return image.resize(size, Image.Resampling.LANCZOS)


def validate_image_path(path: Union[str, Path]) -> Path:
    """
    Validate that image path exists and has valid extension.
    
    Args:
        path: Path to validate.
        
    Returns:
        Validated Path object.
        
    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If extension is not a valid image format.
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    valid_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    if path.suffix.lower() not in valid_extensions:
        raise ValueError(
            f"Invalid image extension: {path.suffix}. "
            f"Supported: {valid_extensions}"
        )
    
    return path

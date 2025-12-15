"""Enhanced image validation utilities."""

from pathlib import Path
from typing import Optional

from PIL import Image

from src.utils import get_logger

logger = get_logger(__name__)

# Image validation constraints
MAX_FILE_SIZE_MB = 10
MIN_DIMENSION = 50
MAX_DIMENSION = 5000


class ImageValidationError(Exception):
    """Exception raised for invalid images."""
    pass


def validate_image_file(path: str | Path) -> None:
    """Validate image file before processing.
    
    Args:
        path: Path to image file
        
    Raises:
        ImageValidationError: If image fails validation
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    
    # Check file exists
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")
    
    # Check file size
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ImageValidationError(
            f"Image too large: {file_size_mb:.1f}MB (max {MAX_FILE_SIZE_MB}MB)"
        )
    
    # Try to open and validate image
    try:
        with Image.open(path) as img:
            # Check dimensions
            width, height = img.size
            
            if width < MIN_DIMENSION or height < MIN_DIMENSION:
                raise ImageValidationError(
                    f"Image too small: {width}×{height} (min {MIN_DIMENSION}×{MIN_DIMENSION})"
                )
            
            if width > MAX_DIMENSION or height > MAX_DIMENSION:
                raise ImageValidationError(
                    f"Image too large: {width}×{height} (max {MAX_DIMENSION}×{MAX_DIMENSION})"
                )
            
            # Check format
            if img.format not in ["JPEG", "PNG", "JPG"]:
                raise ImageValidationError(
                    f"Unsupported format: {img.format} (supported: JPEG, PNG)"
                )
            
    except Exception as e:
        if isinstance(e, (ImageValidationError, FileNotFoundError)):
            raise
        raise ImageValidationError(f"Corrupted or invalid image: {e}") from e
    
    logger.debug(f"Image validation passed: {path.name}")


def load_image_safe(path: str | Path) -> Optional[Image.Image]:
    """Load image with validation and error handling.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image or None if loading fails
    """
    try:
        validate_image_file(path)
        
        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        return img
        
    except ImageValidationError as e:
        logger.warning(f"Image validation failed: {e}")
        return None
    except FileNotFoundError as e:
        logger.error(f"Image file not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading image: {e}")
        return None

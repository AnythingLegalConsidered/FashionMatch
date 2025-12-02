"""
Image utility functions for preprocessing.

Provides functions for:
- Loading images from file paths or URLs
- Resizing and normalizing images
- Converting images to tensors for model input
- Validating image files

Example:
    >>> from src.utils.image_utils import load_image, preprocess_image
    >>> image = load_image("path/to/image.jpg")
    >>> processed = preprocess_image(image)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import requests
from PIL import Image

from src.utils.config import get_settings
from src.utils.exceptions import (
    ImageDownloadError,
    ImageProcessingError,
    InvalidImageError,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Valid image extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

# Default timeout for URL downloads
DEFAULT_TIMEOUT = 30


def load_image(
    source: Union[str, Path],
    timeout: int = DEFAULT_TIMEOUT,
) -> Image.Image:
    """
    Load an image from a file path or URL.

    Automatically detects if the source is a URL or local path and loads
    the image accordingly. Always converts to RGB format.

    Args:
        source: File path (str/Path) or URL string.
        timeout: Timeout in seconds for URL downloads. Defaults to 30.

    Returns:
        PIL Image in RGB format.

    Raises:
        FileNotFoundError: If local file doesn't exist.
        InvalidImageError: If file extension is not valid.
        ImageDownloadError: If URL download fails.
        ImageProcessingError: If image cannot be opened/processed.

    Example:
        >>> image = load_image("data/references/style1.jpg")
        >>> image = load_image("https://example.com/image.jpg")
    """
    source_str = str(source)

    # Check if source is a URL
    if source_str.startswith(("http://", "https://")):
        return _load_from_url(source_str, timeout)
    else:
        return _load_from_path(Path(source_str))


def _load_from_path(path: Path) -> Image.Image:
    """
    Load an image from a local file path.

    Args:
        path: Path to the image file.

    Returns:
        PIL Image in RGB format.

    Raises:
        FileNotFoundError: If file doesn't exist.
        InvalidImageError: If extension is invalid.
        ImageProcessingError: If image cannot be opened.
    """
    # Validate path exists
    if not path.exists():
        logger.error(f"Image file not found: {path}")
        raise FileNotFoundError(f"Image not found: {path}")

    # Validate extension
    if path.suffix.lower() not in VALID_EXTENSIONS:
        logger.error(f"Invalid image extension: {path.suffix}")
        raise InvalidImageError(
            f"Invalid image extension: {path.suffix}",
            path=str(path),
            reason=f"Supported extensions: {VALID_EXTENSIONS}",
        )

    # Load image
    try:
        image = Image.open(path)
        # Convert to RGB (handles RGBA, grayscale, etc.)
        image = image.convert("RGB")
        logger.debug(f"Loaded image from {path}: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        raise ImageProcessingError(
            f"Failed to load image: {e}",
            image_path=str(path),
        )


def _load_from_url(url: str, timeout: int) -> Image.Image:
    """
    Download and load an image from a URL.

    Args:
        url: URL to download the image from.
        timeout: Request timeout in seconds.

    Returns:
        PIL Image in RGB format.

    Raises:
        ImageDownloadError: If download fails.
        ImageProcessingError: If image cannot be processed.
    """
    logger.debug(f"Downloading image from URL: {url}")

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error(f"Timeout downloading image from {url}")
        raise ImageDownloadError(
            f"Timeout downloading image (>{timeout}s)",
            image_url=url,
        )
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download image from {url}: {e}")
        raise ImageDownloadError(
            f"Failed to download image: {e}",
            image_url=url,
        )

    # Load image from response content
    try:
        image = Image.open(io.BytesIO(response.content))
        image = image.convert("RGB")
        logger.debug(f"Downloaded image from {url}: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to process downloaded image from {url}: {e}")
        raise ImageProcessingError(
            f"Failed to process downloaded image: {e}",
            image_path=url,
        )


def resize_image(
    image: Image.Image,
    size: Optional[Tuple[int, int]] = None,
    resample: Image.Resampling = Image.Resampling.LANCZOS,
) -> Image.Image:
    """
    Resize image to target size.

    If size is not provided, reads from config (images.max_size).

    Args:
        image: PIL Image to resize.
        size: Target size (width, height). Uses config if None.
        resample: Resampling filter. Defaults to LANCZOS for quality.

    Returns:
        Resized PIL Image.

    Example:
        >>> image = load_image("image.jpg")
        >>> resized = resize_image(image)  # Uses config size
        >>> resized = resize_image(image, size=(256, 256))
    """
    if size is None:
        settings = get_settings()
        size = tuple(settings.images.max_size)

    if image.size == size:
        return image

    resized = image.resize(size, resample=resample)
    logger.debug(f"Resized image from {image.size} to {size}")
    return resized


def preprocess_image(
    image: Union[str, Path, Image.Image],
    size: Optional[Tuple[int, int]] = None,
) -> Image.Image:
    """
    Full preprocessing pipeline for images.

    Loads (if path/URL), resizes, and prepares for model input.

    Args:
        image: Image path, URL, or PIL Image.
        size: Target size. Uses config if None.

    Returns:
        Preprocessed PIL Image.

    Example:
        >>> processed = preprocess_image("path/to/image.jpg")
        >>> processed = preprocess_image(pil_image, size=(224, 224))
    """
    # Load if not already a PIL Image
    if not isinstance(image, Image.Image):
        image = load_image(image)

    # Get settings for defaults
    settings = get_settings()

    if size is None:
        size = tuple(settings.images.max_size)

    # Resize
    image = resize_image(image, size=size)

    return image


def image_to_numpy(
    image: Image.Image,
    normalize: bool = True,
) -> np.ndarray:
    """
    Convert PIL Image to numpy array.

    Args:
        image: PIL Image (RGB).
        normalize: If True, normalize to [0, 1] range.

    Returns:
        Numpy array of shape (H, W, C) with values in [0, 1] or [0, 255].
    """
    arr = np.array(image, dtype=np.float32)

    if normalize:
        arr = arr / 255.0

    return arr


def validate_image_path(path: Union[str, Path]) -> Path:
    """
    Validate that image path exists and has valid extension.

    Args:
        path: Path to validate.

    Returns:
        Validated Path object.

    Raises:
        FileNotFoundError: If path doesn't exist.
        InvalidImageError: If extension is not valid.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if path.suffix.lower() not in VALID_EXTENSIONS:
        raise InvalidImageError(
            f"Invalid image extension: {path.suffix}",
            path=str(path),
            reason=f"Supported: {VALID_EXTENSIONS}",
        )

    return path


def get_image_info(image: Union[str, Path, Image.Image]) -> dict:
    """
    Get information about an image.

    Args:
        image: Image path, URL, or PIL Image.

    Returns:
        Dictionary with image information (size, mode, format).
    """
    if not isinstance(image, Image.Image):
        image = load_image(image)

    return {
        "size": image.size,
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "format": image.format,
    }


def save_image(
    image: Image.Image,
    path: Union[str, Path],
    quality: int = 95,
) -> Path:
    """
    Save a PIL Image to disk.

    Args:
        image: PIL Image to save.
        path: Destination path.
        quality: JPEG quality (1-100). Defaults to 95.

    Returns:
        Path where image was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Determine format from extension
    save_kwargs = {}
    if path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs["quality"] = quality

    image.save(path, **save_kwargs)
    logger.debug(f"Saved image to {path}")

    return path

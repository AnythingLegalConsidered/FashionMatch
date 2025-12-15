"""Image preprocessing utilities for CLIP and DINOv2 models.

This module provides functions for loading, preprocessing, and saving images
with model-specific normalization for CLIP and DINOv2.
"""

from pathlib import Path
from typing import Callable

import torch
import torchvision.transforms as transforms
from PIL import Image


# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# ImageNet normalization constants (used by DINOv2)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_image(path: str | Path) -> Image.Image:
    """Load an image from disk.
    
    Args:
        path: Path to image file
        
    Returns:
        PIL Image object
        
    Raises:
        FileNotFoundError: If image file doesn't exist
        OSError: If image is corrupted or format is unsupported
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    
    try:
        image = Image.open(path)
        # Verify image can be loaded
        image.verify()
        # Reopen after verify (verify closes the file)
        image = Image.open(path)
        return image
    except Exception as e:
        raise OSError(f"Failed to load image {path}: {str(e)}")


def preprocess_for_clip(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Preprocess image for CLIP model.
    
    Args:
        image: PIL Image to preprocess
        size: Target size (square) for resizing
        
    Returns:
        Preprocessed tensor with shape (3, size, size)
    """
    # Convert to RGB if needed (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
    ])
    
    return transform(image)


def preprocess_for_dino(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Preprocess image for DINOv2 model.
    
    Args:
        image: PIL Image to preprocess
        size: Target size (square) for resizing
        
    Returns:
        Preprocessed tensor with shape (3, size, size)
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    
    return transform(image)


def batch_preprocess(images: list[Image.Image], preprocessor: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:
    """Preprocess a batch of images using the specified preprocessor.
    
    Args:
        images: List of PIL Images
        preprocessor: Function to preprocess each image (e.g., preprocess_for_clip)
        
    Returns:
        Batched tensor with shape (B, 3, H, W) where B is the number of images
    """
    if not images:
        raise ValueError("Cannot preprocess empty image list")
    
    preprocessed = [preprocessor(img) for img in images]
    return torch.stack(preprocessed)


def save_image(image: Image.Image | torch.Tensor, path: str | Path, quality: int = 95) -> None:
    """Save an image to disk.
    
    Args:
        image: PIL Image or torch Tensor to save
        path: Output path
        quality: JPEG quality (1-100), only used for JPEG/JPG formats
        
    Raises:
        ValueError: If image type is unsupported
        OSError: If save operation fails
    """
    path = Path(path)
    
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert tensor to PIL if needed
    if isinstance(image, torch.Tensor):
        # Assume tensor is in range [0, 1] or normalized
        # Denormalize if needed and convert to PIL
        if image.dim() == 4:
            # Batch tensor, take first image
            image = image[0]
        
        # Detach and move to CPU if needed
        image = image.detach().cpu()
        
        # Check if normalized (values outside [0, 1])
        if image.min() < 0 or image.max() > 1:
            # Assume normalized, denormalize (using CLIP constants as default)
            mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
            std = torch.tensor(CLIP_STD).view(3, 1, 1)
            image = image * std + mean
            image = torch.clamp(image, 0, 1)
        
        # Convert to PIL: (C, H, W) -> (H, W, C)
        image = transforms.ToPILImage()(image)
    
    elif not isinstance(image, Image.Image):
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Save with appropriate settings
    try:
        if path.suffix.lower() in ['.jpg', '.jpeg']:
            image.save(path, quality=quality, optimize=True)
        else:
            image.save(path)
    except Exception as e:
        raise OSError(f"Failed to save image to {path}: {str(e)}")


def get_image_size(image: Image.Image | torch.Tensor) -> tuple[int, int]:
    """Get the size (width, height) of an image.
    
    Args:
        image: PIL Image or torch Tensor
        
    Returns:
        Tuple of (width, height)
    """
    if isinstance(image, Image.Image):
        return image.size
    elif isinstance(image, torch.Tensor):
        # Tensor shape is (C, H, W) or (B, C, H, W)
        if image.dim() == 4:
            _, _, h, w = image.shape
        else:
            _, h, w = image.shape
        return (w, h)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


def create_thumbnail(image: Image.Image, max_size: tuple[int, int] = (256, 256)) -> Image.Image:
    """Create a thumbnail of an image while maintaining aspect ratio.
    
    Args:
        image: PIL Image to create thumbnail from
        max_size: Maximum size (width, height) for thumbnail
        
    Returns:
        Thumbnail PIL Image
    """
    # Create a copy to avoid modifying original
    thumb = image.copy()
    thumb.thumbnail(max_size, Image.Resampling.BICUBIC)
    return thumb

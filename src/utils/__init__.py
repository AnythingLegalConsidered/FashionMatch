"""Utility modules for configuration, logging, and image processing."""

import torch

from .config import get_config, load_config, reset_config
from .logger import (
    get_logger,
    log_execution_time,
    log_model_info,
    log_performance,
    set_log_level,
    log_exception,
)
from .image_utils import (
    load_image,
    preprocess_for_clip,
    preprocess_for_dino,
    batch_preprocess,
    save_image,
    get_image_size,
    create_thumbnail,
)

logger = get_logger(__name__)


def detect_device(device_config: str) -> str:
    """Detect and validate device for model inference.
    
    Args:
        device_config: Device configuration ("auto", "cuda", or "cpu")
        
    Returns:
        Selected device string ("cuda" or "cpu")
    """
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_config == "cuda":
        if torch.cuda.is_available():
            device = "cuda"
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
    
    logger.info(f"Using device: {device}")
    return device

__all__ = [
    # Configuration
    "get_config",
    "load_config",
    "reset_config",
    # Logging
    "get_logger",
    "log_execution_time",
    "log_model_info",
    "log_performance",
    "set_log_level",
    "log_exception",
    # Image processing
    "load_image",
    "preprocess_for_clip",
    "preprocess_for_dino",
    "batch_preprocess",
    "save_image",
    "get_image_size",
    "create_thumbnail",
    # Device utilities
    "detect_device",
]

"""
Base encoder abstract class with common functionality.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

from src.domain.interfaces.encoder_interface import EncoderInterface


class BaseEncoder(EncoderInterface, ABC):
    """
    Abstract base class for image encoders with common functionality.
    
    Provides shared methods for image loading and validation.
    """

    def __init__(self, device: str = "cuda"):
        """
        Initialize the encoder.
        
        Args:
            device: Device to run the model on ("cuda" or "cpu").
        """
        self._device = device
        self._model = None
        self._is_loaded = False

    @property
    def device(self) -> str:
        """Return the device the model runs on."""
        return self._device

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    def _load_image(self, image: Union[str, Path, Image.Image]) -> Image.Image:
        """
        Load an image from path or return if already a PIL Image.
        
        Args:
            image: Image path or PIL Image.
            
        Returns:
            PIL Image object.
            
        Raises:
            FileNotFoundError: If image path doesn't exist.
            ValueError: If image format is invalid.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        
        path = Path(image)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        try:
            return Image.open(path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {path}: {e}")

    def _validate_loaded(self) -> None:
        """Raise error if model not loaded."""
        if not self._is_loaded:
            raise RuntimeError(
                f"{self.__class__.__name__} model not loaded. "
                "Call load_model() first."
            )

    @abstractmethod
    def encode(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Generate embedding vector from image."""
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model."""
        pass

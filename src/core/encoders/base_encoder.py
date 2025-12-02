"""
Base encoder abstract class with common functionality.

Provides the foundation for all image encoders (CLIP, DINOv2, etc.) with:
- Abstract interface enforcement
- Device management (CPU/CUDA/MPS)
- Image loading and validation
- Batch encoding support
- Consistent error handling

All concrete encoders (CLIPEncoder, DINOEncoder) must inherit from this class.

Example:
    >>> class MyEncoder(BaseEncoder):
    ...     def encode(self, image):
    ...         # Implementation
    ...         pass
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.domain.interfaces.encoder_interface import EncoderInterface
from src.utils.config import get_settings
from src.utils.exceptions import (
    ImageProcessingError,
    ModelLoadError,
    ModelNotLoadedError,
)
from src.utils.image_utils import load_image, preprocess_image
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseEncoder(EncoderInterface, ABC):
    """
    Abstract base class for image encoders with common functionality.

    Provides shared infrastructure for:
    - Device selection and management
    - Model loading lifecycle
    - Image preprocessing
    - Error handling

    Subclasses must implement:
    - encode(): Generate embedding from a single image
    - load_model(): Load the pretrained model weights
    - embedding_dim: Property returning the embedding dimension
    - model_name: Property returning the model identifier

    Attributes:
        device: The device the model runs on (cuda/cpu/mps).

    Example:
        >>> encoder = CLIPEncoder()  # Inherits from BaseEncoder
        >>> encoder.load_model()
        >>> embedding = encoder.encode("image.jpg")
        >>> print(embedding.shape)
        (512,)
    """

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the encoder.

        Args:
            device: Device to run the model on ("cuda", "cpu", or "mps").
                    If None, reads from config. Falls back to CPU if
                    CUDA/MPS not available.

        Example:
            >>> encoder = MyEncoder()  # Uses config device
            >>> encoder = MyEncoder(device="cpu")  # Force CPU
        """
        self._device = self._resolve_device(device)
        self._model: Optional[torch.nn.Module] = None
        self._preprocessor = None
        self._is_loaded = False

        logger.info(
            f"Initialized {self.__class__.__name__} on device: {self._device}"
        )

    def _resolve_device(self, device: Optional[str]) -> str:
        """
        Resolve the device to use, with fallback logic.

        Priority:
        1. Explicitly provided device
        2. Device from config
        3. CUDA if available
        4. MPS if available (Apple Silicon)
        5. CPU as fallback

        Args:
            device: Requested device or None.

        Returns:
            Resolved device string.
        """
        if device is not None:
            requested = device.lower()
        else:
            settings = get_settings()
            requested = settings.models.device.lower()

        # Validate and fallback
        if requested == "cuda":
            if torch.cuda.is_available():
                logger.debug("Using CUDA device")
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"

        elif requested == "mps":
            if torch.backends.mps.is_available():
                logger.debug("Using MPS device (Apple Silicon)")
                return "mps"
            else:
                logger.warning("MPS requested but not available, falling back to CPU")
                return "cpu"

        else:
            logger.debug("Using CPU device")
            return "cpu"

    @property
    def device(self) -> str:
        """Return the device the model runs on."""
        return self._device

    @property
    def torch_device(self) -> torch.device:
        """Return the torch.device object."""
        return torch.device(self._device)

    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._is_loaded

    def _validate_loaded(self) -> None:
        """
        Raise error if model not loaded.

        Raises:
            ModelNotLoadedError: If model hasn't been loaded yet.
        """
        if not self._is_loaded:
            raise ModelNotLoadedError(
                f"{self.__class__.__name__} model not loaded. Call load_model() first.",
                model_name=self.model_name,
            )

    def _load_image(
        self,
        image: Union[str, Path, Image.Image],
    ) -> Image.Image:
        """
        Load and validate an image.

        Handles both file paths and PIL Images, always returning RGB.

        Args:
            image: Image path (str/Path) or PIL Image.

        Returns:
            PIL Image object in RGB format.

        Raises:
            FileNotFoundError: If image path doesn't exist.
            ImageProcessingError: If image cannot be processed.
        """
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        try:
            return load_image(image)
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise ImageProcessingError(
                f"Failed to load image: {e}",
                image_path=str(image),
            )

    def _preprocess_image(
        self,
        image: Image.Image,
    ) -> Image.Image:
        """
        Preprocess image for the encoder.

        Default implementation resizes to config size.
        Subclasses may override for model-specific preprocessing.

        Args:
            image: PIL Image to preprocess.

        Returns:
            Preprocessed PIL Image.
        """
        return preprocess_image(image)

    @abstractmethod
    def encode(
        self,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Generate embedding vector from an image.

        Must be implemented by subclasses.

        Args:
            image: Image path (str/Path) or PIL Image object.

        Returns:
            Numpy array containing the embedding vector.
            Shape: (embedding_dim,)

        Raises:
            ModelNotLoadedError: If model hasn't been loaded.
            ImageProcessingError: If image cannot be processed.
        """
        pass

    def encode_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images.

        Default implementation processes images sequentially.
        Subclasses may override for optimized batch processing.

        Args:
            images: List of image paths or PIL Images.
            batch_size: Number of images to process at once.

        Returns:
            Numpy array of shape (n_images, embedding_dim).

        Raises:
            ModelNotLoadedError: If model hasn't been loaded.
        """
        self._validate_loaded()

        embeddings = []
        total = len(images)

        for i, image in enumerate(images):
            try:
                embedding = self.encode(image)
                embeddings.append(embedding)

                if (i + 1) % 10 == 0:
                    logger.debug(f"Encoded {i + 1}/{total} images")

            except Exception as e:
                logger.warning(f"Failed to encode image {i}: {e}")
                # Append zeros for failed images
                embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))

        logger.info(f"Encoded {total} images")
        return np.stack(embeddings, axis=0)

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Return the embedding dimension.

        Must be implemented by subclasses.

        Returns:
            Integer dimension of the embedding vector.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Return the model name/identifier.

        Must be implemented by subclasses.

        Returns:
            String identifier for the model (e.g., "ViT-B/32").
        """
        pass

    @abstractmethod
    def load_model(self) -> None:
        """
        Load the pretrained model into memory.

        Must be implemented by subclasses.

        Should set self._model and self._is_loaded = True.

        Raises:
            ModelLoadError: If model cannot be loaded.
        """
        pass

    def unload_model(self) -> None:
        """
        Unload the model from memory.

        Useful for freeing GPU memory when switching models.
        """
        if self._model is not None:
            del self._model
            self._model = None

        if self._preprocessor is not None:
            del self._preprocessor
            self._preprocessor = None

        self._is_loaded = False

        # Clear CUDA cache if using GPU
        if self._device == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"Unloaded {self.__class__.__name__} model")

    def __repr__(self) -> str:
        """String representation of the encoder."""
        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name!r}, "
            f"embedding_dim={self.embedding_dim}, "
            f"device={self._device!r}, "
            f"loaded={self._is_loaded})"
        )

    def __enter__(self) -> "BaseEncoder":
        """Context manager entry - load model."""
        if not self._is_loaded:
            self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - unload model."""
        self.unload_model()

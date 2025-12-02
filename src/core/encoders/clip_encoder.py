"""
CLIP Encoder for semantic image embeddings.

Uses OpenAI's CLIP model via Hugging Face Transformers to generate
semantic embeddings that capture the conceptual content of images.

CLIP excels at understanding:
- Object categories (shirt, pants, shoes)
- Colors and patterns
- Style attributes (casual, formal, vintage)
- Scene context

Example:
    >>> from src.core.encoders.clip_encoder import CLIPEncoder
    >>> encoder = CLIPEncoder()
    >>> encoder.load_model()
    >>> embedding = encoder.encode("image.jpg")
    >>> print(embedding.shape)
    (512,)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.core.encoders.base_encoder import BaseEncoder
from src.utils.config import get_settings
from src.utils.exceptions import EncodingError, ModelLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from config model names to Hugging Face model IDs
CLIP_MODEL_MAPPING = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
    "ViT-L/14@336px": "openai/clip-vit-large-patch14-336",
}

# Embedding dimensions for each model
CLIP_EMBEDDING_DIMS = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768,
}


class CLIPEncoder(BaseEncoder):
    """
    CLIP encoder for generating semantic image embeddings.

    Uses OpenAI's CLIP (Contrastive Language-Image Pre-training) model
    to generate embeddings that capture semantic content of images.

    CLIP embeddings are particularly good for:
    - Understanding what objects are in an image
    - Capturing style and aesthetic qualities
    - Cross-modal similarity (image-text matching)

    Attributes:
        model_name: The CLIP model variant (e.g., "ViT-B/32").
        embedding_dim: Dimension of output embeddings (512 or 768).

    Example:
        >>> encoder = CLIPEncoder()
        >>> encoder.load_model()
        >>> 
        >>> # Encode single image
        >>> embedding = encoder.encode("path/to/image.jpg")
        >>> print(embedding.shape)  # (512,)
        >>> 
        >>> # Use as context manager
        >>> with CLIPEncoder() as encoder:
        ...     embedding = encoder.encode(pil_image)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the CLIP encoder.

        Args:
            model_name: CLIP model variant. If None, reads from config.
                        Options: "ViT-B/32", "ViT-B/16", "ViT-L/14"
            device: Device to run on ("cuda", "cpu", "mps").
                    If None, reads from config.

        Example:
            >>> encoder = CLIPEncoder()  # Uses config
            >>> encoder = CLIPEncoder(model_name="ViT-L/14", device="cuda")
        """
        # Get model name from config if not provided
        if model_name is None:
            settings = get_settings()
            model_name = settings.models.clip.model_name

        self._model_name = model_name
        self._embedding_dim = CLIP_EMBEDDING_DIMS.get(model_name, 512)

        # Initialize base class (handles device resolution)
        super().__init__(device=device)

        # Will be set when model is loaded
        self._processor = None

    @property
    def model_name(self) -> str:
        """Return the CLIP model name."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (512 for base, 768 for large)."""
        return self._embedding_dim

    @property
    def hf_model_id(self) -> str:
        """Return the Hugging Face model ID."""
        return CLIP_MODEL_MAPPING.get(
            self._model_name,
            "openai/clip-vit-base-patch32",  # Default fallback
        )

    def load_model(self) -> None:
        """
        Load the CLIP model and processor from Hugging Face.

        Downloads the model on first use (cached afterwards).

        Raises:
            ModelLoadError: If model cannot be loaded.

        Example:
            >>> encoder = CLIPEncoder()
            >>> encoder.load_model()
            >>> print(encoder.is_loaded())
            True
        """
        if self._is_loaded:
            logger.debug(f"CLIP model {self._model_name} already loaded")
            return

        logger.info(f"Loading CLIP model: {self._model_name} ({self.hf_model_id})")

        try:
            # Import here to avoid slow startup if not used
            from transformers import CLIPModel, CLIPProcessor

            # Load processor (handles image preprocessing)
            self._processor = CLIPProcessor.from_pretrained(self.hf_model_id)

            # Load model
            self._model = CLIPModel.from_pretrained(self.hf_model_id)

            # Move to device
            self._model = self._model.to(self.torch_device)

            # Set to evaluation mode
            self._model.eval()

            self._is_loaded = True
            logger.info(
                f"CLIP model loaded successfully on {self._device} "
                f"(embedding_dim={self._embedding_dim})"
            )

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            raise ModelLoadError(
                f"Failed to load CLIP model '{self._model_name}': {e}",
                model_name=self._model_name,
            )

    @torch.no_grad()
    def encode(
        self,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Generate a normalized embedding vector from an image.

        Args:
            image: Image path (str/Path) or PIL Image object.

        Returns:
            Numpy array of shape (embedding_dim,) with L2-normalized values.

        Raises:
            ModelNotLoadedError: If model hasn't been loaded.
            EncodingError: If encoding fails.

        Example:
            >>> embedding = encoder.encode("image.jpg")
            >>> print(embedding.shape)
            (512,)
            >>> print(np.linalg.norm(embedding))  # Should be ~1.0
            1.0
        """
        self._validate_loaded()

        # Load and preprocess image
        pil_image = self._load_image(image)

        try:
            # Process image for CLIP
            inputs = self._processor(
                images=pil_image,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

            # Get image features
            outputs = self._model.get_image_features(**inputs)

            # Convert to numpy and normalize
            embedding = outputs.cpu().numpy().flatten()

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"CLIP encoding failed: {e}")
            raise EncodingError(
                f"Failed to encode image with CLIP: {e}",
                model_name=self._model_name,
            )

    @torch.no_grad()
    def encode_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Generate embeddings for multiple images efficiently.

        Processes images in batches for better GPU utilization.

        Args:
            images: List of image paths or PIL Images.
            batch_size: Number of images to process at once.

        Returns:
            Numpy array of shape (n_images, embedding_dim).

        Raises:
            ModelNotLoadedError: If model hasn't been loaded.
        """
        self._validate_loaded()

        all_embeddings = []
        total = len(images)

        # Process in batches
        for i in range(0, total, batch_size):
            batch = images[i : i + batch_size]

            # Load all images in batch
            pil_images = []
            for img in batch:
                try:
                    pil_images.append(self._load_image(img))
                except Exception as e:
                    logger.warning(f"Failed to load image: {e}")
                    pil_images.append(Image.new("RGB", (224, 224)))

            try:
                # Process batch
                inputs = self._processor(
                    images=pil_images,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

                # Get features
                outputs = self._model.get_image_features(**inputs)
                embeddings = outputs.cpu().numpy()

                # L2 normalize each embedding
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1)
                embeddings = embeddings / norms

                all_embeddings.append(embeddings)

            except Exception as e:
                logger.warning(f"Batch encoding failed: {e}")
                # Fallback: zeros for failed batch
                all_embeddings.append(
                    np.zeros((len(pil_images), self._embedding_dim), dtype=np.float32)
                )

            if (i + batch_size) % 100 == 0 or i + batch_size >= total:
                logger.debug(f"Encoded {min(i + batch_size, total)}/{total} images")

        result = np.vstack(all_embeddings).astype(np.float32)
        logger.info(f"CLIP batch encoding complete: {result.shape}")

        return result

    def encode_text(self, text: str) -> np.ndarray:
        """
        Generate a normalized embedding from text.

        Useful for text-based search queries.

        Args:
            text: Text description to encode.

        Returns:
            Numpy array of shape (embedding_dim,).

        Example:
            >>> text_emb = encoder.encode_text("blue denim jacket")
            >>> # Can compare with image embeddings for search
        """
        self._validate_loaded()

        try:
            inputs = self._processor(
                text=[text],
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model.get_text_features(**inputs)

            embedding = outputs.cpu().numpy().flatten()

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise EncodingError(
                f"Failed to encode text with CLIP: {e}",
                model_name=self._model_name,
            )

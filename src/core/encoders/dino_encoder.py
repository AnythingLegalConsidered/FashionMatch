"""
DINOv2 Encoder for structural image embeddings.

Uses Meta's DINOv2 model via Hugging Face Transformers to generate
embeddings that capture visual structure and patterns.

DINOv2 excels at understanding:
- Textures and patterns (stripes, dots, knit)
- Visual structure and composition
- Fine-grained visual details
- Shape and silhouette

Example:
    >>> from src.core.encoders.dino_encoder import DINOEncoder
    >>> encoder = DINOEncoder()
    >>> encoder.load_model()
    >>> embedding = encoder.encode("image.jpg")
    >>> print(embedding.shape)
    (384,)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch
from PIL import Image

from src.core.encoders.base_encoder import BaseEncoder
from src.utils.config import get_settings
from src.utils.exceptions import EncodingError, ModelLoadError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Mapping from config model names to Hugging Face model IDs
DINO_MODEL_MAPPING = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}

# Embedding dimensions for each model variant
DINO_EMBEDDING_DIMS = {
    "dinov2_vits14": 384,   # Small
    "dinov2_vitb14": 768,   # Base
    "dinov2_vitl14": 1024,  # Large
    "dinov2_vitg14": 1536,  # Giant
}

# Pooling strategies
PoolingStrategy = Literal["cls", "mean", "mean_cls"]


class DINOEncoder(BaseEncoder):
    """
    DINOv2 encoder for generating structural image embeddings.

    Uses Meta's DINOv2 (Self-DIstillation with NO labels v2) model to
    generate embeddings that capture visual patterns and structure.

    DINOv2 embeddings complement CLIP by focusing on:
    - Visual textures and patterns
    - Structural composition
    - Fine-grained visual features
    - Shape and silhouette details

    Attributes:
        model_name: The DINOv2 model variant (e.g., "dinov2_vits14").
        embedding_dim: Dimension of output embeddings (384, 768, 1024, or 1536).
        pooling: Strategy for converting patch tokens to single embedding.

    Example:
        >>> encoder = DINOEncoder()
        >>> encoder.load_model()
        >>> 
        >>> # Encode single image
        >>> embedding = encoder.encode("path/to/image.jpg")
        >>> print(embedding.shape)  # (384,)
        >>> 
        >>> # Use as context manager
        >>> with DINOEncoder() as encoder:
        ...     embedding = encoder.encode(pil_image)
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        pooling: PoolingStrategy = "cls",
    ):
        """
        Initialize the DINOv2 encoder.

        Args:
            model_name: DINOv2 model variant. If None, reads from config.
                        Options: "dinov2_vits14", "dinov2_vitb14", 
                                 "dinov2_vitl14", "dinov2_vitg14"
            device: Device to run on ("cuda", "cpu", "mps").
                    If None, reads from config.
            pooling: Strategy for pooling patch tokens:
                     - "cls": Use [CLS] token (default, recommended)
                     - "mean": Mean of all patch tokens
                     - "mean_cls": Mean of [CLS] and patch tokens

        Example:
            >>> encoder = DINOEncoder()  # Uses config
            >>> encoder = DINOEncoder(model_name="dinov2_vitb14", pooling="mean")
        """
        # Get model name from config if not provided
        if model_name is None:
            settings = get_settings()
            model_name = settings.models.dino.model_name

        self._model_name = model_name
        self._embedding_dim = DINO_EMBEDDING_DIMS.get(model_name, 384)
        self._pooling = pooling

        # Initialize base class (handles device resolution)
        super().__init__(device=device)

        # Will be set when model is loaded
        self._processor = None

    @property
    def model_name(self) -> str:
        """Return the DINOv2 model name."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self._embedding_dim

    @property
    def pooling(self) -> PoolingStrategy:
        """Return the pooling strategy."""
        return self._pooling

    @property
    def hf_model_id(self) -> str:
        """Return the Hugging Face model ID."""
        return DINO_MODEL_MAPPING.get(
            self._model_name,
            "facebook/dinov2-small",  # Default fallback
        )

    def load_model(self) -> None:
        """
        Load the DINOv2 model and processor from Hugging Face.

        Downloads the model on first use (cached afterwards).

        Raises:
            ModelLoadError: If model cannot be loaded.

        Example:
            >>> encoder = DINOEncoder()
            >>> encoder.load_model()
            >>> print(encoder.is_loaded())
            True
        """
        if self._is_loaded:
            logger.debug(f"DINOv2 model {self._model_name} already loaded")
            return

        logger.info(f"Loading DINOv2 model: {self._model_name} ({self.hf_model_id})")

        try:
            # Import here to avoid slow startup if not used
            from transformers import AutoImageProcessor, AutoModel

            # Load processor (handles image preprocessing)
            self._processor = AutoImageProcessor.from_pretrained(self.hf_model_id)

            # Load model
            self._model = AutoModel.from_pretrained(self.hf_model_id)

            # Move to device
            self._model = self._model.to(self.torch_device)

            # Set to evaluation mode
            self._model.eval()

            self._is_loaded = True
            logger.info(
                f"DINOv2 model loaded successfully on {self._device} "
                f"(embedding_dim={self._embedding_dim}, pooling={self._pooling})"
            )

        except Exception as e:
            logger.error(f"Failed to load DINOv2 model: {e}")
            raise ModelLoadError(
                f"Failed to load DINOv2 model '{self._model_name}': {e}",
                model_name=self._model_name,
            )

    def _pool_features(self, outputs) -> torch.Tensor:
        """
        Pool the model outputs into a single embedding vector.

        DINOv2 outputs include:
        - last_hidden_state: (batch, num_patches + 1, hidden_dim)
          First token is [CLS], rest are patch tokens

        Args:
            outputs: Model outputs from forward pass.

        Returns:
            Pooled tensor of shape (batch, hidden_dim).
        """
        hidden_states = outputs.last_hidden_state

        if self._pooling == "cls":
            # Use [CLS] token (first token)
            return hidden_states[:, 0, :]

        elif self._pooling == "mean":
            # Mean of patch tokens (exclude [CLS])
            patch_tokens = hidden_states[:, 1:, :]
            return patch_tokens.mean(dim=1)

        elif self._pooling == "mean_cls":
            # Mean of all tokens including [CLS]
            return hidden_states.mean(dim=1)

        else:
            # Default to CLS
            return hidden_states[:, 0, :]

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
            (384,)
            >>> print(np.linalg.norm(embedding))  # Should be ~1.0
            1.0
        """
        self._validate_loaded()

        # Load and preprocess image
        pil_image = self._load_image(image)

        try:
            # Process image for DINOv2
            inputs = self._processor(
                images=pil_image,
                return_tensors="pt",
            )

            # Move inputs to device
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

            # Forward pass
            outputs = self._model(**inputs)

            # Pool features to single vector
            pooled = self._pool_features(outputs)

            # Convert to numpy and normalize
            embedding = pooled.cpu().numpy().flatten()

            # L2 normalize
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            return embedding.astype(np.float32)

        except Exception as e:
            logger.error(f"DINOv2 encoding failed: {e}")
            raise EncodingError(
                f"Failed to encode image with DINOv2: {e}",
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
                    # Create placeholder image
                    pil_images.append(Image.new("RGB", (224, 224)))

            try:
                # Process batch
                inputs = self._processor(
                    images=pil_images,
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

                # Forward pass
                outputs = self._model(**inputs)

                # Pool features
                pooled = self._pool_features(outputs)
                embeddings = pooled.cpu().numpy()

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
        logger.info(f"DINOv2 batch encoding complete: {result.shape}")

        return result

    def get_patch_features(
        self,
        image: Union[str, Path, Image.Image],
    ) -> np.ndarray:
        """
        Get individual patch token features (for advanced use).

        Useful for fine-grained analysis like:
        - Attention visualization
        - Part-based matching
        - Segmentation tasks

        Args:
            image: Image path or PIL Image.

        Returns:
            Numpy array of shape (num_patches, hidden_dim).
            For 224x224 images with patch_size=14: (256, hidden_dim).
        """
        self._validate_loaded()

        pil_image = self._load_image(image)

        try:
            inputs = self._processor(
                images=pil_image,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Get patch tokens (exclude [CLS])
            patch_tokens = outputs.last_hidden_state[:, 1:, :]

            return patch_tokens.cpu().numpy().squeeze(0).astype(np.float32)

        except Exception as e:
            logger.error(f"Failed to get patch features: {e}")
            raise EncodingError(
                f"Failed to get patch features: {e}",
                model_name=self._model_name,
            )

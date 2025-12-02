"""
Abstract interface for image encoders.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image


class EncoderInterface(ABC):
    """
    Abstract base class for image encoders.
    
    Defines the contract that CLIP and DINOv2 encoders must implement.
    """

    @abstractmethod
    def encode(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Generate embedding vector from an image.
        
        Args:
            image: Image path (str/Path) or PIL Image object.
            
        Returns:
            Numpy array containing the embedding vector.
        """
        pass

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass

    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model into memory."""
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        pass

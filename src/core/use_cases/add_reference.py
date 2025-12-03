# Add Reference Use Case
"""
Use case for adding reference images to user profile.
Encodes images with hybrid encoder and stores in ChromaDB.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from PIL import Image
import uuid
import logging

from src.core.encoders.hybrid_encoder import HybridEncoder
from src.infrastructure.database.chroma_repository import ChromaRepository

logger = logging.getLogger(__name__)


@dataclass
class AddReferenceResult:
    """Result of adding a reference image."""
    success: bool
    reference_id: str
    message: str
    clip_embedding_size: int = 0
    dino_embedding_size: int = 0


@dataclass
class ReferenceImageData:
    """Data for a reference image."""
    image: Image.Image
    name: str
    category: Optional[str] = None
    tags: Optional[list[str]] = None


class AddReferenceUseCase:
    """
    Use case for adding reference images to user profile.
    
    This encapsulates the business logic of:
    1. Generating hybrid embeddings for reference images
    2. Storing them in the user_references collection
    3. Updating user preference vectors
    """
    
    def __init__(
        self,
        encoder: HybridEncoder,
        repository: ChromaRepository
    ):
        """
        Initialize the use case.
        
        Args:
            encoder: Hybrid encoder for generating embeddings
            repository: ChromaDB repository for storage
        """
        self.encoder = encoder
        self.repository = repository
        
    def execute(
        self,
        image: Union[Image.Image, Path, str],
        name: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list[str]] = None
    ) -> AddReferenceResult:
        """
        Add a reference image to the user profile.
        
        Args:
            image: PIL Image, file path, or path string
            name: Optional name for the reference
            category: Optional category (e.g., "haut", "pantalon", "chaussures")
            tags: Optional list of style tags
            
        Returns:
            AddReferenceResult with success status and details
        """
        try:
            # Convert to PIL Image if needed
            pil_image = self._ensure_pil_image(image)
            
            # Generate unique ID
            reference_id = f"ref_{uuid.uuid4().hex[:12]}"
            
            # Generate filename if not provided
            if name is None:
                name = f"reference_{reference_id}"
            
            # Generate hybrid embeddings
            logger.info(f"Generating embeddings for reference: {name}")
            embeddings = self.encoder.encode(pil_image)
            
            clip_embedding = embeddings["clip"]
            dino_embedding = embeddings["dino"]
            
            # Prepare metadata
            metadata = {
                "type": "reference",
                "name": name,
                "category": category or "general",
                "tags": ",".join(tags) if tags else "",
            }
            
            # Store in user_references collection
            logger.info(f"Storing reference {reference_id} in database")
            self.repository.add_reference(
                reference_id=reference_id,
                clip_embedding=clip_embedding.tolist(),
                dino_embedding=dino_embedding.tolist(),
                metadata=metadata
            )
            
            logger.info(f"Successfully added reference: {name} ({reference_id})")
            
            return AddReferenceResult(
                success=True,
                reference_id=reference_id,
                message=f"Style '{name}' ajouté au profil !",
                clip_embedding_size=len(clip_embedding),
                dino_embedding_size=len(dino_embedding)
            )
            
        except Exception as e:
            logger.error(f"Failed to add reference: {e}")
            return AddReferenceResult(
                success=False,
                reference_id="",
                message=f"Erreur lors de l'ajout: {str(e)}"
            )
    
    def execute_batch(
        self,
        images: list[ReferenceImageData]
    ) -> list[AddReferenceResult]:
        """
        Add multiple reference images at once.
        
        Args:
            images: List of ReferenceImageData objects
            
        Returns:
            List of AddReferenceResult for each image
        """
        results = []
        for img_data in images:
            result = self.execute(
                image=img_data.image,
                name=img_data.name,
                category=img_data.category,
                tags=img_data.tags
            )
            results.append(result)
        return results
    
    def _ensure_pil_image(
        self,
        image: Union[Image.Image, Path, str]
    ) -> Image.Image:
        """Convert input to PIL Image."""
        if isinstance(image, Image.Image):
            return image
        
        path = Path(image) if isinstance(image, str) else image
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        return Image.open(path).convert("RGB")


class ClearReferencesUseCase:
    """Use case for clearing all user references."""
    
    def __init__(self, repository: ChromaRepository):
        """
        Initialize the use case.
        
        Args:
            repository: ChromaDB repository
        """
        self.repository = repository
    
    def execute(self) -> bool:
        """
        Clear all user reference images.
        
        Returns:
            True if successful
        """
        try:
            logger.info("Clearing all user references")
            self.repository.clear_references()
            logger.info("Successfully cleared all references")
            return True
        except Exception as e:
            logger.error(f"Failed to clear references: {e}")
            return False


class GetReferencesUseCase:
    """Use case for retrieving user references."""
    
    def __init__(self, repository: ChromaRepository):
        """
        Initialize the use case.
        
        Args:
            repository: ChromaDB repository
        """
        self.repository = repository
    
    def execute(self) -> list[dict]:
        """
        Get all user reference images metadata.
        
        Returns:
            List of reference metadata dictionaries
        """
        try:
            return self.repository.get_all_references()
        except Exception as e:
            logger.error(f"Failed to get references: {e}")
            return []

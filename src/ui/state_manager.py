"""Session state management for FashionMatch Streamlit UI.

This module manages all persistent state across Streamlit reruns, including
uploaded references, search results, fusion weights, and feedback history.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

from src.database.models import SearchResult
from src.ui.utils import apply_filters
from src.utils.config import FusionWeights


@dataclass
class ReferenceImage:
    """Container for uploaded reference image and its embeddings."""
    
    image: Image.Image
    clip_embedding: Optional[np.ndarray] = None
    dino_embedding: Optional[np.ndarray] = None
    filename: str = ""


@dataclass
class FeedbackEntry:
    """Record of user feedback on a search result."""
    
    item_id: str
    feedback_type: str  # "like" or "dislike"
    timestamp: float
    clip_score: float
    dino_score: float


@dataclass
class FilterSettings:
    """User-defined filter criteria for search results."""
    
    min_price: float = 0.0
    max_price: float = 1000.0
    categories: list[str] = field(default_factory=list)
    min_similarity: float = 0.0
    sort_by: str = "similarity"  # "similarity", "price_asc", "price_desc"


def initialize_session_state() -> None:
    """Initialize all session state variables with defaults."""
    
    if "references" not in st.session_state:
        st.session_state.references = []
    
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    
    if "fusion_weights" not in st.session_state:
        # Will be set from config after app initialization
        st.session_state.fusion_weights = None
    
    if "default_weights" not in st.session_state:
        # Store original weights for reset functionality
        st.session_state.default_weights = None
    
    if "feedback_history" not in st.session_state:
        st.session_state.feedback_history = []
    
    if "filter_settings" not in st.session_state:
        st.session_state.filter_settings = FilterSettings()
    
    if "search_performed" not in st.session_state:
        st.session_state.search_performed = False
    
    if "weights_changed" not in st.session_state:
        st.session_state.weights_changed = False


def add_reference_image(image: Image.Image, filename: str = "") -> None:
    """Add a reference image to session state.
    
    Args:
        image: PIL Image object
        filename: Original filename
    """
    ref = ReferenceImage(image=image, filename=filename)
    st.session_state.references.append(ref)


def update_reference_embeddings(
    index: int,
    clip_embedding: np.ndarray,
    dino_embedding: np.ndarray
) -> None:
    """Update embeddings for a reference image.
    
    Args:
        index: Index of reference in list
        clip_embedding: CLIP embedding vector
        dino_embedding: DINOv2 embedding vector
    """
    if 0 <= index < len(st.session_state.references):
        st.session_state.references[index].clip_embedding = clip_embedding
        st.session_state.references[index].dino_embedding = dino_embedding


def clear_references() -> None:
    """Clear all uploaded reference images."""
    st.session_state.references = []
    st.session_state.search_results = []
    st.session_state.search_performed = False


def update_search_results(results: list[SearchResult]) -> None:
    """Cache latest search results.
    
    Args:
        results: List of SearchResult objects from vector store
    """
    st.session_state.search_results = results
    st.session_state.search_performed = True


def record_feedback(
    item_id: str,
    feedback_type: str,
    clip_score: float,
    dino_score: float
) -> None:
    """Record user feedback on a search result.
    
    Args:
        item_id: ID of the item
        feedback_type: "like" or "dislike"
        clip_score: CLIP similarity score
        dino_score: DINOv2 similarity score
    """
    entry = FeedbackEntry(
        item_id=item_id,
        feedback_type=feedback_type,
        timestamp=time.time(),
        clip_score=clip_score,
        dino_score=dino_score
    )
    st.session_state.feedback_history.append(entry)


def adjust_fusion_weights(
    feedback_type: str,
    clip_score: float,
    dino_score: float,
    adjustment_rate: float = 0.05
) -> FusionWeights:
    """Adjust fusion weights based on user feedback.
    
    Strategy: Increase weight of better-performing model on likes,
    decrease on dislikes. Use small increments and re-normalize.
    
    Args:
        feedback_type: "like" or "dislike"
        clip_score: CLIP similarity score for this item
        dino_score: DINOv2 similarity score for this item
        adjustment_rate: How much to adjust (default 0.05)
        
    Returns:
        New FusionWeights object
    """
    current = st.session_state.fusion_weights
    
    # Determine which model performed better
    clip_better = clip_score > dino_score
    
    # Calculate adjustments
    if feedback_type == "like":
        # Increase weight of better model
        if clip_better:
            new_alpha = min(1.0, current.clip + adjustment_rate)
        else:
            new_alpha = max(0.0, current.clip - adjustment_rate)
    else:  # dislike
        # Decrease weight of better model (or increase worse)
        if clip_better:
            new_alpha = max(0.0, current.clip - adjustment_rate)
        else:
            new_alpha = min(1.0, current.clip + adjustment_rate)
    
    # Re-normalize to sum to 1.0
    new_beta = 1.0 - new_alpha
    
    # Create new weights
    new_weights = FusionWeights(clip=new_alpha, dino=new_beta)
    st.session_state.fusion_weights = new_weights
    st.session_state.weights_changed = True
    
    return new_weights


def reset_fusion_weights() -> None:
    """Reset fusion weights to default values from config."""
    if st.session_state.default_weights is not None:
        st.session_state.fusion_weights = FusionWeights(
            clip=st.session_state.default_weights.clip,
            dino=st.session_state.default_weights.dino
        )


def update_filter_settings(settings: FilterSettings) -> None:
    """Update filter settings.
    
    Args:
        settings: New filter settings
    """
    st.session_state.filter_settings = settings


def get_filtered_results() -> list[SearchResult]:
    """Apply filters to cached search results.
    
    Returns:
        Filtered list of SearchResult objects
    """
    results = st.session_state.search_results
    settings = st.session_state.filter_settings
    
    if not results:
        return []
    
    # Delegate to apply_filters utility
    return apply_filters(results, settings)


def get_reference_embeddings() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Get averaged embeddings from all reference images.
    
    Returns:
        Tuple of (clip_embedding, dino_embedding) or (None, None) if not ready
    """
    refs = st.session_state.references
    
    if not refs:
        return None, None
    
    # Check if all references have embeddings
    if any(r.clip_embedding is None or r.dino_embedding is None for r in refs):
        return None, None
    
    # Average embeddings
    clip_embeddings = np.stack([r.clip_embedding for r in refs])
    dino_embeddings = np.stack([r.dino_embedding for r in refs])
    
    avg_clip = np.mean(clip_embeddings, axis=0).astype(np.float32)
    avg_dino = np.mean(dino_embeddings, axis=0).astype(np.float32)
    
    return avg_clip, avg_dino


def has_encoded_references() -> bool:
    """Check if references are uploaded and encoded.
    
    Returns:
        True if ready for search
    """
    refs = st.session_state.references
    if not refs:
        return False
    return all(r.clip_embedding is not None and r.dino_embedding is not None for r in refs)

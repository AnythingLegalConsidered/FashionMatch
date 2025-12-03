# Session Manager
"""
Streamlit session state management for FashionMatch.

Handles:
- User profile persistence across page navigation
- Model caching with @st.cache_resource
- Reference image tracking
- Fusion weights configuration
"""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np
import streamlit as st


# ============================================
# Data Classes
# ============================================

@dataclass
class ReferenceImages:
    """Container for user reference images info."""
    count: int = 0
    categories: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    last_updated: Optional[str] = None


@dataclass  
class UserProfile:
    """User profile with style preferences."""
    reference_images: ReferenceImages = field(default_factory=ReferenceImages)
    clip_preference_vector: Optional[np.ndarray] = None
    dino_preference_vector: Optional[np.ndarray] = None
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def has_preferences(self) -> bool:
        """Check if user has set any preferences."""
        return self.reference_images.count > 0


@dataclass
class FusionConfig:
    """Fusion weights configuration."""
    clip_weight: float = 0.5
    dino_weight: float = 0.5
    
    def normalize(self):
        """Normalize weights to sum to 1."""
        total = self.clip_weight + self.dino_weight
        if total > 0:
            self.clip_weight /= total
            self.dino_weight /= total


# ============================================
# Cached Resource Loading
# ============================================

@st.cache_resource(show_spinner="Chargement des modèles IA...")
def _load_encoder():
    """Load HybridEncoder (cached across sessions)."""
    try:
        from src.core.encoders.hybrid_encoder import HybridEncoder
        return HybridEncoder()
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles: {e}")
        return None


@st.cache_resource(show_spinner="Connexion à la base de données...")
def _load_repository():
    """Load ChromaRepository (cached across sessions)."""
    try:
        from src.infrastructure.database.chroma_repository import (
            ChromaRepository,
            CHROMADB_AVAILABLE,
        )
        
        if not CHROMADB_AVAILABLE:
            st.warning(
                "ChromaDB n'est pas disponible avec Python 3.14. "
                "Utilisez Python 3.12 ou antérieur pour les fonctionnalités complètes."
            )
            return None
        
        return ChromaRepository()
    except ImportError as e:
        st.warning(f"ChromaDB non disponible: {e}")
        return None
    except Exception as e:
        st.error(f"Erreur de connexion à la base: {e}")
        return None


@st.cache_resource(show_spinner="Initialisation du scorer...")
def _load_scorer():
    """Load WeightedScorer (cached across sessions)."""
    try:
        from src.core.scoring.weighted_scorer import WeightedScorer
        return WeightedScorer()
    except Exception as e:
        st.error(f"Erreur lors du chargement du scorer: {e}")
        return None


# ============================================
# Session Manager
# ============================================

class SessionManager:
    """
    Centralized session state manager for Streamlit.
    
    Provides:
    - Singleton pattern for state consistency
    - Cached model loading
    - User profile management
    - Fusion configuration
    
    Usage:
        >>> manager = SessionManager.get_instance()
        >>> encoder = manager.get_encoder()
        >>> profile = manager.get_profile()
    """
    
    _instance: Optional["SessionManager"] = None
    
    # Session state keys
    PROFILE_KEY = "user_profile"
    FUSION_KEY = "fusion_config"
    ENCODER_KEY = "encoder_loaded"
    REPO_KEY = "repository_loaded"
    
    def __init__(self):
        """Initialize session manager."""
        self._initialize_state()
    
    @classmethod
    def get_instance(cls) -> "SessionManager":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _initialize_state(self):
        """Initialize session state with defaults."""
        if self.PROFILE_KEY not in st.session_state:
            st.session_state[self.PROFILE_KEY] = UserProfile()
        
        if self.FUSION_KEY not in st.session_state:
            st.session_state[self.FUSION_KEY] = FusionConfig()
    
    # =========================================
    # Resource Access
    # =========================================
    
    def get_encoder(self):
        """Get cached HybridEncoder."""
        return _load_encoder()
    
    def get_repository(self):
        """Get cached ChromaRepository."""
        return _load_repository()
    
    def get_scorer(self):
        """Get cached WeightedScorer."""
        return _load_scorer()
    
    # =========================================
    # Profile Management
    # =========================================
    
    def get_profile(self) -> UserProfile:
        """Get current user profile."""
        return st.session_state[self.PROFILE_KEY]
    
    def update_profile(self, profile: UserProfile):
        """Update user profile."""
        st.session_state[self.PROFILE_KEY] = profile
    
    def refresh_profile_from_db(self):
        """Refresh profile data from database."""
        repo = self.get_repository()
        if repo is None:
            return
        
        try:
            references = repo.get_all_references()
            profile = self.get_profile()
            
            # Update reference images info
            categories = set()
            tags = set()
            
            for ref in references:
                if ref.get("category"):
                    categories.add(ref["category"])
                if ref.get("tags"):
                    for tag in ref["tags"].split(","):
                        if tag.strip():
                            tags.add(tag.strip())
            
            profile.reference_images = ReferenceImages(
                count=len(references),
                categories=list(categories),
                tags=list(tags),
            )
            
            # Compute preference vectors
            self._compute_preference_vectors(profile)
            
            self.update_profile(profile)
            
        except Exception as e:
            st.error(f"Erreur lors du rafraîchissement du profil: {e}")
    
    def _compute_preference_vectors(self, profile: UserProfile):
        """Compute average preference vectors from references."""
        repo = self.get_repository()
        if repo is None:
            return
        
        try:
            clip_embeddings, dino_embeddings = repo.get_reference_embeddings()
            
            if clip_embeddings:
                profile.clip_preference_vector = np.mean(
                    np.array(clip_embeddings), axis=0
                )
            
            if dino_embeddings:
                profile.dino_preference_vector = np.mean(
                    np.array(dino_embeddings), axis=0
                )
                
        except Exception as e:
            st.error(f"Erreur lors du calcul des vecteurs de préférence: {e}")
    
    # =========================================
    # Fusion Configuration
    # =========================================
    
    def get_fusion_config(self) -> FusionConfig:
        """Get current fusion configuration."""
        return st.session_state[self.FUSION_KEY]
    
    def update_fusion_config(self, clip_weight: float, dino_weight: float):
        """Update fusion weights."""
        config = FusionConfig(clip_weight=clip_weight, dino_weight=dino_weight)
        config.normalize()
        st.session_state[self.FUSION_KEY] = config
    
    # =========================================
    # Feedback Management
    # =========================================
    
    def add_feedback(self, item_id: str, liked: bool):
        """Record user feedback on an item."""
        profile = self.get_profile()
        profile.feedback_history.append({
            "item_id": item_id,
            "liked": liked,
        })
        self.update_profile(profile)
    
    def get_feedback_stats(self) -> Dict[str, int]:
        """Get feedback statistics."""
        profile = self.get_profile()
        likes = sum(1 for f in profile.feedback_history if f["liked"])
        dislikes = len(profile.feedback_history) - likes
        return {"likes": likes, "dislikes": dislikes, "total": len(profile.feedback_history)}
    
    # =========================================
    # Stats
    # =========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        profile = self.get_profile()
        fusion = self.get_fusion_config()
        repo = self.get_repository()
        
        stats = {
            "reference_count": profile.reference_images.count,
            "has_preferences": profile.has_preferences,
            "clip_weight": fusion.clip_weight,
            "dino_weight": fusion.dino_weight,
            "feedback_count": len(profile.feedback_history),
        }
        
        if repo:
            try:
                stats["items_in_db"] = repo.count()
            except Exception:
                stats["items_in_db"] = 0
        
        return stats

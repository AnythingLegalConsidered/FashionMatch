"""Feedback buttons component for item rating."""

import streamlit as st

from src.database.models import SearchResult
from src.ui.state_manager import adjust_fusion_weights, record_feedback


def render_feedback_buttons(
    item_id: str,
    result: SearchResult,
    key_prefix: str = ""
) -> tuple[bool, bool]:
    """Render like/dislike feedback buttons for an item.
    
    Args:
        item_id: Unique identifier for the item
        result: SearchResult containing scores
        key_prefix: Prefix for button keys to ensure uniqueness
        
    Returns:
        Tuple of (liked, disliked) booleans
    """
    col1, col2 = st.columns(2)
    
    liked = False
    disliked = False
    
    with col1:
        if st.button(
            "👍 Like",
            key=f"{key_prefix}_like_{item_id}",
            use_container_width=True,
            type="secondary"
        ):
            liked = True
            # Record feedback
            record_feedback(
                item_id=item_id,
                feedback_type="like",
                clip_score=result.clip_score,
                dino_score=result.dino_score
            )
            
            # Adjust weights
            new_weights = adjust_fusion_weights(
                feedback_type="like",
                clip_score=result.clip_score,
                dino_score=result.dino_score
            )
            
            st.success(
                f"✅ Feedback recorded! Weights updated: "
                f"CLIP={new_weights.clip:.2f}, DINO={new_weights.dino:.2f}"
            )
    
    with col2:
        if st.button(
            "👎 Dislike",
            key=f"{key_prefix}_dislike_{item_id}",
            use_container_width=True,
            type="secondary"
        ):
            disliked = True
            # Record feedback
            record_feedback(
                item_id=item_id,
                feedback_type="dislike",
                clip_score=result.clip_score,
                dino_score=result.dino_score
            )
            
            # Adjust weights
            new_weights = adjust_fusion_weights(
                feedback_type="dislike",
                clip_score=result.clip_score,
                dino_score=result.dino_score
            )
            
            st.info(
                f"📝 Feedback recorded! Weights updated: "
                f"CLIP={new_weights.clip:.2f}, DINO={new_weights.dino:.2f}"
            )
    
    return liked, disliked

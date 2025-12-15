"""Statistics panel component for search results overview."""

import streamlit as st

from src.database.models import SearchResult


def render_stats_panel(
    results: list[SearchResult],
    feedback_history: list
) -> None:
    """Render statistics dashboard for search results.
    
    Args:
        results: List of SearchResult objects
        feedback_history: List of FeedbackEntry objects
    """
    if not results:
        return
    
    # Calculate statistics
    total_results = len(results)
    avg_similarity = sum(r.similarity_score for r in results) / total_results
    
    # Feedback statistics
    likes = sum(1 for f in feedback_history if f.feedback_type == "like")
    dislikes = sum(1 for f in feedback_history if f.feedback_type == "dislike")
    
    # Model performance (which model has higher avg score in liked items)
    liked_items = [f for f in feedback_history if f.feedback_type == "like"]
    if liked_items:
        avg_clip_liked = sum(f.clip_score for f in liked_items) / len(liked_items)
        avg_dino_liked = sum(f.dino_score for f in liked_items) / len(liked_items)
        better_model = "CLIP" if avg_clip_liked > avg_dino_liked else "DINOv2"
    else:
        better_model = "N/A"
    
    # Display in columns
    st.markdown("### 📊 Search Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Results",
            value=total_results,
            help="Number of items returned"
        )
    
    with col2:
        st.metric(
            label="Avg Similarity",
            value=f"{avg_similarity * 100:.1f}%",
            help="Average fused similarity score"
        )
    
    with col3:
        st.metric(
            label="Feedback",
            value=f"👍 {likes} / 👎 {dislikes}",
            help="Likes and dislikes"
        )
    
    with col4:
        st.metric(
            label="Top Model",
            value=better_model,
            help="Model with higher avg score in liked items"
        )
    
    st.markdown("---")

# Image Card Component
"""
Reusable card component for displaying clothing items with scores.

Features:
- Image display (local or URL fallback)
- Item metadata (title, price, size, brand)
- Match score progress bar
- Like/Dislike feedback buttons with unique keys
"""
import streamlit as st
from typing import Callable, Optional
from pathlib import Path

from src.domain.entities.clothing_item import ClothingItem


# ============================================
# Styling
# ============================================

CARD_CSS = """
<style>
.clothing-card {
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    background-color: #fafafa;
}
.clothing-card:hover {
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}
.match-score-high {
    color: #28a745;
    font-weight: bold;
}
.match-score-medium {
    color: #ffc107;
    font-weight: bold;
}
.match-score-low {
    color: #dc3545;
    font-weight: bold;
}
</style>
"""


def _get_score_class(score: float) -> str:
    """Get CSS class based on score value."""
    if score >= 0.7:
        return "match-score-high"
    elif score >= 0.4:
        return "match-score-medium"
    return "match-score-low"


def _get_score_emoji(score: float) -> str:
    """Get emoji based on score value."""
    if score >= 0.8:
        return "🔥"
    elif score >= 0.6:
        return "✨"
    elif score >= 0.4:
        return "👍"
    return "🤔"


# ============================================
# Image Loading
# ============================================

def _get_image_source(item: ClothingItem) -> Optional[str]:
    """
    Get the best available image source for an item.
    
    Priority:
    1. Local image path (if exists)
    2. Image URL
    
    Args:
        item: The clothing item.
        
    Returns:
        Image path/URL or None if no image available.
    """
    # Try local path first
    if item.local_image_path:
        local_path = Path(item.local_image_path)
        if local_path.exists():
            return str(local_path)
    
    # Fall back to URL
    if item.image_url:
        return item.image_url
    
    return None


# ============================================
# Main Component
# ============================================

def render_image_card(
    item: ClothingItem,
    score: float,
    on_like: Optional[Callable[[str], None]] = None,
    on_dislike: Optional[Callable[[str], None]] = None,
    show_feedback_buttons: bool = True,
    compact: bool = False,
) -> None:
    """
    Render a clothing item card with image, metadata, and feedback buttons.
    
    Args:
        item: The clothing item to display.
        score: Match score between 0 and 1.
        on_like: Callback when like button is clicked (receives item.id).
        on_dislike: Callback when dislike button is clicked (receives item.id).
        show_feedback_buttons: Whether to show like/dislike buttons.
        compact: Use compact layout (smaller image, less details).
        
    Example:
        >>> def handle_like(item_id):
        ...     st.toast(f"Liked {item_id}!")
        >>> 
        >>> render_image_card(item, score=0.85, on_like=handle_like)
    """
    # Inject CSS once per session
    if "card_css_injected" not in st.session_state:
        st.markdown(CARD_CSS, unsafe_allow_html=True)
        st.session_state.card_css_injected = True
    
    # Clamp score to valid range
    score = max(0.0, min(1.0, score))
    score_percent = int(score * 100)
    score_emoji = _get_score_emoji(score)
    
    # Get image source
    image_source = _get_image_source(item)
    
    # Card container with border effect using expander-like styling
    with st.container():
        # Use columns for layout: Image | Details
        if compact:
            img_col, detail_col = st.columns([1, 2])
        else:
            img_col, detail_col = st.columns([1, 1.5])
        
        # =========================================
        # Left Column: Image
        # =========================================
        with img_col:
            if image_source:
                try:
                    st.image(
                        image_source,
                        use_container_width=True,
                        caption=None,
                    )
                except Exception:
                    st.image(
                        "https://via.placeholder.com/200x250?text=Image+non+disponible",
                        use_container_width=True,
                    )
            else:
                # Placeholder for missing image
                st.markdown(
                    """
                    <div style="
                        background: #f0f0f0;
                        height: 200px;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        border-radius: 8px;
                        color: #888;
                    ">
                        📷 Pas d'image
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        
        # =========================================
        # Right Column: Details
        # =========================================
        with detail_col:
            # Title
            title = item.title or "Article sans titre"
            if len(title) > 50:
                title = title[:47] + "..."
            st.markdown(f"**{title}**")
            
            # Price and Size row
            price_str = f"{item.price:.2f} {item.currency}" if item.price else "Prix non renseigné"
            size_str = f"Taille: {item.size}" if item.size else ""
            
            col_price, col_size = st.columns(2)
            with col_price:
                st.markdown(f"💰 **{price_str}**")
            with col_size:
                if size_str:
                    st.markdown(f"📏 {size_str}")
            
            # Brand and Condition (if not compact)
            if not compact:
                if item.brand:
                    st.caption(f"🏷️ {item.brand}")
                if item.condition:
                    st.caption(f"📦 {item.condition}")
            
            # Match Score Progress Bar
            st.markdown(f"{score_emoji} **Match: {score_percent}%**")
            st.progress(score, text=None)
            
            # Feedback Buttons
            if show_feedback_buttons:
                btn_col1, btn_col2, spacer = st.columns([1, 1, 2])
                
                with btn_col1:
                    if st.button(
                        "👍",
                        key=f"btn_like_{item.id}",
                        use_container_width=True,
                        help="J'aime ce style",
                    ):
                        if on_like:
                            on_like(item.id)
                        st.toast(f"👍 {title[:20]}... ajouté aux favoris!")
                
                with btn_col2:
                    if st.button(
                        "👎",
                        key=f"btn_dislike_{item.id}",
                        use_container_width=True,
                        help="Pas mon style",
                    ):
                        if on_dislike:
                            on_dislike(item.id)
                        st.toast(f"👎 Noté, on adapte les recommandations!")
            
            # Link to original item
            if item.item_url:
                st.markdown(
                    f"[🔗 Voir sur Vinted]({item.item_url})",
                    unsafe_allow_html=True,
                )
        
        # Visual separator
        st.divider()


# ============================================
# Grid Layout Helper
# ============================================

def render_image_grid(
    items: list[tuple[ClothingItem, float]],
    columns: int = 2,
    on_like: Optional[Callable[[str], None]] = None,
    on_dislike: Optional[Callable[[str], None]] = None,
    show_feedback_buttons: bool = True,
) -> None:
    """
    Render a grid of clothing item cards.
    
    Args:
        items: List of (ClothingItem, score) tuples.
        columns: Number of columns in the grid.
        on_like: Callback for like button.
        on_dislike: Callback for dislike button.
        show_feedback_buttons: Whether to show feedback buttons.
        
    Example:
        >>> items_with_scores = [(item1, 0.9), (item2, 0.75), (item3, 0.6)]
        >>> render_image_grid(items_with_scores, columns=3)
    """
    if not items:
        st.info("Aucun article à afficher.")
        return
    
    # Create grid layout
    cols = st.columns(columns)
    
    for idx, (item, score) in enumerate(items):
        col_idx = idx % columns
        
        with cols[col_idx]:
            render_image_card(
                item=item,
                score=score,
                on_like=on_like,
                on_dislike=on_dislike,
                show_feedback_buttons=show_feedback_buttons,
                compact=True,  # Use compact mode in grid
            )


# ============================================
# Skeleton Loader
# ============================================

def render_card_skeleton(count: int = 3) -> None:
    """
    Render placeholder skeleton cards while loading.
    
    Args:
        count: Number of skeleton cards to show.
    """
    for _ in range(count):
        with st.container():
            col1, col2 = st.columns([1, 1.5])
            
            with col1:
                st.markdown(
                    """
                    <div style="
                        background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                        height: 180px;
                        border-radius: 8px;
                        animation: pulse 1.5s infinite;
                    "></div>
                    """,
                    unsafe_allow_html=True,
                )
            
            with col2:
                st.markdown("⏳ Chargement...")
                st.progress(0.5)
            
            st.divider()

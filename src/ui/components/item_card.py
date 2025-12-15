"""Item card component for displaying fashion items."""

from pathlib import Path

import streamlit as st

from src.database.models import SearchResult
from src.ui.components.feedback_buttons import render_feedback_buttons
from src.ui.utils import (
    format_price,
    format_similarity_score,
    get_score_color,
    load_image_from_path,
    truncate_text,
)


def render_item_card(
    result: SearchResult,
    show_scores: bool = True,
    key_prefix: str = ""
) -> None:
    """Render a fashion item card with image, metadata, and feedback buttons.
    
    Args:
        result: SearchResult object containing item and scores
        show_scores: Whether to display similarity scores
        key_prefix: Prefix for component keys
    """
    item = result.item
    
    # Container for card
    with st.container():
        st.markdown("---")
        
        # Main layout: image on left, details on right
        col_img, col_details = st.columns([1, 2])
        
        # Image column
        with col_img:
            # Try to load image from local path first, then URL
            image = None
            if item.local_image_path:
                image = load_image_from_path(item.local_image_path)
            
            if image:
                st.image(image, use_container_width=True)
            elif item.image_url:
                try:
                    st.image(item.image_url, use_container_width=True)
                except Exception:
                    # Local fallback for failed image loads
                    st.markdown(
                        '<div style="background-color: #f0f0f0; padding: 4rem 2rem; '
                        'text-align: center; border-radius: 8px; color: #999;">'
                        '<div style="font-size: 3rem; margin-bottom: 0.5rem;">📷</div>'
                        '<div>No Image Available</div></div>',
                        unsafe_allow_html=True
                    )
            else:
                # Local fallback for missing image paths
                st.markdown(
                    '<div style="background-color: #f0f0f0; padding: 4rem 2rem; '
                    'text-align: center; border-radius: 8px; color: #999;">'
                    '<div style="font-size: 3rem; margin-bottom: 0.5rem;">📷</div>'
                    '<div>No Image Available</div></div>',
                    unsafe_allow_html=True
                )
        
        # Details column
        with col_details:
            # Title
            st.markdown(f"### {truncate_text(item.title, 60)}")
            
            # Metadata row
            metadata_parts = []
            if item.brand:
                metadata_parts.append(f"🏷️ **{item.brand}**")
            if item.category:
                metadata_parts.append(f"📁 {item.category}")
            
            if metadata_parts:
                st.markdown(" | ".join(metadata_parts))
            
            # Price
            if item.price is not None:
                st.markdown(f"### {format_price(item.price)}")
            
            # Scores
            if show_scores:
                st.markdown("**Similarity Scores:**")
                
                score_col1, score_col2, score_col3 = st.columns(3)
                
                with score_col1:
                    color = get_score_color(result.similarity_score)
                    st.markdown(
                        f'<div style="background-color: {color}20; padding: 0.5rem; '
                        f'border-radius: 8px; text-align: center;">'
                        f'<div style="font-size: 1.5rem; font-weight: 700; color: {color};">'
                        f'{format_similarity_score(result.similarity_score)}</div>'
                        f'<div style="font-size: 0.8rem; color: #666;">Fused</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with score_col2:
                    st.markdown(
                        f'<div style="background-color: #E3F2FD; padding: 0.5rem; '
                        f'border-radius: 8px; text-align: center;">'
                        f'<div style="font-size: 1.2rem; font-weight: 600; color: #1976D2;">'
                        f'{format_similarity_score(result.clip_score)}</div>'
                        f'<div style="font-size: 0.8rem; color: #666;">CLIP</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                
                with score_col3:
                    st.markdown(
                        f'<div style="background-color: #E8F5E9; padding: 0.5rem; '
                        f'border-radius: 8px; text-align: center;">'
                        f'<div style="font-size: 1.2rem; font-weight: 600; color: #388E3C;">'
                        f'{format_similarity_score(result.dino_score)}</div>'
                        f'<div style="font-size: 0.8rem; color: #666;">DINO</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            
            st.markdown("")  # Spacing
            
            # Action buttons
            btn_col1, btn_col2 = st.columns([2, 1])
            
            with btn_col1:
                if item.url:
                    st.markdown(
                        f'<a href="{item.url}" target="_blank">'
                        f'<button style="background-color: #1976D2; color: white; '
                        f'border: none; padding: 0.5rem 1rem; border-radius: 8px; '
                        f'cursor: pointer; width: 100%;">🔗 View on Vinted</button></a>',
                        unsafe_allow_html=True
                    )
            
            with btn_col2:
                st.markdown("")  # Placeholder for alignment
            
            st.markdown("")  # Spacing
            
            # Feedback buttons
            render_feedback_buttons(
                item_id=item.item_id,
                result=result,
                key_prefix=key_prefix
            )

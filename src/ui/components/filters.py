"""Filter sidebar component."""

import streamlit as st

from src.ui.state_manager import FilterSettings
from src.ui.utils import get_price_range, get_unique_categories


def render_filters(results: list) -> FilterSettings:
    """Render filter controls in sidebar.
    
    Args:
        results: List of SearchResult objects
        
    Returns:
        FilterSettings object with user selections
    """
    st.sidebar.markdown("### 🔍 Filters")
    
    if not results:
        st.sidebar.info("Perform a search to enable filters")
        return FilterSettings()
    
    # Price range filter
    min_price, max_price = get_price_range(results)
    
    st.sidebar.markdown("#### Price Range")
    price_range = st.sidebar.slider(
        "Select price range",
        min_value=float(min_price),
        max_value=float(max_price),
        value=(float(min_price), float(max_price)),
        step=1.0,
        help="Filter items by price"
    )
    
    # Category filter
    categories = get_unique_categories(results)
    
    if categories:
        st.sidebar.markdown("#### Categories")
        selected_categories = st.sidebar.multiselect(
            "Select categories",
            options=categories,
            default=[],
            help="Filter by item category"
        )
    else:
        selected_categories = []
    
    # Similarity threshold
    st.sidebar.markdown("#### Minimum Similarity")
    min_similarity = st.sidebar.slider(
        "Similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        format="%.2f",
        help="Filter items below this similarity score"
    )
    
    # Sort order
    st.sidebar.markdown("#### Sort By")
    sort_by = st.sidebar.selectbox(
        "Sort results by",
        options=["similarity", "price_asc", "price_desc"],
        format_func=lambda x: {
            "similarity": "Similarity (High to Low)",
            "price_asc": "Price (Low to High)",
            "price_desc": "Price (High to Low)"
        }[x],
        help="Sort order for search results"
    )
    
    # Create filter settings
    settings = FilterSettings(
        min_price=price_range[0],
        max_price=price_range[1],
        categories=selected_categories,
        min_similarity=min_similarity,
        sort_by=sort_by
    )
    
    return settings

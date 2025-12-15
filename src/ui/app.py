"""FashionMatch Streamlit UI - Main Application.

An interactive visual search interface for finding similar fashion items
using dual AI models (CLIP + DINOv2) with adaptive fusion weights.
"""

import streamlit as st

from src.core import get_hybrid_scorer
from src.database import get_vector_store
from src.ui.components import (
    render_filters,
    render_item_card,
    render_stats_panel,
    render_weight_display,
)
from src.ui.state_manager import (
    add_reference_image,
    clear_references,
    get_filtered_results,
    get_reference_embeddings,
    has_encoded_references,
    initialize_session_state,
    update_filter_settings,
    update_reference_embeddings,
    update_search_results,
)
from src.ui.styles import get_custom_css
from src.ui.utils import create_thumbnail, load_image_from_upload
from src.utils import get_config, get_logger, log_exception
from src.utils.config import FusionWeights

logger = get_logger(__name__)


# Cache encoder and vector store to avoid reloading
@st.cache_resource
def initialize_models():
    """Initialize and cache AI models and vector store.
    
    Returns:
        Tuple of (config, scorer, vector_store)
    """
    try:
        config = get_config()
        scorer = get_hybrid_scorer(config.models)
        
        clip_dim = scorer.clip_encoder.embedding_dim
        dino_dim = scorer.dino_encoder.embedding_dim
        
        vector_store = get_vector_store(
            config=config.database,
            clip_dim=clip_dim,
            dino_dim=dino_dim
        )
        
        return config, scorer, vector_store
    
    except Exception as e:
        log_exception(logger, "initialize models", e)
        st.error(f"Failed to initialize models: {e}")
        st.stop()


def main():
    """Main application entry point."""
    
    # Page configuration
    st.set_page_config(
        page_title="FashionMatch - Visual Fashion Search",
        page_icon="👗",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Initialize session state
    initialize_session_state()
    
    # Initialize models
    config, scorer, vector_store = initialize_models()
    
    # Set default fusion weights from config if not set
    if st.session_state.fusion_weights is None:
        st.session_state.fusion_weights = config.models.fusion_weights
        st.session_state.default_weights = config.models.fusion_weights
    
    # ==================== HEADER ====================
    st.title("👗 FashionMatch")
    st.markdown("**Visual Fashion Search** powered by CLIP + DINOv2")
    
    # Check if vector store has items
    item_count = vector_store.count()
    
    if item_count == 0:
        st.warning(
            "⚠️ **No items in database!** "
            "Run the embedding pipeline first to populate the vector store:\n\n"
            "`python -m src.core.embedding_pipeline --mode all`"
        )
        st.stop()
    
    st.info(f"📦 Database contains **{item_count:,}** fashion items")
    
    # ==================== SIDEBAR ====================
    with st.sidebar:
        st.markdown("## 📸 Upload References")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload reference images",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Upload one or more reference images to find similar items"
        )
        
        # Handle uploads
        if uploaded_files:
            # Check if new uploads (compare with session state)
            current_filenames = [f.name for f in uploaded_files]
            stored_filenames = [r.filename for r in st.session_state.references]
            
            if current_filenames != stored_filenames:
                # Clear and reload references
                st.session_state.references = []
                
                for uploaded_file in uploaded_files:
                    try:
                        image = load_image_from_upload(uploaded_file)
                        add_reference_image(image, uploaded_file.name)
                    except Exception as e:
                        st.error(f"Failed to load {uploaded_file.name}: {e}")
        
        # Show reference previews
        if st.session_state.references:
            st.markdown(f"**{len(st.session_state.references)} reference(s) uploaded**")
            
            # Show thumbnails in grid
            cols = st.columns(3)
            for idx, ref in enumerate(st.session_state.references):
                with cols[idx % 3]:
                    thumbnail = create_thumbnail(ref.image, (100, 100))
                    st.image(thumbnail, use_container_width=True)
            
            # Encode button
            if not has_encoded_references():
                if st.button("🔄 Encode References", use_container_width=True, type="primary"):
                    with st.spinner("Encoding images with CLIP + DINOv2..."):
                        try:
                            images = [ref.image for ref in st.session_state.references]
                            clip_embs, dino_embs = scorer.encode_dual(images)
                            
                            for idx, (clip_emb, dino_emb) in enumerate(zip(clip_embs, dino_embs)):
                                update_reference_embeddings(idx, clip_emb, dino_emb)
                            
                            st.success("✅ References encoded successfully!")
                            st.rerun()
                        
                        except Exception as e:
                            log_exception(logger, "encode references", e)
                            st.error(f"Encoding failed: {e}")
            else:
                st.success("✅ References ready for search")
            
            # Clear button
            if st.button("🗑️ Clear All", use_container_width=True):
                clear_references()
                st.rerun()
        
        st.markdown("---")
        
        # Weight display
        if st.session_state.fusion_weights:
            render_weight_display(st.session_state.fusion_weights)
        
        st.markdown("---")
        
        # Filters
        filter_settings = render_filters(st.session_state.search_results)
        # Persist filter settings to session state
        update_filter_settings(filter_settings)
    
    # ==================== MAIN PANEL ====================
    
    # Search button
    if has_encoded_references():
        if st.button("🔍 Search Similar Items", use_container_width=True, type="primary"):
            with st.spinner("Searching database..."):
                try:
                    # Get averaged embeddings
                    clip_query, dino_query = get_reference_embeddings()
                    
                    if clip_query is not None and dino_query is not None:
                        # Perform search
                        results = vector_store.search(
                            clip_query=clip_query,
                            dino_query=dino_query,
                            top_k=50,
                            fusion_weights=st.session_state.fusion_weights
                        )
                        
                        # Update session state
                        update_search_results(results)
                        
                        st.success(f"✅ Found {len(results)} similar items!")
                    else:
                        st.error("Failed to get reference embeddings")
                
                except Exception as e:
                    log_exception(logger, "search", e)
                    st.error(f"Search failed: {e}")
    else:
        st.info("👆 Upload and encode reference images to start searching")
    
    # Update scorer weights if they changed (from feedback)
    if "weights_changed" in st.session_state and st.session_state.weights_changed:
        scorer.update_weights(st.session_state.fusion_weights)
        st.session_state.weights_changed = False
    
    # Display results
    if st.session_state.search_performed:
        # Apply filters
        filtered_results = get_filtered_results()
        
        if not filtered_results:
            st.warning("No results match your filters. Try adjusting the criteria.")
        else:
            # Statistics panel
            render_stats_panel(
                results=filtered_results,
                feedback_history=st.session_state.feedback_history
            )
            
            # Results header
            st.markdown(f"### 🎯 Recommended Items ({len(filtered_results)} results)")
            
            # Pagination
            items_per_page = 12
            total_pages = (len(filtered_results) + items_per_page - 1) // items_per_page
            
            if total_pages > 1:
                page = st.selectbox(
                    "Page",
                    options=range(1, total_pages + 1),
                    format_func=lambda x: f"Page {x} of {total_pages}"
                )
            else:
                page = 1
            
            # Get page slice
            start_idx = (page - 1) * items_per_page
            end_idx = start_idx + items_per_page
            page_results = filtered_results[start_idx:end_idx]
            
            # Display items in grid (3 columns)
            for i in range(0, len(page_results), 3):
                cols = st.columns(3)
                
                for j, col in enumerate(cols):
                    if i + j < len(page_results):
                        with col:
                            result = page_results[i + j]
                            render_item_card(
                                result=result,
                                show_scores=True,
                                key_prefix=f"page{page}_item{i+j}"
                            )
    
    else:
        # Empty state
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem; color: #666;">
                <div style="font-size: 5rem; margin-bottom: 1rem;">🔍</div>
                <h2>No Search Results Yet</h2>
                <p style="font-size: 1.1rem;">Upload reference images and click "Search" to find similar fashion items!</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ==================== FOOTER ====================
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            <small>FashionMatch v1.0 | Powered by CLIP + DINOv2 | Built with Streamlit</small>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

# Browse Page - Découvrir
"""
Page for browsing personalized clothing recommendations.

Displays items ranked by similarity to user's reference style,
with filtering options and feedback buttons.
"""
import streamlit as st
from typing import Optional

# Configure page
st.set_page_config(
    page_title="Découvrir - FashionMatch",
    page_icon="🔍",
    layout="wide",
)


# ============================================
# Session State Initialization
# ============================================

def init_session_state():
    """Initialize session state variables."""
    if "ignored_ids" not in st.session_state:
        st.session_state.ignored_ids = []
    
    if "liked_ids" not in st.session_state:
        st.session_state.liked_ids = []


# ============================================
# Resource Loading
# ============================================

def get_repository():
    """Get ChromaDB repository."""
    try:
        from src.infrastructure.database.chroma_repository import (
            ChromaRepository,
            CHROMADB_AVAILABLE,
        )
        
        if not CHROMADB_AVAILABLE:
            return None
        
        if "repository" not in st.session_state:
            st.session_state.repository = ChromaRepository()
        
        return st.session_state.repository
        
    except Exception as e:
        st.error(f"Erreur de connexion à la base: {e}")
        return None


def get_recommendations_use_case():
    """Get GetRecommendationsUseCase."""
    from src.core.use_cases.get_recommendations import GetRecommendationsUseCase
    
    repo = get_repository()
    if repo is None:
        return None
    
    if "get_recommendations_uc" not in st.session_state:
        st.session_state.get_recommendations_uc = GetRecommendationsUseCase(repo)
    
    return st.session_state.get_recommendations_uc


def get_add_to_references_use_case():
    """Get AddItemToReferencesUseCase."""
    from src.core.use_cases.get_recommendations import AddItemToReferencesUseCase
    
    repo = get_repository()
    if repo is None:
        return None
    
    if "add_to_refs_uc" not in st.session_state:
        st.session_state.add_to_refs_uc = AddItemToReferencesUseCase(repo)
    
    return st.session_state.add_to_refs_uc


# ============================================
# Feedback Handlers
# ============================================

def handle_like(item_id: str):
    """Handle like button click."""
    # Add to references for learning
    add_to_refs = get_add_to_references_use_case()
    
    if add_to_refs and add_to_refs.execute(item_id):
        st.session_state.liked_ids.append(item_id)
        st.toast(f"👍 Ajouté à vos préférences !")
    else:
        st.toast("❌ Erreur lors de l'ajout")
    
    # Rerun to refresh recommendations
    st.rerun()


def handle_dislike(item_id: str):
    """Handle dislike button click."""
    # Add to ignored list
    if item_id not in st.session_state.ignored_ids:
        st.session_state.ignored_ids.append(item_id)
    
    st.toast("👎 Article masqué")
    
    # Rerun to refresh recommendations
    st.rerun()


# ============================================
# UI Components
# ============================================

def render_header():
    """Render page header."""
    st.title("🔍 Découvrir")
    st.markdown(
        """
        Explorez les articles qui correspondent à votre style.
        Les recommandations sont basées sur vos images de référence.
        """
    )
    st.divider()


def render_no_profile_warning():
    """Render warning when user has no profile."""
    st.warning(
        "⚠️ **Veuillez d'abord uploader des photos de référence.**\n\n"
        "Pour obtenir des recommandations personnalisées, vous devez d'abord "
        "définir votre style en ajoutant des images de vêtements que vous aimez."
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("📤 Ajouter des images de référence", type="primary", use_container_width=True):
            st.switch_page("pages/01_upload.py")
    
    # Show demo info
    st.info(
        "💡 **Astuce**: Vous pouvez aussi utiliser la page Paramètres pour "
        "ajuster les poids de fusion CLIP/DINO une fois votre profil créé."
    )


def render_filters() -> dict:
    """Render filter sidebar and return filter values."""
    with st.sidebar:
        st.subheader("🔧 Filtres")
        
        # Category filter
        category = st.selectbox(
            "Catégorie",
            options=["Toutes", "haut", "pantalon", "robe", "jupe", "veste", "manteau", "chaussures", "accessoire"],
            index=0,
        )
        
        # Price range
        st.markdown("**Prix (€)**")
        col1, col2 = st.columns(2)
        with col1:
            min_price = st.number_input("Min", min_value=0.0, value=0.0, step=5.0)
        with col2:
            max_price = st.number_input("Max", min_value=0.0, value=0.0, step=5.0)
        
        # Size filter
        size = st.text_input("Taille", placeholder="Ex: M, 38, L...")
        
        # Number of results
        limit = st.slider("Nombre de résultats", min_value=5, max_value=50, value=20, step=5)
        
        st.divider()
        
        # Fusion weights
        st.subheader("⚖️ Pondération")
        clip_weight = st.slider(
            "CLIP (Sémantique)",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Importance de la similarité sémantique (style, couleurs, textures)"
        )
        dino_weight = 1.0 - clip_weight
        st.caption(f"DINO (Structurel): {dino_weight:.1f}")
        
        st.divider()
        
        # Stats
        st.subheader("📊 Stats session")
        st.metric("Articles masqués", len(st.session_state.ignored_ids))
        st.metric("Articles likés", len(st.session_state.liked_ids))
        
        # Reset ignored
        if st.button("🔄 Réinitialiser les filtres", use_container_width=True):
            st.session_state.ignored_ids = []
            st.session_state.liked_ids = []
            st.rerun()
    
    return {
        "category": category if category != "Toutes" else None,
        "min_price": min_price if min_price > 0 else None,
        "max_price": max_price if max_price > 0 else None,
        "size": size if size else None,
        "limit": limit,
        "clip_weight": clip_weight,
        "dino_weight": dino_weight,
    }


def render_recommendations(filter_values: dict):
    """Fetch and render recommendations."""
    from src.core.use_cases.get_recommendations import RecommendationFilters
    from src.infrastructure.database.chroma_repository import FusionWeights
    from src.ui.components.image_card import render_image_card
    
    # Get use case
    recommendations_uc = get_recommendations_use_case()
    
    if recommendations_uc is None:
        st.error(
            "❌ Impossible de charger le système de recommandations. "
            "Vérifiez que ChromaDB est installé et compatible."
        )
        return
    
    # Build filters
    filters = RecommendationFilters(
        category=filter_values["category"],
        min_price=filter_values["min_price"],
        max_price=filter_values["max_price"],
        size=filter_values["size"],
        ignored_ids=st.session_state.ignored_ids,
    )
    
    # Build weights
    weights = FusionWeights(
        clip=filter_values["clip_weight"],
        dino=filter_values["dino_weight"],
    )
    
    # Execute with loading spinner
    with st.spinner("🔍 Recherche en cours..."):
        result = recommendations_uc.execute(
            limit=filter_values["limit"],
            weights=weights,
            filters=filters,
        )
    
    # Check if user has references
    if result.reference_count == 0:
        render_no_profile_warning()
        return
    
    # Show results info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📌 Références utilisées", result.reference_count)
    with col2:
        st.metric("🔍 Articles trouvés", len(result.items))
    with col3:
        st.metric("🗄️ Base de données", result.total_items_searched)
    
    st.divider()
    
    # Check for empty results
    if not result.items:
        st.info(
            "😕 Aucun article trouvé avec ces critères. "
            "Essayez d'ajuster les filtres ou d'ajouter plus d'images de référence."
        )
        return
    
    # Display results in grid (2 columns)
    st.subheader(f"✨ {len(result.items)} articles pour vous")
    
    # Create 2-column grid
    cols = st.columns(2)
    
    for idx, (item, score) in enumerate(result.items):
        col_idx = idx % 2
        
        with cols[col_idx]:
            render_image_card(
                item=item,
                score=score,
                on_like=handle_like,
                on_dislike=handle_dislike,
                show_feedback_buttons=True,
                compact=False,
            )


def render_no_items_in_db():
    """Render message when database is empty."""
    st.info(
        "📭 **La base de données est vide.**\n\n"
        "Aucun article n'a encore été scrapé. Utilisez le script de scraping "
        "pour ajouter des articles à la base de données."
    )
    
    st.code(
        "python scripts/scrape_test.py",
        language="bash"
    )


# ============================================
# Main Page
# ============================================

def main():
    """Main page entry point."""
    init_session_state()
    render_header()
    
    # Check if repository is available
    repo = get_repository()
    
    if repo is None:
        st.error(
            "❌ ChromaDB n'est pas disponible. "
            "Cette fonctionnalité nécessite Python < 3.14."
        )
        return
    
    # Check if database has items
    try:
        item_count = repo.count()
        if item_count == 0:
            render_no_items_in_db()
            # Still show filters and check for references
            filter_values = render_filters()
            
            # Check references
            ref_count = repo.get_reference_count()
            if ref_count == 0:
                st.divider()
                render_no_profile_warning()
            return
    except Exception as e:
        st.error(f"Erreur de base de données: {e}")
        return
    
    # Render filters and get values
    filter_values = render_filters()
    
    # Render recommendations
    render_recommendations(filter_values)


# Run the page
if __name__ == "__main__":
    main()
else:
    main()

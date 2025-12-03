# FashionMatch - Main Application
"""
Streamlit application entry point for FashionMatch.

This is the main landing page that provides navigation to:
- Browse: Search and discover clothing items
- Upload: Add reference images to define your style
- Settings: Configure fusion weights and preferences

Run with:
    streamlit run src/ui/app.py
"""
import streamlit as st
from pathlib import Path
import sys

# Ensure src is in path
src_path = Path(__file__).parent.parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


# ============================================
# Page Configuration
# ============================================

def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="FashionMatch - Recommandations Vestimentaires IA",
        page_icon="👗",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/AnythingLegalConsidered/FashionMatch",
            "Report a bug": "https://github.com/AnythingLegalConsidered/FashionMatch/issues",
            "About": "# FashionMatch\nRecommandations vestimentaires intelligentes basées sur CLIP + DINOv2.",
        },
    )


# ============================================
# Sidebar
# ============================================

def render_sidebar():
    """Render the sidebar with navigation and stats."""
    with st.sidebar:
        st.title("👗 FashionMatch")
        st.caption("Recommandations IA hybrides")
        
        st.divider()
        
        # Quick stats
        try:
            from src.ui.state.session_manager import SessionManager
            manager = SessionManager.get_instance()
            stats = manager.get_stats()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("📌 Références", stats.get("reference_count", 0))
            with col2:
                st.metric("🗄️ Articles", stats.get("items_in_db", 0))
            
            # Fusion weights display
            st.caption("Pondération actuelle:")
            clip_w = stats.get("clip_weight", 0.5)
            dino_w = stats.get("dino_weight", 0.5)
            st.progress(clip_w, text=f"CLIP: {clip_w:.0%}")
            st.progress(dino_w, text=f"DINO: {dino_w:.0%}")
            
        except Exception:
            st.info("Chargement des statistiques...")
        
        st.divider()
        
        # Navigation info
        st.markdown(
            """
            ### 📍 Navigation
            
            Utilisez les pages dans le menu ci-dessus:
            
            - **🏠 Accueil** - Cette page
            - **👔 Mon Style** - Ajouter des références
            - **🔍 Découvrir** - Parcourir les articles
            - **⚙️ Paramètres** - Configuration
            """
        )


# ============================================
# Main Content
# ============================================

def render_hero():
    """Render the hero section."""
    st.title("🛍️ FashionMatch")
    st.subheader("Découvrez des vêtements qui correspondent à votre style")
    
    st.markdown(
        """
        **FashionMatch** utilise une approche IA hybride combinant:
        
        - 🧠 **CLIP** (Sémantique) - Comprend le style, les couleurs, les textures
        - 🔬 **DINOv2** (Structurel) - Analyse les formes, coupes, motifs
        
        Pour commencer, **ajoutez des images de référence** qui représentent votre style !
        """
    )


def render_getting_started():
    """Render getting started section."""
    st.divider()
    st.header("🚀 Comment ça marche ?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            ### 1️⃣ Définissez votre style
            
            Uploadez des images de vêtements que vous aimez.
            L'IA apprend vos préférences à partir de ces références.
            """
        )
        if st.button("📤 Ajouter des images", use_container_width=True):
            st.switch_page("pages/01_upload.py")
    
    with col2:
        st.markdown(
            """
            ### 2️⃣ Parcourez les articles
            
            Explorez les articles scrapés de Vinted.
            L'IA classe les articles par similarité avec votre style.
            """
        )
        if st.button("🔍 Découvrir", use_container_width=True):
            st.switch_page("pages/02_browse.py")
    
    with col3:
        st.markdown(
            """
            ### 3️⃣ Affinez les résultats
            
            Ajustez les poids CLIP/DINO pour personnaliser
            l'équilibre entre similarité sémantique et structurelle.
            """
        )
        if st.button("⚙️ Paramètres", use_container_width=True):
            st.switch_page("pages/03_settings.py")


def render_tech_stack():
    """Render technical stack section."""
    st.divider()
    
    with st.expander("🔧 Stack technique", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                """
                **Modèles IA:**
                - CLIP ViT-B/32 (512 dimensions)
                - DINOv2 ViT-S/14 (384 dimensions)
                
                **Base de données:**
                - ChromaDB (stockage vectoriel)
                - Late fusion pour recherche hybride
                """
            )
        
        with col2:
            st.markdown(
                """
                **Framework:**
                - Python 3.12+
                - Streamlit (UI)
                - Playwright (scraping)
                
                **Architecture:**
                - Clean Architecture
                - Domain-Driven Design
                """
            )


def render_footer():
    """Render footer."""
    st.divider()
    st.caption(
        "🎓 Projet portfolio - IA Vestimentaire | "
        "[GitHub](https://github.com/AnythingLegalConsidered/FashionMatch)"
    )


# ============================================
# Main
# ============================================

def main():
    """Main application entry point."""
    configure_page()
    render_sidebar()
    
    # Main content
    render_hero()
    render_getting_started()
    render_tech_stack()
    render_footer()


if __name__ == "__main__":
    main()

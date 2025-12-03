# Upload Page - Mon Style
"""
Page for uploading reference images to define user style profile.
Users can upload images of clothing they like, which are encoded
and stored to build their preference profile.
"""
import streamlit as st
from PIL import Image
import io
from typing import Optional

# Configure page (must be first Streamlit command in file for pages)
st.set_page_config(
    page_title="Mon Style - FashionMatch",
    page_icon="👔",
    layout="wide",
)


# ============================================
# Session State & Resource Loading
# ============================================

def get_session_manager():
    """Get or create SessionManager singleton."""
    from src.ui.state.session_manager import SessionManager
    return SessionManager.get_instance()


def get_add_reference_use_case():
    """Get AddReferenceUseCase with cached resources."""
    from src.core.use_cases.add_reference import AddReferenceUseCase
    
    if "add_reference_uc" not in st.session_state:
        manager = get_session_manager()
        encoder = manager.get_encoder()
        repository = manager.get_repository()
        
        if encoder and repository:
            st.session_state.add_reference_uc = AddReferenceUseCase(
                encoder=encoder,
                repository=repository
            )
        else:
            st.session_state.add_reference_uc = None
    
    return st.session_state.add_reference_uc


def get_clear_references_use_case():
    """Get ClearReferencesUseCase."""
    from src.core.use_cases.add_reference import ClearReferencesUseCase
    
    if "clear_references_uc" not in st.session_state:
        manager = get_session_manager()
        repository = manager.get_repository()
        
        if repository:
            st.session_state.clear_references_uc = ClearReferencesUseCase(
                repository=repository
            )
        else:
            st.session_state.clear_references_uc = None
    
    return st.session_state.clear_references_uc


def get_references_use_case():
    """Get GetReferencesUseCase."""
    from src.core.use_cases.add_reference import GetReferencesUseCase
    
    if "get_references_uc" not in st.session_state:
        manager = get_session_manager()
        repository = manager.get_repository()
        
        if repository:
            st.session_state.get_references_uc = GetReferencesUseCase(
                repository=repository
            )
        else:
            st.session_state.get_references_uc = None
    
    return st.session_state.get_references_uc


# ============================================
# Category Options
# ============================================

CATEGORIES = [
    "général",
    "haut",
    "pantalon",
    "robe",
    "jupe",
    "veste",
    "manteau",
    "chaussures",
    "accessoire",
    "sac",
]

STYLE_TAGS = [
    "casual",
    "formel",
    "streetwear",
    "vintage",
    "minimaliste",
    "bohème",
    "sportif",
    "élégant",
    "décontracté",
    "classique",
]


# ============================================
# UI Components
# ============================================

def render_header():
    """Render page header."""
    st.title("👔 Mon Style")
    st.markdown(
        """
        Ajoutez des images de vêtements que vous aimez pour **entraîner l'IA** 
        à comprendre votre style. Plus vous ajoutez d'images, plus les recommandations 
        seront précises !
        """
    )
    st.divider()


def render_upload_section():
    """Render the image upload section."""
    st.subheader("📤 Ajouter des images de référence")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choisissez une ou plusieurs images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Formats supportés: JPG, JPEG, PNG, WEBP",
    )
    
    if not uploaded_files:
        st.info(
            "💡 **Astuce**: Ajoutez des photos de vêtements similaires à ce que vous aimez porter. "
            "Captures d'écran de Vinted, photos personnelles, ou images de mode fonctionnent très bien !"
        )
        return
    
    # Preview uploaded images
    st.markdown("### 👀 Aperçu")
    
    # Create columns for image preview
    cols = st.columns(min(len(uploaded_files), 4))
    
    for i, uploaded_file in enumerate(uploaded_files):
        col_idx = i % 4
        with cols[col_idx]:
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption=uploaded_file.name[:20] + "...", use_container_width=True)
            except Exception as e:
                st.error(f"Erreur: {uploaded_file.name}")
    
    st.divider()
    
    # Category and tags selection
    col1, col2 = st.columns(2)
    
    with col1:
        category = st.selectbox(
            "Catégorie (optionnel)",
            options=CATEGORIES,
            index=0,
            help="Sélectionnez le type de vêtement"
        )
    
    with col2:
        selected_tags = st.multiselect(
            "Tags de style (optionnel)",
            options=STYLE_TAGS,
            default=[],
            help="Sélectionnez les tags qui décrivent votre style"
        )
    
    st.divider()
    
    # Add button
    if st.button("✨ Ajouter au profil", type="primary", use_container_width=True):
        process_uploads(uploaded_files, category, selected_tags)


def process_uploads(uploaded_files, category: str, tags: list[str]):
    """Process and add uploaded images to profile."""
    add_reference_uc = get_add_reference_use_case()
    
    if add_reference_uc is None:
        st.error(
            "⚠️ Les modèles IA ne sont pas disponibles. "
            "Vérifiez que ChromaDB est installé et compatible avec votre version de Python."
        )
        return
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    success_count = 0
    error_count = 0
    
    for i, uploaded_file in enumerate(uploaded_files):
        # Update progress
        progress = (i + 1) / len(uploaded_files)
        progress_bar.progress(progress)
        status_text.text(f"Traitement de {uploaded_file.name}...")
        
        try:
            # Open image
            image = Image.open(uploaded_file).convert("RGB")
            
            # Get name without extension
            name = uploaded_file.name.rsplit(".", 1)[0]
            
            # Add reference
            result = add_reference_uc.execute(
                image=image,
                name=name,
                category=category if category != "général" else None,
                tags=tags if tags else None,
            )
            
            if result.success:
                success_count += 1
            else:
                error_count += 1
                st.warning(f"⚠️ {uploaded_file.name}: {result.message}")
                
        except Exception as e:
            error_count += 1
            st.error(f"❌ Erreur avec {uploaded_file.name}: {str(e)}")
    
    # Clear progress
    progress_bar.empty()
    status_text.empty()
    
    # Show result
    if success_count > 0:
        st.success(f"✅ {success_count} image(s) ajoutée(s) à votre profil !")
        st.balloons()
    
    if error_count > 0:
        st.warning(f"⚠️ {error_count} image(s) n'ont pas pu être traitées.")


def render_current_profile():
    """Render current profile references."""
    st.subheader("📊 Mon profil actuel")
    
    get_refs_uc = get_references_use_case()
    
    if get_refs_uc is None:
        st.info("Les modèles IA ne sont pas chargés.")
        return
    
    references = get_refs_uc.execute()
    
    if not references:
        st.info(
            "🎯 Votre profil est vide. "
            "Ajoutez des images ci-dessus pour commencer à construire votre profil de style !"
        )
        return
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Images de référence", len(references))
    
    with col2:
        categories = set(r.get("category", "général") for r in references)
        st.metric("Catégories", len(categories))
    
    with col3:
        all_tags = []
        for r in references:
            tags_str = r.get("tags", "")
            if tags_str:
                all_tags.extend(tags_str.split(","))
        st.metric("Tags uniques", len(set(all_tags)))
    
    # Reference list
    with st.expander("📋 Voir les références", expanded=False):
        for ref in references:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text(f"📌 {ref.get('name', ref['id'])}")
                category = ref.get("category", "général")
                tags = ref.get("tags", "")
                info = f"  Catégorie: {category}"
                if tags:
                    info += f" | Tags: {tags}"
                st.caption(info)
            with col2:
                added_at = ref.get("added_at", "")
                if added_at:
                    st.caption(f"Ajouté: {added_at[:10]}")
            st.divider()


def render_reset_section():
    """Render profile reset section."""
    st.subheader("🗑️ Réinitialiser le profil")
    
    st.warning(
        "⚠️ Cette action supprimera toutes vos images de référence et réinitialisera votre profil. "
        "Cette action est irréversible."
    )
    
    # Two-step confirmation
    col1, col2 = st.columns([1, 3])
    
    with col1:
        confirm = st.checkbox("Je confirme")
    
    with col2:
        if confirm:
            if st.button("🗑️ Réinitialiser mon profil", type="secondary"):
                clear_refs_uc = get_clear_references_use_case()
                
                if clear_refs_uc is None:
                    st.error("Erreur: Impossible de réinitialiser le profil.")
                    return
                
                if clear_refs_uc.execute():
                    st.success("✅ Profil réinitialisé avec succès !")
                    # Clear cached use cases to refresh
                    for key in ["add_reference_uc", "clear_references_uc", "get_references_uc"]:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
                else:
                    st.error("❌ Erreur lors de la réinitialisation.")


# ============================================
# Main Page
# ============================================

def main():
    """Main page entry point."""
    render_header()
    
    # Two-column layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_upload_section()
    
    with col2:
        render_current_profile()
    
    st.divider()
    render_reset_section()


# Run the page
if __name__ == "__main__":
    main()
else:
    main()


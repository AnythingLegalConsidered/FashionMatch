"""Weight display component for fusion weights visualization."""

import streamlit as st

from src.utils.config import FusionWeights
from src.ui.state_manager import reset_fusion_weights


def render_weight_display(fusion_weights: FusionWeights) -> None:
    """Render fusion weights visualization with gauge and controls.
    
    Args:
        fusion_weights: Current fusion weights
    """
    st.sidebar.markdown("### ⚖️ Fusion Weights")
    
    # Explanation
    with st.sidebar.expander("ℹ️ What are fusion weights?"):
        st.markdown("""
        Fusion weights control how much each AI model contributes to the final similarity score:
        
        - **CLIP** (α): Captures semantic meaning (style, colors, patterns)
        - **DINOv2** (β): Captures structural details (shape, texture, layout)
        
        Weights sum to 1.0 and are automatically adjusted based on your feedback!
        """)
    
    # Visual gauge
    clip_pct = fusion_weights.clip * 100
    dino_pct = fusion_weights.dino * 100
    
    st.sidebar.markdown(
        f'<div style="display: flex; height: 40px; border-radius: 20px; '
        f'overflow: hidden; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">'
        f'<div style="width: {clip_pct}%; background: linear-gradient(90deg, #1E88E5, #42A5F5); '
        f'display: flex; align-items: center; justify-content: center; '
        f'color: white; font-weight: 600; font-size: 0.9rem;">CLIP {clip_pct:.0f}%</div>'
        f'<div style="width: {dino_pct}%; background: linear-gradient(90deg, #43A047, #66BB6A); '
        f'display: flex; align-items: center; justify-content: center; '
        f'color: white; font-weight: 600; font-size: 0.9rem;">DINO {dino_pct:.0f}%</div>'
        f'</div>',
        unsafe_allow_html=True
    )
    
    # Numeric display
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric(
            label="🔵 CLIP (α)",
            value=f"{fusion_weights.clip:.3f}",
            help="Semantic similarity weight"
        )
    
    with col2:
        st.metric(
            label="🟢 DINO (β)",
            value=f"{fusion_weights.dino:.3f}",
            help="Structural similarity weight"
        )
    
    # Reset button
    if st.sidebar.button("🔄 Reset to Default", use_container_width=True):
        reset_fusion_weights()
        st.sidebar.success("✅ Weights reset to default values")
        st.rerun()

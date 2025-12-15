"""Integration tests for UI components."""

import pytest

from src.ui.state_manager import (
    FilterSettings,
    adjust_fusion_weights,
    get_filtered_results,
    initialize_session_state,
)
from src.utils.config import FusionWeights


class MockSessionState:
    """Mock Streamlit session state."""
    
    def __init__(self):
        self.data = {}
    
    def __setattr__(self, name, value):
        if name == "data":
            super().__setattr__(name, value)
        else:
            self.data[name] = value
    
    def __getattr__(self, name):
        if name == "data":
            return super().__getattribute__(name)
        return self.data.get(name)
    
    def __contains__(self, name):
        return name in self.data


@pytest.fixture
def mock_st_session_state(monkeypatch):
    """Mock Streamlit session state."""
    import streamlit as st
    
    mock_state = MockSessionState()
    monkeypatch.setattr("streamlit.session_state", mock_state, raising=False)
    
    return mock_state


class TestStateManager:
    """Test session state management."""
    
    def test_initialize_session_state(self, mock_st_session_state):
        """Test session state initialization."""
        initialize_session_state()
        
        import streamlit as st
        
        assert "references" in st.session_state
        assert "search_results" in st.session_state
        assert "filter_settings" in st.session_state
        assert "feedback_history" in st.session_state
    
    def test_filter_settings_persistence(self, mock_st_session_state):
        """Test filter settings are persisted."""
        import streamlit as st
        
        initialize_session_state()
        
        # Update filter settings
        from src.ui.state_manager import update_filter_settings
        
        settings = FilterSettings(
            min_price=10.0,
            max_price=50.0,
            categories=["chemises"],
            min_similarity=0.7
        )
        
        update_filter_settings(settings)
        
        assert st.session_state.filter_settings.min_price == 10.0
        assert st.session_state.filter_settings.max_price == 50.0


class TestFeedbackMechanism:
    """Test feedback and weight adjustment."""
    
    def test_like_adjusts_weights(self, mock_st_session_state):
        """Test like feedback increases better model weight."""
        import streamlit as st
        
        initialize_session_state()
        
        # Set initial weights
        st.session_state.fusion_weights = FusionWeights(clip=0.6, dino=0.4)
        
        # CLIP scored higher
        new_weights = adjust_fusion_weights(
            feedback_type="like",
            clip_score=0.9,
            dino_score=0.7
        )
        
        # CLIP weight should increase
        assert new_weights.clip > 0.6
        assert new_weights.clip + new_weights.dino == pytest.approx(1.0)
    
    def test_dislike_adjusts_weights(self, mock_st_session_state):
        """Test dislike feedback decreases better model weight."""
        import streamlit as st
        
        initialize_session_state()
        
        st.session_state.fusion_weights = FusionWeights(clip=0.6, dino=0.4)
        
        # DINO scored higher
        new_weights = adjust_fusion_weights(
            feedback_type="dislike",
            clip_score=0.6,
            dino_score=0.8
        )
        
        # DINO weight should decrease
        assert new_weights.dino < 0.4
        assert new_weights.clip + new_weights.dino == pytest.approx(1.0)


class TestFilterLogic:
    """Test result filtering logic."""
    
    def test_filter_by_price(self, mock_st_session_state, mock_fashion_items):
        """Test price range filtering."""
        import streamlit as st
        
        initialize_session_state()
        
        # Mock search results
        from src.database.models import SearchResult
        
        results = [
            SearchResult(
                item_id=item.item_id,
                similarity_score=0.8,
                clip_score=0.75,
                dino_score=0.85,
                item=item
            )
            for item in mock_fashion_items
        ]
        
        st.session_state.search_results = results
        st.session_state.filter_settings = FilterSettings(
            min_price=20.0,
            max_price=40.0
        )
        
        filtered = get_filtered_results()
        
        assert all(20.0 <= r.item.price <= 40.0 for r in filtered)
    
    def test_filter_by_category(self, mock_st_session_state, mock_fashion_items):
        """Test category filtering."""
        import streamlit as st
        
        initialize_session_state()
        
        from src.database.models import SearchResult
        
        results = [
            SearchResult(
                item_id=item.item_id,
                similarity_score=0.8,
                clip_score=0.75,
                dino_score=0.85,
                item=item
            )
            for item in mock_fashion_items
        ]
        
        st.session_state.search_results = results
        st.session_state.filter_settings = FilterSettings(
            categories=["chemises"]
        )
        
        filtered = get_filtered_results()
        
        assert all(r.item.category == "chemises" for r in filtered)
    
    def test_filter_by_similarity(self, mock_st_session_state, mock_fashion_items):
        """Test similarity threshold filtering."""
        import streamlit as st
        
        initialize_session_state()
        
        from src.database.models import SearchResult
        
        results = [
            SearchResult(
                item_id=item.item_id,
                similarity_score=0.5 + i * 0.1,
                clip_score=0.5,
                dino_score=0.5,
                item=item
            )
            for i, item in enumerate(mock_fashion_items)
        ]
        
        st.session_state.search_results = results
        st.session_state.filter_settings = FilterSettings(
            min_similarity=0.7
        )
        
        filtered = get_filtered_results()
        
        assert all(r.similarity_score >= 0.7 for r in filtered)

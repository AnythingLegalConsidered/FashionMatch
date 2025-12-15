"""Custom CSS styling for FashionMatch UI."""


def get_custom_css() -> str:
    """Generate custom CSS for the FashionMatch UI.
    
    Returns:
        CSS string to inject via st.markdown
    """
    return """
    <style>
    /* Main app styling */
    .main {
        padding: 1rem;
    }
    
    /* Item card styling */
    .item-card {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        background: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .item-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    
    /* Image styling */
    .item-image {
        border-radius: 8px;
        width: 100%;
        object-fit: cover;
    }
    
    /* Score display */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.2rem;
    }
    
    .score-high {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    
    .score-medium {
        background-color: #FFF9C4;
        color: #F57F17;
    }
    
    .score-low {
        background-color: #FFEBEE;
        color: #C62828;
    }
    
    /* Feedback buttons */
    .feedback-button {
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-size: 1.2rem;
        cursor: pointer;
        transition: all 0.2s;
        margin: 0.25rem;
    }
    
    .like-button {
        background-color: #E8F5E9;
        color: #2E7D32;
    }
    
    .like-button:hover {
        background-color: #C8E6C9;
        transform: scale(1.1);
    }
    
    .dislike-button {
        background-color: #FFEBEE;
        color: #C62828;
    }
    
    .dislike-button:hover {
        background-color: #FFCDD2;
        transform: scale(1.1);
    }
    
    /* Weight display */
    .weight-gauge {
        display: flex;
        height: 30px;
        border-radius: 15px;
        overflow: hidden;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .weight-clip {
        background: linear-gradient(90deg, #1E88E5, #42A5F5);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .weight-dino {
        background: linear-gradient(90deg, #43A047, #66BB6A);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    /* Stats panel */
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        margin-bottom: 1.5rem;
    }
    
    .stat-item {
        text-align: center;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }
    
    .stat-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Filter section */
    .filter-section {
        background-color: #F5F5F5;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Metadata display */
    .metadata-row {
        display: flex;
        align-items: center;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .metadata-icon {
        margin-right: 0.5rem;
        color: #666;
    }
    
    /* Empty state */
    .empty-state {
        text-align: center;
        padding: 3rem;
        color: #666;
    }
    
    .empty-state-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    
    /* Responsive grid */
    @media (max-width: 768px) {
        .item-card {
            margin-bottom: 1.5rem;
        }
    }
    
    /* Link styling */
    a {
        color: #1E88E5;
        text-decoration: none;
        font-weight: 500;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Reference image preview */
    .reference-preview {
        border-radius: 8px;
        border: 2px solid #E0E0E0;
        padding: 0.5rem;
        margin: 0.5rem;
        display: inline-block;
    }
    
    /* Success message styling */
    .success-toast {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 4px solid #43A047;
    }
    
    /* Loading spinner custom color */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
    </style>
    """

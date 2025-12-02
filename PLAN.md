# 📋 FashionMatch - Technical Roadmap

> Detailed development plan organized in 5 phases with precise technical tasks.

---

## 🗂️ Project Structure (Clean Architecture)

```
fashionmatch/
│
├── 📁 config/                          # Configuration Layer
│   ├── config.yaml                     # Main app configuration
│   ├── config.example.yaml             # Template for new users
│   ├── logging.yaml                    # Logging configuration
│   └── __init__.py
│
├── 📁 data/                            # Data Layer (gitignored)
│   ├── references/                     # User's style reference images
│   │   └── .gitkeep
│   ├── scraped/                        # Raw scraped images & metadata
│   │   ├── images/
│   │   └── metadata/
│   └── chroma/                         # ChromaDB persistence
│
├── 📁 src/                             # Source Code (Clean Architecture)
│   │
│   ├── 📁 domain/                      # Domain Layer (Entities & Interfaces)
│   │   ├── __init__.py
│   │   ├── entities/
│   │   │   ├── __init__.py
│   │   │   ├── clothing_item.py        # ClothingItem dataclass
│   │   │   ├── embedding.py            # Embedding value object
│   │   │   └── user_preference.py      # UserPreference entity
│   │   └── interfaces/
│   │       ├── __init__.py
│   │       ├── encoder_interface.py    # Abstract base for encoders
│   │       ├── scraper_interface.py    # Abstract base for scrapers
│   │       └── repository_interface.py # Abstract base for storage
│   │
│   ├── 📁 core/                        # Application Core (Use Cases)
│   │   ├── __init__.py
│   │   ├── encoders/
│   │   │   ├── __init__.py
│   │   │   ├── base_encoder.py         # Abstract encoder class
│   │   │   ├── clip_encoder.py         # CLIP implementation
│   │   │   ├── dino_encoder.py         # DINOv2 implementation
│   │   │   └── hybrid_encoder.py       # Dual encoder orchestrator
│   │   ├── scoring/
│   │   │   ├── __init__.py
│   │   │   ├── similarity.py           # Cosine similarity functions
│   │   │   ├── weighted_scorer.py      # Late fusion scorer
│   │   │   └── feedback_optimizer.py   # Dynamic weight adjustment
│   │   ├── pipelines/
│   │   │   ├── __init__.py
│   │   │   ├── embedding_pipeline.py   # Full embedding workflow
│   │   │   └── recommendation_pipeline.py # End-to-end recommendations
│   │   └── use_cases/
│   │       ├── __init__.py
│   │       ├── add_reference.py        # Add reference image use case
│   │       ├── get_recommendations.py  # Get recommendations use case
│   │       └── process_feedback.py     # Process user feedback
│   │
│   ├── 📁 infrastructure/              # Infrastructure Layer
│   │   ├── __init__.py
│   │   ├── scraper/
│   │   │   ├── __init__.py
│   │   │   ├── vinted_scraper.py       # Playwright scraper
│   │   │   ├── parsers/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── listing_parser.py   # Parse listing pages
│   │   │   │   └── item_parser.py      # Parse item details
│   │   │   └── rate_limiter.py         # Request throttling
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── chroma_repository.py    # ChromaDB implementation
│   │   │   ├── collections.py          # Collection management
│   │   │   └── migrations.py           # Schema migrations
│   │   └── external/
│   │       ├── __init__.py
│   │       └── model_downloader.py     # Download pretrained models
│   │
│   ├── 📁 ui/                          # Presentation Layer
│   │   ├── __init__.py
│   │   ├── app.py                      # Streamlit main entry
│   │   ├── pages/
│   │   │   ├── __init__.py
│   │   │   ├── 01_upload.py            # Reference upload page
│   │   │   ├── 02_browse.py            # Browse recommendations
│   │   │   └── 03_settings.py          # Configuration page
│   │   ├── components/
│   │   │   ├── __init__.py
│   │   │   ├── image_card.py           # Clothing item card
│   │   │   ├── feedback_buttons.py     # Like/Dislike buttons
│   │   │   ├── similarity_meter.py     # Score visualization
│   │   │   └── filters.py              # Category/price filters
│   │   └── state/
│   │       ├── __init__.py
│   │       └── session_manager.py      # Streamlit session state
│   │
│   └── 📁 utils/                       # Shared Utilities
│       ├── __init__.py
│       ├── config.py                   # Pydantic config loader
│       ├── logger.py                   # Structured logging
│       ├── image_utils.py              # Image preprocessing
│       ├── validators.py               # Input validation
│       └── exceptions.py               # Custom exceptions
│
├── 📁 tests/                           # Test Suite
│   ├── __init__.py
│   ├── conftest.py                     # Pytest fixtures
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_encoders.py
│   │   ├── test_scorer.py
│   │   └── test_parsers.py
│   ├── integration/
│   │   ├── __init__.py
│   │   ├── test_pipeline.py
│   │   └── test_database.py
│   └── fixtures/
│       └── sample_images/
│
├── 📁 notebooks/                       # Jupyter Experimentation
│   ├── 01_clip_exploration.ipynb
│   ├── 02_dino_exploration.ipynb
│   ├── 03_fusion_experiments.ipynb
│   └── 04_scraper_dev.ipynb
│
├── 📁 scripts/                         # CLI Scripts
│   ├── scrape.py                       # Run scraper
│   ├── embed.py                        # Generate embeddings
│   └── evaluate.py                     # Evaluate model performance
│
├── 📁 docs/                            # Documentation
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
│
├── .env.example                        # Environment variables template
├── .gitignore
├── .pre-commit-config.yaml             # Code quality hooks
├── pyproject.toml                      # Project metadata & tools config
├── requirements.txt                    # Production dependencies
├── requirements-dev.txt                # Development dependencies
├── Makefile                            # Common commands
├── README.md
└── PLAN.md
```

---

## 🚀 Phase 1: Project Setup & Foundation

**Duration**: 2-3 days  
**Goal**: Establish a solid, professional project foundation.

### Tasks

| ID | Task | Description | Files |
|----|------|-------------|-------|
| 1.1 | **Initialize Repository** | Create Git repo with proper `.gitignore` | `.gitignore` |
| 1.2 | **Setup Virtual Environment** | Create venv with Python 3.10+ | `requirements.txt` |
| 1.3 | **Configure Dependencies** | Define all project dependencies | `requirements.txt`, `requirements-dev.txt` |
| 1.4 | **Setup Pydantic Config** | Type-safe configuration management | `src/utils/config.py`, `config/config.yaml` |
| 1.5 | **Implement Logger** | Structured logging with rotation | `src/utils/logger.py`, `config/logging.yaml` |
| 1.6 | **Create Base Exceptions** | Custom exception hierarchy | `src/utils/exceptions.py` |
| 1.7 | **Setup Pre-commit Hooks** | Black, isort, flake8, mypy | `.pre-commit-config.yaml` |
| 1.8 | **Create Makefile** | Common dev commands | `Makefile` |
| 1.9 | **Write Domain Entities** | Core dataclasses | `src/domain/entities/` |
| 1.10 | **Define Interfaces** | Abstract base classes | `src/domain/interfaces/` |

### Deliverables

- [ ] Running virtual environment
- [ ] Configuration loading from YAML
- [ ] Structured logging active
- [ ] Pre-commit hooks passing
- [ ] Domain entities defined

### Configuration Schema (config.yaml)

```yaml
# Application Settings
app:
  name: "FashionMatch"
  version: "0.1.0"
  debug: false

# Model Configuration
models:
  device: "cuda"  # or "cpu"
  clip:
    model_name: "ViT-B/32"
    embedding_dim: 512
  dino:
    model_name: "dinov2_vits14"
    embedding_dim: 384
  fusion:
    strategy: "weighted_average"
    weights:
      clip: 0.5
      dino: 0.5

# Scraper Configuration
scraper:
  base_url: "https://www.vinted.fr"
  user_agent: "Mozilla/5.0 ..."
  max_pages_per_category: 10
  request_delay:
    min_seconds: 1.0
    max_seconds: 3.0
  timeout_seconds: 30
  headless: true

# Database Configuration
database:
  provider: "chroma"
  chroma:
    persist_directory: "./data/chroma"
    collection_name: "clothing_items"

# Image Processing
images:
  max_size: [224, 224]
  normalize: true
  cache_dir: "./data/scraped/images"

# UI Configuration
ui:
  items_per_page: 20
  show_scores: true
  theme: "light"
```

---

## 🧠 Phase 2: AI Core (Encoders & Scoring)

**Duration**: 4-5 days  
**Goal**: Implement the hybrid CLIP + DINOv2 embedding system.

### Tasks

| ID | Task | Description | Files |
|----|------|-------------|-------|
| 2.1 | **Image Preprocessor** | Resize, normalize, convert to tensor | `src/utils/image_utils.py` |
| 2.2 | **Abstract Encoder Base** | Define encoder interface | `src/core/encoders/base_encoder.py` |
| 2.3 | **CLIP Encoder** | Load CLIP, implement encode() | `src/core/encoders/clip_encoder.py` |
| 2.4 | **DINOv2 Encoder** | Load DINOv2, implement encode() | `src/core/encoders/dino_encoder.py` |
| 2.5 | **Hybrid Encoder** | Orchestrate both encoders | `src/core/encoders/hybrid_encoder.py` |
| 2.6 | **Cosine Similarity** | Similarity computation functions | `src/core/scoring/similarity.py` |
| 2.7 | **WeightedScorer Class** | Late fusion implementation | `src/core/scoring/weighted_scorer.py` |
| 2.8 | **Embedding Pipeline** | End-to-end image → embeddings | `src/core/pipelines/embedding_pipeline.py` |
| 2.9 | **Model Downloader** | Auto-download pretrained weights | `src/infrastructure/external/model_downloader.py` |
| 2.10 | **Unit Tests (Encoders)** | Test encoder outputs | `tests/unit/test_encoders.py` |

### Key Classes

```python
# src/core/encoders/base_encoder.py
from abc import ABC, abstractmethod
from typing import Union
import numpy as np
from PIL import Image

class BaseEncoder(ABC):
    """Abstract base class for image encoders."""
    
    @abstractmethod
    def encode(self, image: Union[str, Image.Image]) -> np.ndarray:
        """Generate embedding vector from image."""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the embedding dimension."""
        pass
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the pretrained model."""
        pass
```

```python
# src/core/scoring/weighted_scorer.py
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass
class FusionWeights:
    clip: float = 0.5
    dino: float = 0.5
    
    def __post_init__(self):
        assert abs(self.clip + self.dino - 1.0) < 1e-6, "Weights must sum to 1"

class WeightedScorer:
    """Late fusion scorer combining CLIP and DINO similarities."""
    
    def __init__(self, weights: FusionWeights):
        self.weights = weights
    
    def compute_score(
        self, 
        clip_sim: float, 
        dino_sim: float
    ) -> float:
        """Compute weighted fusion score."""
        return (
            self.weights.clip * clip_sim + 
            self.weights.dino * dino_sim
        )
    
    def update_weights(self, new_weights: FusionWeights) -> None:
        """Update fusion weights based on feedback."""
        self.weights = new_weights
```

### Deliverables

- [ ] CLIP encoder loading and encoding images
- [ ] DINOv2 encoder loading and encoding images
- [ ] Hybrid encoder returning both embeddings
- [ ] WeightedScorer computing fusion scores
- [ ] All encoder tests passing

---

## 🌐 Phase 3: Data Pipeline (Scraping)

**Duration**: 3-4 days  
**Goal**: Build a robust, respectful Vinted scraper.

### Tasks

| ID | Task | Description | Files |
|----|------|-------------|-------|
| 3.1 | **Abstract Scraper Interface** | Define scraper contract | `src/domain/interfaces/scraper_interface.py` |
| 3.2 | **Rate Limiter** | Respectful request throttling | `src/infrastructure/scraper/rate_limiter.py` |
| 3.3 | **Listing Parser** | Extract items from search pages | `src/infrastructure/scraper/parsers/listing_parser.py` |
| 3.4 | **Item Parser** | Extract item details | `src/infrastructure/scraper/parsers/item_parser.py` |
| 3.5 | **Vinted Scraper** | Main Playwright scraper class | `src/infrastructure/scraper/vinted_scraper.py` |
| 3.6 | **Image Downloader** | Download & cache images | `src/infrastructure/scraper/vinted_scraper.py` |
| 3.7 | **Scraper CLI Script** | Command-line interface | `scripts/scrape.py` |
| 3.8 | **Unit Tests (Parsers)** | Test HTML parsing | `tests/unit/test_parsers.py` |
| 3.9 | **Scraper Notebook** | Interactive development | `notebooks/04_scraper_dev.ipynb` |

### Key Classes

```python
# src/infrastructure/scraper/vinted_scraper.py
from playwright.async_api import async_playwright
from typing import List, AsyncGenerator
from src.domain.entities.clothing_item import ClothingItem

class VintedScraper:
    """Playwright-based Vinted scraper."""
    
    def __init__(self, config: ScraperConfig):
        self.config = config
        self.rate_limiter = RateLimiter(
            min_delay=config.request_delay.min_seconds,
            max_delay=config.request_delay.max_seconds
        )
    
    async def scrape_category(
        self, 
        category: str, 
        max_pages: int = 10
    ) -> AsyncGenerator[ClothingItem, None]:
        """Scrape items from a category."""
        pass
    
    async def scrape_item_details(
        self, 
        item_url: str
    ) -> ClothingItem:
        """Scrape detailed info from item page."""
        pass
    
    async def download_image(
        self, 
        image_url: str, 
        save_path: str
    ) -> str:
        """Download and cache item image."""
        pass
```

```python
# src/domain/entities/clothing_item.py
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

@dataclass
class ClothingItem:
    """Represents a clothing item from Vinted."""
    
    id: str
    title: str
    price: float
    currency: str = "EUR"
    brand: Optional[str] = None
    size: Optional[str] = None
    condition: Optional[str] = None
    category: Optional[str] = None
    image_url: str = ""
    local_image_path: Optional[str] = None
    item_url: str = ""
    description: Optional[str] = None
    seller_id: Optional[str] = None
    scraped_at: datetime = field(default_factory=datetime.utcnow)
    
    # Embeddings (populated later)
    clip_embedding: Optional[List[float]] = None
    dino_embedding: Optional[List[float]] = None
```

### Deliverables

- [ ] Functional Vinted scraper with Playwright
- [ ] Rate limiting respecting Vinted
- [ ] Image downloading and caching
- [ ] CLI script for batch scraping
- [ ] Parser tests passing

---

## 💾 Phase 4: Database Integration (ChromaDB)

**Duration**: 2-3 days  
**Goal**: Persist embeddings and enable fast vector search.

### Tasks

| ID | Task | Description | Files |
|----|------|-------------|-------|
| 4.1 | **Abstract Repository Interface** | Define storage contract | `src/domain/interfaces/repository_interface.py` |
| 4.2 | **ChromaDB Repository** | Implement vector storage | `src/infrastructure/database/chroma_repository.py` |
| 4.3 | **Collection Manager** | Handle collections lifecycle | `src/infrastructure/database/collections.py` |
| 4.4 | **Embedding Storage** | Store CLIP + DINO embeddings | `src/infrastructure/database/chroma_repository.py` |
| 4.5 | **Similarity Search** | Query similar items | `src/infrastructure/database/chroma_repository.py` |
| 4.6 | **Hybrid Search** | Combined CLIP + DINO search | `src/core/use_cases/get_recommendations.py` |
| 4.7 | **Reference Storage** | Store user's reference images | `src/core/use_cases/add_reference.py` |
| 4.8 | **Integration Tests** | Test full pipeline | `tests/integration/test_database.py` |

### Key Classes

```python
# src/infrastructure/database/chroma_repository.py
import chromadb
from typing import List, Dict, Any, Optional
from src.domain.entities.clothing_item import ClothingItem

class ChromaRepository:
    """ChromaDB-based vector storage."""
    
    def __init__(self, config: DatabaseConfig):
        self.client = chromadb.PersistentClient(
            path=config.chroma.persist_directory
        )
        self.collection = self._get_or_create_collection(
            config.chroma.collection_name
        )
    
    def add_item(
        self, 
        item: ClothingItem,
        clip_embedding: List[float],
        dino_embedding: List[float]
    ) -> str:
        """Add item with embeddings to the database."""
        pass
    
    def search_similar(
        self,
        query_embedding: List[float],
        embedding_type: str = "clip",  # or "dino"
        n_results: int = 20,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[ClothingItem]:
        """Find similar items by embedding."""
        pass
    
    def hybrid_search(
        self,
        clip_embedding: List[float],
        dino_embedding: List[float],
        weights: FusionWeights,
        n_results: int = 20
    ) -> List[Tuple[ClothingItem, float]]:
        """Perform hybrid search with late fusion."""
        pass
```

### Database Schema

```
ChromaDB Collections:
│
├── clothing_items_clip      # CLIP embeddings
│   ├── embeddings: float[512]
│   ├── metadata: {id, title, price, brand, ...}
│   └── documents: [image_path]
│
├── clothing_items_dino      # DINO embeddings
│   ├── embeddings: float[384]
│   ├── metadata: {id, title, price, brand, ...}
│   └── documents: [image_path]
│
└── user_references          # User's reference images
    ├── embeddings: float[512+384]  # Concatenated
    ├── metadata: {source, added_at, feedback_count}
    └── documents: [reference_path]
```

### Deliverables

- [ ] ChromaDB persistence working
- [ ] Items stored with dual embeddings
- [ ] Similarity search functional
- [ ] Hybrid search implemented
- [ ] Integration tests passing

---

## 🖥️ Phase 5: UI & Feedback Loop (Streamlit)

**Duration**: 3-4 days  
**Goal**: Create an interactive interface with learning capabilities.

### Tasks

| ID | Task | Description | Files |
|----|------|-------------|-------|
| 5.1 | **Streamlit App Structure** | Multi-page app setup | `src/ui/app.py` |
| 5.2 | **Upload Page** | Reference image upload | `src/ui/pages/01_upload.py` |
| 5.3 | **Browse Page** | Recommendation gallery | `src/ui/pages/02_browse.py` |
| 5.4 | **Settings Page** | Configuration UI | `src/ui/pages/03_settings.py` |
| 5.5 | **Image Card Component** | Clothing item display | `src/ui/components/image_card.py` |
| 5.6 | **Feedback Buttons** | Like/Dislike with callbacks | `src/ui/components/feedback_buttons.py` |
| 5.7 | **Similarity Meter** | Score visualization | `src/ui/components/similarity_meter.py` |
| 5.8 | **Session Manager** | State persistence | `src/ui/state/session_manager.py` |
| 5.9 | **Feedback Optimizer** | Weight adjustment from feedback | `src/core/scoring/feedback_optimizer.py` |
| 5.10 | **Process Feedback Use Case** | Handle user preferences | `src/core/use_cases/process_feedback.py` |

### Key Components

```python
# src/ui/components/image_card.py
import streamlit as st
from src.domain.entities.clothing_item import ClothingItem

def render_image_card(
    item: ClothingItem,
    score: float,
    on_like: callable,
    on_dislike: callable,
    show_score: bool = True
) -> None:
    """Render a clothing item card with feedback buttons."""
    
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(item.local_image_path, use_column_width=True)
        
        with col2:
            st.markdown(f"**{item.title}**")
            st.write(f"💰 {item.price} {item.currency}")
            if item.brand:
                st.write(f"🏷️ {item.brand}")
            if item.size:
                st.write(f"📏 {item.size}")
            
            if show_score:
                st.progress(score, text=f"Match: {score:.0%}")
            
            col_like, col_dislike = st.columns(2)
            with col_like:
                if st.button("👍", key=f"like_{item.id}"):
                    on_like(item.id)
            with col_dislike:
                if st.button("👎", key=f"dislike_{item.id}"):
                    on_dislike(item.id)
```

```python
# src/core/scoring/feedback_optimizer.py
from typing import List, Tuple
from src.core.scoring.weighted_scorer import FusionWeights

class FeedbackOptimizer:
    """Adjust fusion weights based on user feedback."""
    
    def __init__(self, learning_rate: float = 0.1):
        self.learning_rate = learning_rate
        self.feedback_history: List[Tuple[str, bool]] = []  # (item_id, liked)
    
    def record_feedback(
        self, 
        item_id: str, 
        liked: bool,
        clip_score: float,
        dino_score: float
    ) -> None:
        """Record user feedback for an item."""
        self.feedback_history.append((item_id, liked, clip_score, dino_score))
    
    def compute_optimal_weights(
        self, 
        current_weights: FusionWeights
    ) -> FusionWeights:
        """Compute improved weights based on feedback."""
        # Simple heuristic: boost weight of model that better predicted likes
        # More sophisticated: gradient descent on feedback loss
        pass
```

### UI Wireframe

```
┌─────────────────────────────────────────────────────────────┐
│  🎨 FashionMatch                     [Upload] [Browse] [⚙️]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📸 Your Style References (3 images)                        │
│  ┌─────┐ ┌─────┐ ┌─────┐                                    │
│  │     │ │     │ │     │  [+ Add More]                      │
│  └─────┘ └─────┘ └─────┘                                    │
│                                                             │
│  ──────────────────────────────────────────────────────────│
│                                                             │
│  🔍 Recommended for You                    Showing 1-20     │
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │             │  │             │  │             │         │
│  │   [Image]   │  │   [Image]   │  │   [Image]   │         │
│  │             │  │             │  │             │         │
│  ├─────────────┤  ├─────────────┤  ├─────────────┤         │
│  │ Title...    │  │ Title...    │  │ Title...    │         │
│  │ 💰 25€      │  │ 💰 18€      │  │ 💰 32€      │         │
│  │ ████████ 87%│  │ ██████░░ 76%│  │ █████░░░ 71%│         │
│  │ [👍] [👎]   │  │ [👍] [👎]   │  │ [👍] [👎]   │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                             │
│                    [Load More...]                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Deliverables

- [ ] Multi-page Streamlit app
- [ ] Image upload functional
- [ ] Recommendation gallery
- [ ] Like/Dislike buttons working
- [ ] Feedback affecting future recommendations

---

## 📅 Timeline Summary

```
Week 1: ████████████████████████████████████████
        Phase 1 (Setup)      Phase 2 (AI Core)
        Days 1-3             Days 4-8

Week 2: ████████████████████████████████████████
        Phase 2 (cont.)  Phase 3 (Scraping)  Phase 4 (DB)
        Days 8-9         Days 10-13          Days 14-16

Week 3: ████████████████████████████████████████
        Phase 4 (cont.)  Phase 5 (UI)    Testing & Polish
        Day 17           Days 18-21      Days 22-25
```

---

## ✅ Definition of Done

For each phase to be considered complete:

1. **Code Quality**
   - [ ] All functions have type hints
   - [ ] Docstrings on all public methods
   - [ ] No hardcoded values (all in config)
   - [ ] Pre-commit hooks passing

2. **Testing**
   - [ ] Unit tests for new code
   - [ ] Integration tests where applicable
   - [ ] Coverage > 80%

3. **Documentation**
   - [ ] README updated if needed
   - [ ] Docstrings complete
   - [ ] Architecture diagrams current

4. **Functionality**
   - [ ] Feature works as specified
   - [ ] Error handling in place
   - [ ] Logging implemented

---

## 🎯 Success Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Code Coverage | > 80% | pytest-cov |
| Type Coverage | > 90% | mypy |
| Embedding Speed | < 100ms/image | Benchmark script |
| Search Latency | < 50ms | Benchmark script |
| User Satisfaction | Qualitative | Manual testing |
| Precision@10 | > 70% | Evaluation script |

---

## 📚 Resources

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Playwright Python](https://playwright.dev/python/)
- [Streamlit Docs](https://docs.streamlit.io/)

---

> **Note for Recruiters**: This project demonstrates proficiency in:
> - Clean Architecture principles
> - Modern AI/ML pipelines
> - Async programming (Playwright)
> - Vector databases
> - Interactive UI development
> - Professional Python practices (typing, testing, config management)

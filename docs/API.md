# FashionMatch API Reference

Comprehensive API documentation for programmatic usage.

## Core Modules

### Configuration (`src.utils.config`)

#### `get_config() -> AppConfig`

Get or create cached configuration instance.

```python
from src.utils import get_config

config = get_config()
print(config.models.clip_model)
```

**Returns**: `AppConfig` singleton instance

---

#### `AppConfig`

Root configuration model.

**Fields**:
- `models`: `ModelConfig` - AI model settings
- `scraper`: `ScraperConfig` - Web scraper settings
- `database`: `DatabaseConfig` - ChromaDB settings
- `data_dir`: `str` - Root data directory
- `references_dir`: `str` - Reference images path
- `scraped_dir`: `str` - Scraped data path
- `log_level`: `str` - Logging level

**Methods**:
- `from_yaml(path: str) -> AppConfig`: Load from YAML file

---

#### `FusionWeights`

Fusion weight configuration.

**Fields**:
- `clip`: `float` [0.0, 1.0] - CLIP weight (α)
- `dino`: `float` [0.0, 1.0] - DINOv2 weight (β)

**Validation**: `clip + dino == 1.0`

```python
from src.utils.config import FusionWeights

weights = FusionWeights(clip=0.7, dino=0.3)
```

---

### Encoders (`src.core.encoders`)

#### `get_clip_encoder(model_name, device) -> CLIPEncoder`

Get or create cached CLIP encoder.

**Parameters**:
- `model_name`: `str` - CLIP variant (e.g., "openai/clip-vit-base-patch32")
- `device`: `str` - "cuda" or "cpu"

**Returns**: `CLIPEncoder` instance

```python
from src.core.encoders import get_clip_encoder

encoder = get_clip_encoder("openai/clip-vit-base-patch32", "cuda")
embeddings = encoder.encode(images)  # List of PIL Images
```

---

#### `CLIPEncoder.encode(images) -> np.ndarray`

Encode images to CLIP embeddings.

**Parameters**:
- `images`: `list[PIL.Image]` - Input images

**Returns**: `np.ndarray` of shape `(N, embedding_dim)`, dtype `float32`

**Properties**:
- `embedding_dim`: `int` - Embedding dimension (512/768)

---

#### `get_dino_encoder(model_name, device) -> DINOEncoder`

Get or create cached DINOv2 encoder.

**Parameters**:
- `model_name`: `str` - DINOv2 variant (e.g., "dinov2_vits14")
- `device`: `str` - "cuda" or "cpu"

**Returns**: `DINOEncoder` instance

```python
from src.core.encoders import get_dino_encoder

encoder = get_dino_encoder("dinov2_vits14", "cuda")
embeddings = encoder.encode(images)
```

---

### Hybrid Scorer (`src.core.scorer`)

#### `get_hybrid_scorer(config) -> HybridScorer`

Get or create cached hybrid scorer.

**Parameters**:
- `config`: `ModelConfig` - Model configuration

**Returns**: `HybridScorer` instance

```python
from src.core import get_hybrid_scorer
from src.utils import get_config

config = get_config()
scorer = get_hybrid_scorer(config.models)
```

---

#### `HybridScorer.encode_dual(images) -> tuple[np.ndarray, np.ndarray]`

Encode images with both CLIP and DINOv2.

**Parameters**:
- `images`: `list[PIL.Image]` - Input images

**Returns**: Tuple of `(clip_embeddings, dino_embeddings)`

```python
clip_embs, dino_embs = scorer.encode_dual(images)
```

---

#### `HybridScorer.compute_similarity(...) -> float`

Compute fused similarity score.

**Parameters**:
- `query_clip`: `np.ndarray` - Query CLIP embedding
- `query_dino`: `np.ndarray` - Query DINOv2 embedding
- `candidate_clip`: `np.ndarray` - Candidate CLIP embedding
- `candidate_dino`: `np.ndarray` - Candidate DINOv2 embedding

**Returns**: `float` - Fused similarity score [0.0, 1.0]

---

#### `HybridScorer.update_weights(fusion_weights)`

Update fusion weights dynamically.

**Parameters**:
- `fusion_weights`: `FusionWeights` - New weights

```python
from src.utils.config import FusionWeights

new_weights = FusionWeights(clip=0.8, dino=0.2)
scorer.update_weights(new_weights)
```

---

### Vector Store (`src.database.vector_store`)

#### `get_vector_store(config, clip_dim, dino_dim) -> VectorStore`

Get or create cached vector store.

**Parameters**:
- `config`: `DatabaseConfig` - Database configuration
- `clip_dim`: `int` - CLIP embedding dimension (e.g., 512)
- `dino_dim`: `int` - DINOv2 embedding dimension (e.g., 384)

**Returns**: `VectorStore` instance

```python
from src.core import get_hybrid_scorer
from src.database import get_vector_store
from src.utils import get_config

config = get_config()
scorer = get_hybrid_scorer(config.models)

# Get dimensions from encoders
clip_dim = scorer.clip_encoder.embedding_dim
dino_dim = scorer.dino_encoder.embedding_dim

store = get_vector_store(config.database, clip_dim, dino_dim)
```

---

#### `VectorStore.add_items(items, batch_size) -> BatchInsertResult`

Add fashion items to vector store.

**Parameters**:
- `items`: `list[FashionItem]` - Items with embeddings (must have both CLIP and DINOv2 embeddings)
- `batch_size`: `int` (optional) - Batch size for insertion

**Returns**: `BatchInsertResult` with statistics

**Raises**:
- `VectorStoreError`: If any item is missing embeddings (both CLIP and DINOv2 required)
- `EmbeddingDimensionError`: If embedding dimensions don't match expected dimensions
- `BatchInsertError`: If items fail to insert into database

**Behavior**:
- All items must have both `clip_embedding` and `dino_embedding` set (not None)
- Embeddings are validated against expected dimensions before insertion
- If validation fails, raises error immediately without partial insertion

```python
from src.database.models import FashionItem

items = [FashionItem(...), ...]
try:
    result = store.add_items(items, batch_size=32)
    print(f"Success: {result.successful}, Failed: {result.failed}")
except VectorStoreError as e:
    print(f"Invalid items: {e}")
```

---

#### `VectorStore.search(...) -> list[SearchResult]`

Search for similar items.

**Parameters**:
- `clip_query`: `np.ndarray` - CLIP query embedding
- `dino_query`: `np.ndarray` - DINOv2 query embedding
- `top_k`: `int` - Number of results
- `fusion_weights`: `FusionWeights` (optional) - Custom fusion weights

**Returns**: `list[SearchResult]` ordered by similarity (descending)

```python
results = store.search(
    clip_query=clip_emb,
    dino_query=dino_emb,
#### `VectorStore.get_by_ids(ids) -> list[FashionItem]`

Retrieve items by IDs.

**Parameters**:
- `ids`: `list[str]` - Item IDs to retrieve

**Returns**: `list[FashionItem]` - Only items that exist (filters out nonexistent IDs)

**Behavior**:
- Returns only items that exist in the vector store
- Nonexistent IDs are filtered out with warning logs (no error raised)
- Empty list returned if no IDs match
- Missing items are logged but do not interrupt retrieval

```python
# Retrieve specific items (mixed existing and nonexistent)
items = store.get_by_ids(["item_001", "item_002", "nonexistent"])
# Returns only item_001 and item_002 if they exist
# Logs: "Retrieved 2/3 items, skipped 1 missing"

# All nonexistent IDs
items = store.get_by_ids(["nonexistent_1", "nonexistent_2"])
# Returns empty list []
```

---# `VectorStore.get_by_ids(ids) -> list[FashionItem]`

Retrieve items by IDs.

**Parameters**:
- `ids`: `list[str]` - Item IDs

**Returns**: `list[FashionItem]`

---

#### `VectorStore.delete_items(ids) -> int`

Delete items from vector store.

**Parameters**:
- `ids`: `list[str]` - Item IDs to delete

**Returns**: `int` - Number of items deleted

---

### Web Scraper (`src.scraper.vinted_scraper`)

#### `VintedScraper.scrape_category(category, max_pages) -> ScrapedBatch`

Scrape Vinted category.

**Parameters**:
- `category`: `str` - Category slug (e.g., "chemises")
- `max_pages`: `int` - Maximum pages to scrape

**Returns**: `ScrapedBatch` with items and metadata

**Raises**:
- `NavigationError`: Page load failure
- `ParsingError`: HTML/JSON parsing failure

```python
from src.scraper import VintedScraper
from src.utils import get_config

config = get_config()
scraper = VintedScraper(config.scraper)

batch = scraper.scrape_category("chemises", max_pages=5)
print(f"Scraped {len(batch.items)} items")
```

---

## Data Models

### FashionItem

Fashion item with embeddings and metadata.

**Fields**:
- `item_id`: `str` - Unique identifier
- `title`: `str` - Item title
- `price`: `float` - Price in EUR
- `url`: `str` - Vinted URL
- `image_url`: `str` - Primary image URL
- `local_image_path`: `str | None` - Local image path
- `brand`: `str | None` - Brand name
- `category`: `str | None` - Category
- `clip_embedding`: `np.ndarray | None` - CLIP embedding
- `dino_embedding`: `np.ndarray | None` - DINOv2 embedding
- `additional_metadata`: `dict` - Extra metadata

```python
from src.database.models import FashionItem

item = FashionItem(
    item_id="item_001",
    title="Vintage Jacket",
    price=45.00,
    url="https://vinted.fr/items/001",
    image_url="https://...",
    brand="Levi's",
    category="vestes"
)
```

---

### SearchResult

Search result with similarity scores.

**Fields**:
- `item_id`: `str` - Item identifier
- `similarity_score`: `float` - Fused similarity [0.0, 1.0]
- `clip_score`: `float` - CLIP-only score
- `dino_score`: `float` - DINOv2-only score
- `item`: `FashionItem` - Full item data

---

## Performance Monitoring (`src.utils.performance`)

#### `get_performance_monitor() -> PerformanceMonitor`

Get global performance monitor.

**Returns**: `PerformanceMonitor` singleton

```python
from src.utils.performance import get_performance_monitor

monitor = get_performance_monitor()
monitor.enable()

with monitor.measure("encode_batch", items_count=32):
    embeddings = encoder.encode(images)

monitor.print_report()
```

---

#### `PerformanceMonitor.measure(operation, items_count)`

Context manager for measuring performance.

**Parameters**:
- `operation`: `str` - Operation name
- `items_count`: `int` - Number of items processed

**Usage**:
```python
with monitor.measure("search", items_count=10):
    results = store.search(...)
```

---

## Complete Example

```python
#!/usr/bin/env python3
"""Complete FashionMatch workflow example."""

from src.core import get_hybrid_scorer
from src.database import get_vector_store
from src.utils import get_config
from src.utils.image_utils import load_image
from src.utils.performance import get_performance_monitor

# Initialize
config = get_config()
scorer = get_hybrid_scorer(config.models)

# Get embedding dimensions from encoders
clip_dim = scorer.clip_encoder.embedding_dim
dino_dim = scorer.dino_encoder.embedding_dim

store = get_vector_store(config.database, clip_dim, dino_dim)
monitor = get_performance_monitor()
monitor.enable()

# Load and encode query
query_img = load_image("data/references/shirt.jpg")

with monitor.measure("encode_query", items_count=1):
    clip_emb, dino_emb = scorer.encode_dual([query_img])

# Search database
with monitor.measure("search", items_count=20):
    results = store.search(
        clip_query=clip_emb[0],
        dino_query=dino_emb[0],
        top_k=20
    )

# Display results
print(f"\n🔍 Found {len(results)} similar items:\n")
for i, result in enumerate(results[:10], 1):
    print(f"{i}. {result.item.title}")
    print(f"   Score: {result.similarity_score:.3f} "
          f"(CLIP: {result.clip_score:.3f}, DINO: {result.dino_score:.3f})")
    print(f"   Price: €{result.item.price:.2f}")
    print(f"   URL: {result.item.url}\n")

# Print performance report
monitor.print_report()
```

---

## Error Handling

All functions raise appropriate exceptions:

- `FileNotFoundError`: File not found
- `ValueError`: Invalid parameter values
- `VectorStoreError`: Missing embeddings or invalid items for vector store operations
- `EmbeddingDimensionError`: Embedding dimension mismatch (expected vs actual)
- `BatchInsertError`: Database insertion failure (contains failed IDs)
- `NavigationError`: Scraper navigation failure
- `ParsingError`: HTML/JSON parsing failure
- `ItemNotFoundError`: Requested item not found in vector store

**Error Handling Contracts**:

- **`add_items`**: Raises `VectorStoreError` if any item is missing embeddings, validates all items before insertion
- **`get_by_ids`**: Returns empty list or filtered results; does not raise error for nonexistent IDs
- **`search`**: Returns empty list if vector store is empty; validates embedding dimensions

**Example**:
```python
from src.database.exceptions import VectorStoreError, EmbeddingDimensionError

try:
    # Ensure items have embeddings
    result = store.add_items(items)
    print(f"Added {result.success_count} items")
except VectorStoreError as e:
    print(f"Invalid items (missing embeddings): {e}")
except EmbeddingDimensionError as e:
    print(f"Dimension mismatch: {e}")

# get_by_ids never raises for missing IDs
items = store.get_by_ids(["id1", "nonexistent"])  # Returns only found items
```

---

**API Version**: 1.0  
**Last Updated**: December 2025

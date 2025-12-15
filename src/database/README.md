# Database Module

Vector store management for fashion item embeddings using ChromaDB.

## Overview

The database module implements a dual-collection ChromaDB architecture for storing and retrieving fashion item embeddings. It maintains separate collections for CLIP (semantic) and DINOv2 (structural) embeddings while keeping metadata synchronized across collections.

## Architecture

### Dual-Collection Design

```
Vector Store
├── {collection_name}_clip      # CLIP embeddings (semantic)
│   ├── IDs: [item_id_1, item_id_2, ...]
│   ├── Embeddings: [clip_emb_1, clip_emb_2, ...]
│   └── Metadata: [meta_1, meta_2, ...]
│
└── {collection_name}_dino      # DINOv2 embeddings (structural)
    ├── IDs: [item_id_1, item_id_2, ...]  # Synchronized with CLIP
    ├── Embeddings: [dino_emb_1, dino_emb_2, ...]
    └── Metadata: [meta_1, meta_2, ...]  # Duplicated for consistency
```

**Benefits:**
- Native vector search in both embedding spaces
- Independent distance metrics per collection
- Late fusion of search results with configurable weights
- Clean separation of semantic vs. structural similarity

### Components

#### 1. **VectorStore** (`vector_store.py`)
Main class for interacting with ChromaDB collections.

**Key Features:**
- Dual-collection management with synchronized IDs
- Batch insertion with validation
- Fusion search combining CLIP and DINOv2 results
- CRUD operations: add, search, get, update, delete
- Singleton caching for performance

**Methods:**
```python
# Add items
result = store.add_items(items, batch_size=100)

# Search with dual embeddings
results = store.search(
    clip_query=clip_emb,
    dino_query=dino_emb,
    top_k=10,
    fusion_weights=FusionWeights(clip=0.6, dino=0.4)
)

# Get by IDs
items = store.get_by_ids(["item_1", "item_2"])

# Update item
success = store.update_item("item_1", updated_item)

# Delete items
count = store.delete_items(["item_1", "item_2"])

# Utilities
total = store.count()
all_ids = store.get_all_ids()
store.clear()  # ⚠️ Deletes all data!
```

#### 2. **FashionItem** (`models.py`)
Pydantic model for fashion items with embeddings.

**Fields:**
```python
item_id: str                        # Unique identifier
title: str                          # Item name/description
price: Optional[float]              # Price (validated non-negative)
brand: Optional[str]                # Brand name
category: Optional[str]             # Item category
url: Optional[str]                  # Source URL
image_url: Optional[str]            # Main image URL
clip_embedding: Optional[np.ndarray] # CLIP embedding (float32, 1D)
dino_embedding: Optional[np.ndarray] # DINOv2 embedding (float32, 1D)
additional_metadata: dict[str, Any]  # Extra fields
```

**Validators:**
- `clip_embedding` and `dino_embedding`: Must be 1D float32 numpy arrays
- `price`: Must be non-negative if provided

#### 3. **SearchResult** (`models.py`)
Model for search results with similarity scores.

**Fields:**
```python
item_id: str                    # Item ID
similarity_score: float         # Fused similarity ([-1, 1], validated)
clip_score: float              # CLIP similarity ([-1, 1], validated)
dino_score: float              # DINOv2 similarity ([-1, 1], validated)
item: FashionItem              # Full item data
```

**Validators:**
- All scores must be in range [-1.0, 1.0], otherwise ValueError is raised

#### 4. **Exceptions** (`exceptions.py`)
Custom exception hierarchy for database operations.

```python
VectorStoreError               # Base exception
├── EmbeddingDimensionError    # Wrong embedding dimensions
├── ItemNotFoundError          # Item doesn't exist
├── CollectionError            # Collection access/creation failed
└── BatchInsertError           # Batch operation failed
```

## Usage Examples

### Basic Setup

```python
from src.database import get_vector_store, FashionItem
from src.utils import load_config

# Load configuration
config = load_config()

# Get vector store (singleton)
store = get_vector_store(
    config=config.database,
    clip_dim=512,    # CLIP embedding dimension
    dino_dim=384     # DINOv2 embedding dimension
)
```

### Adding Items

```python
import numpy as np

# Create fashion items with embeddings
items = [
    FashionItem(
        item_id="item_1",
        title="Blue Denim Jacket",
        price=59.99,
        brand="Levi's",
        category="jackets",
        url="https://example.com/item_1",
        image_url="https://example.com/item_1.jpg",
        clip_embedding=np.random.rand(512).astype(np.float32),
        dino_embedding=np.random.rand(384).astype(np.float32)
    ),
    # ... more items
]

# Batch insert (raises BatchInsertError if any fail)
try:
    result = store.add_items(items, batch_size=100)
    print(f"Inserted {result.success_count} items in {result.total_time:.2f}s")
except BatchInsertError as e:
    print(f"Failed to insert {len(e.failed_ids)} items: {e.failed_ids}")
```

### Searching

```python
# Encode query image (using encoders from src.core)
from src.core import get_clip_encoder, get_dino_encoder

clip_enc = get_clip_encoder(config.models.clip_model)
dino_enc = get_dino_encoder(config.models.dino_model)

clip_query = clip_enc.encode_image(query_image)
dino_query = dino_enc.encode_image(query_image)

# Search with fusion
results = store.search(
    clip_query=clip_query,
    dino_query=dino_query,
    top_k=10,
    fusion_weights=config.models.fusion_weights
)

# Process results
for result in results:
    print(f"ID: {result.item_id}")
    print(f"Fused Score: {result.similarity_score:.3f}")
    print(f"  CLIP: {result.clip_score:.3f}, DINO: {result.dino_score:.3f}")
    print(f"Title: {result.item.title}")
    print(f"Price: ${result.item.price}")
    print()
```

### Retrieving Items

```python
# Get specific items by ID (raises ItemNotFoundError if any missing)
try:
    items = store.get_by_ids(["item_1", "item_2", "item_3"])
    for item in items:
        print(f"Found: {item.title}")
except ItemNotFoundError as e:
    print(f"Item not found: {e.item_id}")
```

### Updating Items

```python
# Get existing item
try:
    items = store.get_by_ids(["item_1"])
    item = items[0]
    
    # Modify item
    item.price = 49.99  # Updated price
    item.additional_metadata["discount"] = 0.15
    
    # Update in store
    success = store.update_item("item_1", item)
    print("Item updated successfully")
except ItemNotFoundError as e:
    print(f"Item not found: {e.item_id}")
```

### Deleting Items

```python
# Delete specific items
deleted = store.delete_items(["item_1", "item_2"])
print(f"Deleted {deleted} items")
```

## Integration with Scraper

Convert scraped Vinted items to fashion items:

```python
from src.scraper import VintedScraper
from src.database import vinted_item_to_fashion_item

# Scrape items
scraper = VintedScraper(config.scraper)
batch = await scraper.scrape_category("jackets", max_pages=5)

# Convert to fashion items (without embeddings)
fashion_items = [
    vinted_item_to_fashion_item(vinted_item)
    for vinted_item in batch.items
]

# Add embeddings (using encoders)
for item, vinted_item in zip(fashion_items, batch.items):
    if vinted_item.image_path:
        from PIL import Image
        image = Image.open(vinted_item.image_path)
        
        item.clip_embedding = clip_enc.encode_image(image)
        item.dino_embedding = dino_enc.encode_image(image)

# Insert into vector store
result = store.add_items(fashion_items)
```

## Configuration

Database settings in `config/config.yaml`:

```yaml
database:
  persist_directory: "data/chromadb"  # ChromaDB storage path
  collection_name: "fashion_items"    # Base collection name
  distance_metric: "cosine"           # cosine, l2, or ip
  batch_size: 100                     # Batch insertion size
```

## Performance Considerations

### Batch Operations
- Use batch insertion for better performance (default: 100 items/batch)
- Progress bars shown for large batches (>1 batch)
- Failed items tracked individually

### Singleton Caching
- `get_vector_store()` caches instances by configuration
- Reuses connections for same persist_directory + collection_name
- Reduces initialization overhead

### Embedding Validation
- Dimensions checked once per operation (not per item)
- Early validation prevents partial insertions
- Clear error messages with expected/actual dimensions

### Distance Metrics
- **Cosine**: Best for normalized embeddings (default)
- **L2**: Euclidean distance, sensitive to magnitude
- **IP**: Inner product, for non-normalized embeddings

## Error Handling

The database module raises custom exceptions for various failure modes:

```python
from src.database.exceptions import (
    BatchInsertError,
    EmbeddingDimensionError,
    ItemNotFoundError,
    VectorStoreError
)

# Adding items - raises BatchInsertError if any fail
try:
    result = store.add_items(items)
    print(f"All {result.success_count} items inserted successfully")
except EmbeddingDimensionError as e:
    print(f"Dimension mismatch: expected {e.expected_dim}, got {e.actual_dim}")
except BatchInsertError as e:
    print(f"Failed to insert {len(e.failed_ids)} items: {e.failed_ids}")
except VectorStoreError as e:
    print(f"Database error: {e}")

# Getting items - raises ItemNotFoundError if any ID is missing
try:
    items = store.get_by_ids(["item_1", "item_2"])
    print(f"Retrieved {len(items)} items")
except ItemNotFoundError as e:
    print(f"Item not found: {e.item_id}")

# Updating items - raises ItemNotFoundError if ID doesn't exist
try:
    success = store.update_item("item_1", updated_item)
    print("Item updated successfully")
except ItemNotFoundError as e:
    print(f"Cannot update, item not found: {e.item_id}")
```

**Exception Behavior:**
- `add_items()`: Raises `BatchInsertError` immediately if any items fail to insert
- `get_by_ids()`: Raises `ItemNotFoundError` on first missing item ID
- `update_item()`: Raises `ItemNotFoundError` if item doesn't exist
- `delete_items()`: Returns count of confirmed deletions (does not raise on missing items)

## Limitations

1. **Python Version**: Requires Python 3.10-3.12 (ChromaDB constraint)
2. **Embedding Dimensions**: Must be consistent across all items in a collection
3. **ID Synchronization**: Manual deletion from one collection breaks sync (use `delete_items()`)
4. **Metadata Size**: ChromaDB has limits on metadata size (~1MB per item)
5. **Scale**: ChromaDB is best for datasets <10M items (consider Qdrant/Weaviate for larger)

## Testing

```python
# Test basic operations
def test_vector_store():
    config = load_config()
    store = get_vector_store(config.database, 512, 384)
    
    # Test add
    item = FashionItem(
        item_id="test_1",
        title="Test Item",
        clip_embedding=np.random.rand(512).astype(np.float32),
        dino_embedding=np.random.rand(384).astype(np.float32)
    )
    result = store.add_items([item])
    assert result.success_count == 1
    
    # Test search
    results = store.search(
        clip_query=item.clip_embedding,
        dino_query=item.dino_embedding,
        top_k=1
    )
    assert len(results) == 1
    assert results[0].item_id == "test_1"
    
    # Test delete
    deleted = store.delete_items(["test_1"])
    assert deleted == 1
    
    print("All tests passed!")
```

## Next Steps

1. **Implement UI**: Streamlit interface for searching and browsing
2. **Add Filtering**: Support metadata filters in search (price range, brand, category)
3. **Implement Reranking**: Two-stage search with reranking
4. **Add Monitoring**: Track search latency, storage size, hit rates
5. **Migration Tools**: Import/export utilities for backup/restore

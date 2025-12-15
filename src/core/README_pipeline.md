# Embedding Pipeline

End-to-end orchestration for generating and storing dual fashion item embeddings.

## Overview

The `EmbeddingPipeline` class connects all FashionMatch components into a unified workflow:
- **Input**: Reference images or scraped Vinted batches
- **Processing**: Dual encoding with CLIP + DINOv2
- **Output**: Synchronized embeddings stored in ChromaDB

The pipeline supports **incremental updates** (skips already-processed items), **batch processing** (efficient memory usage), and **detailed progress tracking** (tqdm progress bars + structured logs).

## Architecture

```
┌─────────────────┐
│  Input Sources  │
├─────────────────┤
│ • References    │  ──┐
│ • Scraped Data  │    │
└─────────────────┘    │
                       ▼
┌─────────────────────────────────┐
│     EmbeddingPipeline           │
├─────────────────────────────────┤
│ • Load images from disk         │
│ • Filter existing embeddings    │
│ • Batch encode (CLIP + DINOv2)  │
│ • Create FashionItem objects    │
│ • Store in ChromaDB             │
└─────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────┐
│      Vector Store (ChromaDB)    │
├─────────────────────────────────┤
│ • {collection}_clip             │
│ • {collection}_dino             │
└─────────────────────────────────┘
```

## Components

### 1. **EmbeddingPipeline Class**

Main orchestrator managing encoders, vector store, and processing logic.

**Initialization:**
```python
from src.core import EmbeddingPipeline

# Initialize with default config
pipeline = EmbeddingPipeline()

# Or with custom config
pipeline = EmbeddingPipeline(config_path="custom_config.yaml")
```

**Key Attributes:**
- `config`: Application configuration (AppConfig)
- `scorer`: HybridScorer with CLIP + DINOv2 encoders
- `vector_store`: ChromaDB VectorStore instance

### 2. **PipelineStats Model**

Pydantic model tracking processing statistics:

```python
class PipelineStats(BaseModel):
    total_items: int          # Total items discovered
    processed_items: int      # Successfully encoded and stored
    skipped_items: int        # Already in vector store
    failed_items: int         # Errors during processing
    failed_ids: list[str]     # IDs of failed items
    duration_seconds: float   # Total processing time
    source: str              # "references" or "scraped"
```

## Usage

### Process Reference Images

Reference images are standalone files without scraper metadata:

```python
from pathlib import Path
from src.core import EmbeddingPipeline

pipeline = EmbeddingPipeline()

# Process all images in directory
stats = pipeline.process_references(
    reference_dir=Path("data/references"),
    batch_size=32
)

print(f"Processed: {stats.processed_items}")
print(f"Skipped: {stats.skipped_items}")
print(f"Failed: {stats.failed_items}")
```

**What happens:**
1. Scans directory recursively for `.jpg`, `.jpeg`, `.png` files
2. Generates unique IDs: `ref_{filename}_{timestamp}`
3. Checks `vector_store.get_all_ids()` to skip existing items
4. Loads images in batches
5. Encodes with CLIP + DINOv2
6. Creates `FashionItem` objects with minimal metadata:
   - `title`: filename
   - `price`: 0.0
   - `url`: `file://{absolute_path}`
   - `local_image_path`: path to file
7. Stores in vector store

### Process Scraped Batches

Scraped batches are JSON files created by `VintedScraper`:

```python
pipeline = EmbeddingPipeline()

# Process all batches in directory
stats = pipeline.process_scraped_batches(
    scraped_dir=Path("data/scraped"),
    batch_size=32
)

print(f"Processed: {stats.processed_items}")
print(f"Duration: {stats.duration_seconds:.2f}s")
```

**What happens:**
1. Lists all batch files via `list_batches()`
2. Loads each batch with `load_batch()`
3. Filters items already in vector store
4. For each new item:
   - Checks `local_image_paths` exists
   - Loads primary image from disk
   - Converts `VintedItem` → `FashionItem` via `vinted_item_to_fashion_item()`
5. Encodes in sub-batches
6. Stores with full metadata (title, price, brand, category, URL, etc.)

### Process Everything

Process both sources in one call:

```python
pipeline = EmbeddingPipeline()

results = pipeline.process_all(
    reference_dir=Path("data/references"),
    scraped_dir=Path("data/scraped"),
    batch_size=32
)

# Results is dict with "references" and "scraped" keys
for source, stats in results.items():
    print(f"{source}: {stats.processed_items} items processed")
```

## CLI Usage

Run the pipeline from command line:

### Basic Commands

```bash
# Process everything (references + scraped)
python -m src.core.embedding_pipeline --mode all

# Process only references
python -m src.core.embedding_pipeline --mode references

# Process only scraped batches
python -m src.core.embedding_pipeline --mode scraped
```

### Advanced Options

```bash
# Custom config file
python -m src.core.embedding_pipeline --config custom_config.yaml

# Custom directories
python -m src.core.embedding_pipeline \
    --mode all \
    --references-dir /path/to/references \
    --scraped-dir /path/to/scraped

# Larger batch size for more GPU memory
python -m src.core.embedding_pipeline --batch-size 64

# Clear vector store before processing (DANGEROUS!)
python -m src.core.embedding_pipeline --force --mode all
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--config` | Path | `config/config.yaml` | Path to config file |
| `--mode` | Choice | `all` | Processing mode: `references`, `scraped`, or `all` |
| `--references-dir` | Path | From config | Reference images directory |
| `--scraped-dir` | Path | From config | Scraped batches directory |
| `--batch-size` | int | 32 | Encoding batch size |
| `--force` | Flag | False | Clear vector store (requires confirmation) |

### Output

The CLI prints a formatted table with results:

```
================================================================================
EMBEDDING PIPELINE RESULTS
================================================================================

REFERENCES:
  Total items:      150
  Processed:        145
  Skipped:          5
  Failed:           0
  Duration:         23.45s

SCRAPED:
  Total items:      523
  Processed:        498
  Skipped:          20
  Failed:           5
  Duration:         78.92s
  Failed IDs:       vinted_12345, vinted_67890, ... and 3 more

TOTAL:
  Processed:        643
  Failed:           5
  Total Duration:   102.37s
================================================================================
```

## Incremental Updates

The pipeline is **idempotent** - re-running won't create duplicates:

```python
# First run: processes 100 new items
stats1 = pipeline.process_references(ref_dir)
print(stats1.processed_items)  # 100

# Second run: skips all existing items
stats2 = pipeline.process_references(ref_dir)
print(stats2.skipped_items)    # 100
print(stats2.processed_items)  # 0

# Add new images and run again: only processes new ones
# (new_images added to ref_dir)
stats3 = pipeline.process_references(ref_dir)
print(stats3.skipped_items)    # 100
print(stats3.processed_items)  # 10 (new images)
```

**How it works:**
1. Query `vector_store.get_all_ids()` at start
2. Cache existing IDs in memory
3. Filter input items: `if item_id not in existing_ids`
4. Only encode and store new items

**Benefits:**
- **Resumable**: Interrupted runs can be restarted
- **Efficient**: No wasted encoding on existing items
- **Safe**: No duplicate embeddings in vector store

## Progress Tracking

The pipeline provides detailed progress information:

### Progress Bars (tqdm)

```
Processing references: 100%|████████████| 50/50 [00:12<00:00,  4.12img/s]
Processing batches: 100%|█████████████████| 15/15 [01:23<00:00,  5.56s/batch]
```

### Structured Logs

```
INFO  src.core.embedding_pipeline - Initializing EmbeddingPipeline
INFO  src.core.encoders.clip_encoder - Loading CLIP model: openai/clip-vit-base-patch32
INFO  src.core.encoders.dino_encoder - Loading DINOv2 model: dinov2_vits14
INFO  src.database.vector_store - Collections initialized: fashion_items_clip (0 items)
INFO  src.core.embedding_pipeline - Processing reference images from data/references/
INFO  src.core.embedding_pipeline - Found 150 image files
INFO  src.core.embedding_pipeline - Found 5 existing items in vector store
INFO  src.core.embedding_pipeline - Processing 145 new items, skipping 5 existing
INFO  src.core.embedding_pipeline - Reference processing complete: 145 processed, 5 skipped, 0 failed in 23.45s
```

## Error Handling

The pipeline handles errors gracefully at multiple levels:

### Per-Item Errors (Non-Fatal)

These log warnings and skip the item:

- **Image load failure**: Corrupted file, wrong format
- **Missing image path**: `local_image_paths` is empty
- **Image file deleted**: Path exists in metadata but not on disk

```python
stats = pipeline.process_references(ref_dir)
print(f"Failed items: {stats.failed_items}")
print(f"Failed IDs: {stats.failed_ids}")
```

### Batch Errors

Handled with `BatchInsertError` from vector store:

```python
# Pipeline catches BatchInsertError internally
# and logs failed IDs, continuing with remaining batches
```

### Fatal Errors

These raise exceptions and stop execution:

- **Config load failure**: Invalid YAML, missing required fields
- **Encoder initialization failure**: Model download error, OOM
- **Vector store connection failure**: ChromaDB not accessible
- **Invalid directory**: `reference_dir` or `scraped_dir` doesn't exist

```python
try:
    pipeline = EmbeddingPipeline()
except Exception as e:
    print(f"Failed to initialize pipeline: {e}")
```

## Configuration

Pipeline behavior is controlled by `config/config.yaml`:

```yaml
models:
  clip_model: "openai/clip-vit-base-patch32"  # CLIP model
  dino_model: "dinov2_vits14"                   # DINOv2 model
  fusion_weights:
    clip: 0.6
    dino: 0.4

database:
  persist_directory: "data/chromadb"
  collection_name: "fashion_items"
  batch_size: 100

scraper:
  output_dir: "data/scraped"
  references_dir: "data/references"
```

## Integration Examples

### With Jupyter Notebooks

```python
# notebook.ipynb
from src.core import EmbeddingPipeline
from pathlib import Path

# Initialize
pipeline = EmbeddingPipeline()

# Process with custom settings
stats = pipeline.process_references(
    reference_dir=Path("data/my_references"),
    batch_size=16  # Smaller batches for limited GPU memory
)

# Display results
import pandas as pd
df = pd.DataFrame([{
    "Source": stats.source,
    "Total": stats.total_items,
    "Processed": stats.processed_items,
    "Skipped": stats.skipped_items,
    "Failed": stats.failed_items,
    "Duration": f"{stats.duration_seconds:.2f}s"
}])
display(df)
```

### With Custom Workflows

```python
from src.core import EmbeddingPipeline
from src.database import get_vector_store

# Initialize pipeline
pipeline = EmbeddingPipeline()

# Process specific batches
batch_dir = Path("data/scraped")
specific_batches = ["20231215_123456", "20231216_234567"]

for batch_id in specific_batches:
    batch = load_batch(batch_dir / batch_id)
    # ... custom filtering logic ...
    # Use pipeline._encode_and_store_batch() for encoding
```

### Monitoring Vector Store

```python
pipeline = EmbeddingPipeline()

# Check store before processing
print(f"Items in store: {pipeline.vector_store.count()}")
print(f"Existing IDs: {len(pipeline.vector_store.get_all_ids())}")

# Process
stats = pipeline.process_all(ref_dir, scraped_dir)

# Check store after processing
print(f"Items after: {pipeline.vector_store.count()}")
```

## Performance Considerations

### Batch Size

- **Small (8-16)**: Low GPU memory usage, slower processing
- **Medium (32-64)**: Balanced performance (recommended)
- **Large (128+)**: Fastest, requires high GPU memory

```python
# Adjust based on available GPU memory
pipeline.process_references(ref_dir, batch_size=16)  # ~2GB VRAM
pipeline.process_references(ref_dir, batch_size=64)  # ~8GB VRAM
```

### Parallel Processing

Current implementation is sequential per source. Future optimizations:
- Parallel batch loading (async I/O)
- Multi-GPU encoding (data parallelism)
- Concurrent vector store insertion

### Memory Management

- Images are loaded in batches and released after encoding
- Encoders use singleton caching (models loaded once)
- Vector store uses batched insertion (configurable batch size)

## Troubleshooting

### Issue: Out of Memory (OOM)

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size: `--batch-size 16`
2. Use smaller models: Change `clip_model` in config
3. Process on CPU: Set `device: "cpu"` in config

### Issue: Slow Processing

**Symptoms**: <1 item/second throughput

**Solutions:**
1. Increase batch size (if GPU allows)
2. Check GPU utilization: `nvidia-smi`
3. Verify models are on GPU (check logs)

### Issue: High Skipped Count

**Symptoms**: Most items show as "skipped"

**Solutions:**
1. Check if vector store already populated: `vector_store.count()`
2. Verify IDs are unique (reference IDs use timestamp)
3. Use `--force` to clear and reprocess (CAUTION: deletes data)

### Issue: Missing Images

**Symptoms**: High `failed_items` count, warnings in logs

**Solutions:**
1. Verify image paths in scraped batches
2. Check `local_image_paths` is populated
3. Ensure images weren't deleted after scraping

## Testing

### Basic Smoke Test

```python
from src.core import EmbeddingPipeline
from pathlib import Path

# Initialize
pipeline = EmbeddingPipeline()

# Test with small dataset
test_dir = Path("tests/fixtures/images")
stats = pipeline.process_references(test_dir, batch_size=4)

assert stats.processed_items > 0
assert stats.failed_items == 0
print("✅ Pipeline test passed")
```

### Incremental Update Test

```python
# First run
stats1 = pipeline.process_references(test_dir)
count1 = stats1.processed_items

# Second run (should skip all)
stats2 = pipeline.process_references(test_dir)
assert stats2.skipped_items == count1
assert stats2.processed_items == 0
print("✅ Incremental test passed")
```

## Next Steps

After running the pipeline:

1. **Verify embeddings**: Check vector store count matches expected
2. **Test search**: Use `vector_store.search()` with query image
3. **Build UI**: Create Streamlit interface for image search
4. **Monitor performance**: Track encoding throughput, storage size
5. **Set up automation**: Schedule periodic processing of new scraped data

## Example Workflow

Complete workflow from scraping to searchable embeddings:

```bash
# 1. Scrape items
python -m src.scraper.cli \
    --category women-jackets \
    --max-pages 10 \
    --enrich-details

# 2. Generate embeddings
python -m src.core.embedding_pipeline --mode all

# 3. Verify storage
python -c "
from src.core import EmbeddingPipeline
p = EmbeddingPipeline()
print(f'Total items: {p.vector_store.count()}')
"

# 4. Test search (from notebook or UI)
# See UI documentation for search interface
```

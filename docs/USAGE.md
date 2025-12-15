# FashionMatch Usage Guide

Complete guide to using FashionMatch for fashion item search and discovery.

## Quick Start

```bash
# 1. Scrape Vinted data
python -m src.scraper.cli --category "chemises" --pages 5

# 2. Process embeddings
python -m src.core.embedding_pipeline --mode all

# 3. Launch UI
streamlit run src/ui/app.py
```

## Workflow Overview

1. **Scrape** → Collect fashion items from Vinted
2. **Embed** → Generate AI embeddings (CLIP + DINOv2)
3. **Search** → Find similar items using visual similarity
4. **Refine** → Use feedback to improve results

---

## 1. Web Scraping

### Basic Scraping

Scrape a category with default settings:

```bash
python -m src.scraper.cli --category "chemises" --pages 10
```

### Advanced Options

```bash
python -m src.scraper.cli \
  --category "robes" \
  --pages 20 \
  --output ./data/scraped \
  --download-images \
  --max-concurrent 5
```

### Scraper Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--category` | Vinted category (required) | - |
| `--pages` | Number of pages to scrape | 10 |
| `--output` | Output directory | `./data/scraped` |
| `--download-images` | Download product images | False |
| `--max-concurrent` | Concurrent downloads | 5 |
| `--delay` | Delay between requests (seconds) | 1-3 |

### Available Categories

Common Vinted categories:
- `chemises` - Shirts
- `robes` - Dresses
- `pantalons` - Pants
- `vestes` - Jackets
- `chaussures` - Shoes
- `sacs` - Bags
- `pull-sweats` - Sweaters
- `t-shirts-debardeurs` - T-shirts

### Output Format

Scraped data is saved as JSON:

```json
{
  "items": [
    {
      "item_id": "1234567890",
      "title": "Vintage Denim Jacket",
      "price": 45.00,
      "url": "https://www.vinted.fr/items/...",
      "image_url": "https://...",
      "brand": "Levi's",
      "size": "L",
      "condition": "Good",
      "category": "Vestes"
    }
  ],
  "category": "vestes",
  "timestamp": "2025-01-01T12:00:00"
}
```

---

## 2. Embedding Generation

### Process All Sources

Generate embeddings for both references and scraped items:

```bash
python -m src.core.embedding_pipeline --mode all
```

### Process Only References

Process reference images in `data/references/`:

```bash
python -m src.core.embedding_pipeline --mode references
```

### Process Only Scraped Items

Process scraped items from `data/scraped/`:

```bash
python -m src.core.embedding_pipeline --mode scraped
```

### Pipeline Options

```bash
python -m src.core.embedding_pipeline \
  --mode all \
  --batch-size 64 \
  --config ./config/config.yaml \
  --profile
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | Processing mode (all/references/scraped) | all |
| `--batch-size` | Batch size for encoding | 32 |
| `--config` | Config file path | `./config/config.yaml` |
| `--force` | Clear database before processing | False |
| `--profile` | Enable performance profiling | False |

### Clear and Reprocess

**WARNING**: This deletes all embeddings!

```bash
python -m src.core.embedding_pipeline --force --mode all
```

### Monitor Progress

The pipeline shows real-time progress:

```
Processing references...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 50/50 100% 0:00:15
Processed: 50 | Skipped: 0 | Failed: 0

Processing scraped items...
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 200/200 100% 0:01:30
Processed: 200 | Skipped: 0 | Failed: 3

Pipeline Statistics:
  Total processed: 250
  Total skipped: 0
  Total failed: 3
  Duration: 105.3s
```

### Performance Profiling

Enable detailed performance metrics with `--profile`:

```bash
python -m src.core.embedding_pipeline --mode all --profile
```

Output includes timing and memory usage:

```
================================================================================
PERFORMANCE REPORT
================================================================================

encode_batch:
  Calls:            8
  Total Duration:   45.23s
  Avg Duration:     5.654s
  Min Duration:     4.821s
  Max Duration:     6.102s
  Avg Memory:       +234.5 MB
  Total Items:      250
  Avg Throughput:   44.2 items/s
  Avg GPU Memory:   +512.3 MB

insert_batch:
  Calls:            8
  Total Duration:   12.45s
  Avg Duration:     1.556s
  Min Duration:     1.234s
  Max Duration:     1.892s
  Avg Memory:       +12.3 MB
  Total Items:      250
  Avg Throughput:   160.3 items/s
================================================================================
```

---

## 3. Streamlit UI

### Launch Application

```bash
# Using launcher script
python run_app.py

# Or directly
streamlit run src/ui/app.py

# Custom port
streamlit run src/ui/app.py --server.port 8502
```

Access at `http://localhost:8501`

### UI Workflow

#### Step 1: Upload References

1. Click "Browse files" in sidebar
2. Select 1-5 reference images (JPG/PNG)
3. Preview thumbnails appear
4. Click "🔄 Encode References"

#### Step 2: Search

1. Click "🔍 Search Similar Items"
2. Wait for results (progress spinner)
3. Browse recommended items

#### Step 3: Provide Feedback

For each result:
- Click **👍 Like** if item matches your style
- Click **👎 Dislike** if it doesn't

Fusion weights adjust automatically based on feedback!

#### Step 4: Apply Filters

In the sidebar:
- **Price range**: Adjust slider
- **Categories**: Select categories to include
- **Similarity threshold**: Set minimum score
- **Sort by**: Similarity, price (asc/desc)

### Fusion Weights

The UI displays real-time fusion weights:

```
CLIP:  ████████████░░░░░░░░ 60%
DINO:  ████████░░░░░░░░░░░░ 40%
```

- **CLIP**: Semantic understanding (style, context)
- **DINO**: Structural analysis (patterns, textures)

Weights adjust by ±0.05 per feedback event.

### Keyboard Shortcuts

- **Ctrl+R**: Refresh app
- **Ctrl+K**: Focus search
- **Esc**: Close modals

---

## 4. Configuration

### Model Selection

Edit `config/config.yaml`:

```yaml
models:
  clip_model: "openai/clip-vit-large-patch14"  # Larger = better quality
  dino_model: "dinov2_vitl14"  # vits14 < vitb14 < vitl14 < vitg14
  device: "cuda"  # Use GPU
```

### CLIP Models

| Model | Embedding Dim | Speed | Quality |
|-------|---------------|-------|---------|
| `clip-vit-base-patch32` | 512 | Fastest | Good |
| `clip-vit-base-patch16` | 512 | Fast | Better |
| `clip-vit-large-patch14` | 768 | Slower | Best |

### DINOv2 Models

| Model | Embedding Dim | Speed | Quality |
|-------|---------------|-------|---------|
| `dinov2_vits14` | 384 | Fastest | Good |
| `dinov2_vitb14` | 768 | Fast | Better |
| `dinov2_vitl14` | 1024 | Slower | Great |
| `dinov2_vitg14` | 1536 | Slowest | Best |

### Fusion Weights

Adjust initial weights:

```yaml
fusion_weights:
  clip: 0.7  # More semantic weight
  dino: 0.3  # Less structural weight
```

### Database Settings

```yaml
database:
  persist_directory: "./data/chroma"
  batch_size: 64  # Increase for faster processing
  distance_metric: "cosine"  # or "l2", "ip"
```

---

## 5. Advanced Usage

### Batch Processing

Process large datasets efficiently:

```bash
# Scrape multiple categories
for category in chemises robes pantalons vestes; do
  python -m src.scraper.cli --category "$category" --pages 20
done

# Process all at once
python -m src.core.embedding_pipeline --mode scraped --batch-size 128
```

### Performance Optimization

#### GPU Acceleration

```yaml
models:
  device: "cuda"
database:
  batch_size: 128  # Larger batches for GPU
```

#### CPU Mode

```yaml
models:
  device: "cpu"
database:
  batch_size: 16  # Smaller batches for CPU
```

### Custom Search Scripts

Create custom search scripts:

```python
from src.core import get_hybrid_scorer
from src.database import get_vector_store
from src.utils import get_config
from src.utils.image_utils import load_image

# Initialize
config = get_config()
scorer = get_hybrid_scorer(config.models)
vector_store = get_vector_store(config.database)

# Load and encode query image
query_img = load_image("path/to/reference.jpg")
clip_emb, dino_emb = scorer.encode_dual([query_img])

# Search
results = vector_store.search(
    clip_query=clip_emb[0],
    dino_query=dino_emb[0],
    top_k=20
)

# Display results
for result in results:
    print(f"{result.item.title}: {result.similarity_score:.3f}")
```

---

## 6. Troubleshooting

### No Results After Search

**Check**:
1. Database is populated: `python -m src.core.embedding_pipeline --mode all`
2. References are encoded (green checkmark in UI)
3. Filters aren't too restrictive

### Slow Performance

**Solutions**:
- Use GPU: `device: "cuda"`
- Reduce batch size if OOM errors
- Use smaller models: `dinov2_vits14`, `clip-vit-base-patch32`
- Close other applications

### Poor Search Quality

**Improve by**:
1. Use higher-quality reference images
2. Upload multiple references (2-5 images)
3. Adjust fusion weights based on feedback
4. Try larger models for better embeddings

### Scraper Errors

**Common issues**:
- **Rate limiting**: Increase delays in config
- **Timeout**: Increase `timeout` setting
- **Invalid category**: Check category name spelling
- **Network errors**: Check internet connection

### Database Issues

**Reset database**:
```bash
rm -rf data/chroma/*
python -m src.core.embedding_pipeline --force --mode all
```

---

## 7. Best Practices

### Reference Images

✅ **Do**:
- Use clear, well-lit product photos
- Include multiple angles/variations
- Match the style you're searching for
- Use 2-5 references for best results

❌ **Don't**:
- Use blurry or dark images
- Include multiple items in one image
- Use screenshots with UI elements
- Upload unrelated images

### Scraping

✅ **Do**:
- Respect rate limits (use delays)
- Scrape during off-peak hours
- Download images incrementally
- Verify scraped data quality

❌ **Don't**:
- Scrape aggressively (risk of blocking)
- Ignore robots.txt
- Store unnecessary data
- Scrape without permission

### Search Workflow

✅ **Do**:
- Provide consistent feedback
- Use filters to narrow results
- Try different fusion weights
- Experiment with multiple references

❌ **Don't**:
- Expect perfect results immediately
- Ignore feedback mechanism
- Use filters that exclude all results
- Mix unrelated reference styles

---

## 8. Example Workflows

### Fashion Designer Use Case

```bash
# 1. Collect inspiration from Vinted
python -m src.scraper.cli --category "robes" --pages 50

# 2. Process embeddings
python -m src.core.embedding_pipeline --mode all

# 3. Launch UI
streamlit run src/ui/app.py

# 4. Upload design sketches as references
# 5. Find similar real products
# 6. Refine with feedback
```

### Vintage Collector Use Case

```bash
# 1. Scrape vintage categories
python -m src.scraper.cli --category "vestes" --pages 30

# 2. Add personal collection as references
cp ~/vintage_collection/*.jpg data/references/

# 3. Generate embeddings
python -m src.core.embedding_pipeline --mode all

# 4. Search for similar vintage items
```

---

## 9. API Usage

See [API.md](API.md) for programmatic usage.

---

## 10. Next Steps

- Explore [notebooks/](../notebooks/) for experimentation
- Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
- Check [API.md](API.md) for developer documentation
- Join community discussions on GitHub

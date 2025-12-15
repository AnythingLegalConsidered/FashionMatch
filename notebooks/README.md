# FashionMatch Notebooks

Interactive Jupyter notebooks for experimentation, analysis, and debugging.

## Setup

Install Jupyter and visualization libraries:

```bash
pip install jupyter matplotlib seaborn scikit-learn umap-learn
```

Launch Jupyter:

```bash
jupyter notebook
```

## Available Notebooks

### 1. Quick Start (`01_quick_start.ipynb`)
- Load configuration and initialize models
- Encode a reference image
- Search vector store
- Visualize similarity scores
- **Perfect for**: First-time users and onboarding

### 2. Encoder Comparison (`02_encoder_comparison.ipynb`)
- Compare CLIP vs DINOv2 embeddings
- Visualize embedding spaces (t-SNE/UMAP)
- Analyze model strengths for different items
- Test fusion weights (α=0.3/0.5/0.7)
- **Perfect for**: Understanding model behavior

### 3. Scraper Demo (`03_scraper_demo.ipynb`)
- Run scraper on single category
- Parse and display scraped items
- Download images interactively
- Handle errors and retries
- **Perfect for**: Debugging scraper issues

### 4. Performance Profiling (`04_performance_profiling.ipynb`)
- Benchmark encoding speed
- Measure search latency
- Profile memory usage
- Compare CPU vs GPU performance
- **Perfect for**: Optimization

### 5. Fusion Weight Tuning (`05_fusion_weight_tuning.ipynb`)
- Load pre-computed embeddings
- Simulate feedback loop
- Visualize weight evolution
- Evaluate precision@k
- **Perfect for**: Hyperparameter tuning

## Prerequisites

Before running notebooks, ensure:

1. **Dependencies installed**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Data available**:
   ```bash
   # Scrape some data
   python -m src.scraper.cli --category "chemises" --pages 5
   
   # Generate embeddings
   python -m src.core.embedding_pipeline --mode all
   ```

3. **Jupyter installed**:
   ```bash
   pip install jupyter matplotlib seaborn
   ```

## Usage Examples

### Quick Similarity Search

```python
# In notebook
from src.core import get_hybrid_scorer
from src.database import get_vector_store
from src.utils import get_config
from src.utils.image_utils import load_image

# Initialize
config = get_config()
scorer = get_hybrid_scorer(config.models)
store = get_vector_store(config.database)

# Encode and search
image = load_image("data/references/shirt.jpg")
clip_emb, dino_emb = scorer.encode_dual([image])
results = store.search(clip_emb[0], dino_emb[0], top_k=10)

# Display
for r in results:
    print(f"{r.item.title}: {r.similarity_score:.3f}")
```

### Visualize Embeddings

```python
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Get embeddings from database
items = store.list_ids()[:100]
results = store.get_by_ids(items)
clip_embs = [r.item.clip_embedding for r in results]

# Reduce dimensions
tsne = TSNE(n_components=2, random_state=42)
embs_2d = tsne.fit_transform(clip_embs)

# Plot
plt.figure(figsize=(10, 10))
plt.scatter(embs_2d[:, 0], embs_2d[:, 1])
plt.title("CLIP Embeddings (t-SNE)")
plt.show()
```

## Tips

- **Use GPU**: Set `device: "cuda"` in config for faster processing
- **Sample data**: Work with subsets for quick experiments
- **Save checkpoints**: Save intermediate results to avoid re-computation
- **Clear outputs**: Clear notebook outputs before committing to git

## Troubleshooting

### Kernel Dies During Encoding

**Solution**: Reduce batch size or use CPU mode

```python
# Use smaller batches
config.database.batch_size = 16
config.models.device = "cpu"
```

### Out of Memory

**Solution**: Process in smaller chunks

```python
# Process in batches
for i in range(0, len(items), 100):
    batch = items[i:i+100]
    process_batch(batch)
```

### Plots Not Showing

**Solution**: Use inline backend

```python
%matplotlib inline
import matplotlib.pyplot as plt
```

## Contributing

Add new notebooks following the naming convention:
- `XX_descriptive_name.ipynb`
- Include markdown cells explaining each step
- Add comments in code cells
- Test with fresh kernel before committing

## Resources

- [Jupyter Documentation](https://jupyter.org/documentation)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)

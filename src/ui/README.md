# FashionMatch UI

Interactive visual search interface for finding similar fashion items using dual AI models (CLIP + DINOv2).

## 🚀 Quick Start

### Launch the App

**Option 1: Using launcher script** (recommended)
```bash
python run_app.py
```

**Option 2: Direct Streamlit command**
```bash
streamlit run src/ui/app.py
```

**Option 3: From VS Code**
- Open `src/ui/app.py`
- Press F5 or click "Run" in the top-right corner

The app will open in your default browser at `http://localhost:8501`

### Prerequisites

1. **Populated Database**: Run the embedding pipeline first:
   ```bash
   python -m src.core.embedding_pipeline --mode all
   ```

2. **Dependencies Installed**: Ensure Streamlit is installed:
   ```bash
   pip install streamlit
   ```

## ✨ Features

### 1. **Reference Image Upload**
- Drag-and-drop multiple images (JPG, JPEG, PNG)
- Preview uploaded references in sidebar
- One-click encoding with CLIP + DINOv2

### 2. **Hybrid Visual Search**
- Dual-embedding query combining semantic (CLIP) and structural (DINOv2) similarity
- Configurable fusion weights for personalized results
- Top-K retrieval with similarity scores

### 3. **Interactive Feedback Loop**
- 👍 Like / 👎 Dislike buttons on each result
- Automatic weight adjustment based on feedback
- Transparency: See individual CLIP/DINO scores

### 4. **Advanced Filtering**
- **Price range**: Slider to filter by price
- **Categories**: Multi-select category filter
- **Similarity threshold**: Minimum score filter
- **Sorting**: By similarity, price (asc/desc)

### 5. **Fusion Weight Visualization**
- Visual gauge showing CLIP vs DINOv2 proportions
- Numeric display with 3 decimal precision
- Reset button to restore defaults
- Explanatory tooltips

### 6. **Statistics Dashboard**
- Total results count
- Average similarity score
- Feedback summary (likes/dislikes)
- Model performance metrics

### 7. **Responsive Grid Layout**
- 3-column item grid
- Hover effects and smooth animations
- Mobile-friendly design
- Pagination for large result sets

## 📖 User Workflow

### Step 1: Upload References
1. Click "**Browse files**" in the sidebar or drag images
2. Preview thumbnails appear
3. Click "**🔄 Encode References**" to process images

### Step 2: Search
1. Click "**🔍 Search Similar Items**" button
2. Wait for results (progress spinner shows status)
3. View recommended items in the main panel

### Step 3: Review Results
- **Item cards** show:
  - Product image
  - Title, brand, category
  - Price
  - Similarity scores (Fused, CLIP, DINO)
  - Link to original listing
  - Feedback buttons

### Step 4: Provide Feedback
- Click **👍 Like** if item matches your style
- Click **👎 Dislike** if item doesn't match
- Weights automatically adjust based on which model (CLIP/DINO) scored higher

### Step 5: Refine Search
- **Adjust filters** in sidebar (price, category, similarity)
- **Manually tune weights** if desired
- **Re-run search** with new weights for updated results

## ⚙️ Configuration

Settings are controlled via `config/config.yaml`:

```yaml
models:
  clip_model: "openai/clip-vit-base-patch32"
  dino_model: "dinov2_vits14"
  fusion_weights:
    clip: 0.6  # Default CLIP weight
    dino: 0.4  # Default DINOv2 weight
  device: "auto"

database:
  collection_name: "fashion_items"
  distance_metric: "cosine"
```

**Key Settings:**
- `clip_model`: CLIP model identifier (Hugging Face)
- `dino_model`: DINOv2 variant (vits14, vitb14, vitl14, vitg14)
- `fusion_weights`: Initial α (CLIP) and β (DINOv2) values
- `distance_metric`: Similarity metric (cosine, l2, ip)

## 🎨 UI Components

### State Management (`state_manager.py`)
- Session state persistence across reruns
- Reference image storage with embeddings
- Search results caching
- Feedback history tracking
- Filter settings

### Components (`components/`)
- **feedback_buttons.py**: Like/dislike interaction
- **item_card.py**: Fashion item display
- **filters.py**: Filter sidebar controls
- **weight_display.py**: Fusion weight gauge
- **stats_panel.py**: Statistics dashboard

### Utilities (`utils.py`)
- Image loading and preprocessing
- Price and score formatting
- Filter application
- Embedding averaging

### Styling (`styles.py`)
- Custom CSS for visual appeal
- Hover effects and animations
- Responsive breakpoints
- Color-coded scores

## 🔧 Advanced Features

### Weight Adjustment Algorithm

When user provides feedback:
1. Compare CLIP vs DINOv2 scores for the item
2. **On Like**: Increase weight of better-performing model by 0.05
3. **On Dislike**: Decrease weight of better-performing model by 0.05
4. Re-normalize weights to sum to 1.0
5. Update scorer with new weights

**Example:**
```
Item scores: CLIP=0.85, DINO=0.70
User clicks 👍 Like
→ CLIP performed better
→ New weights: α=0.65, β=0.35 (CLIP increased)
```

### Multi-Reference Search

When multiple references are uploaded:
1. Each reference is encoded separately
2. Embeddings are averaged: `mean(clip_embs)`, `mean(dino_embs)`
3. Single query vector sent to vector store
4. Results represent items similar to **all** references

### Pagination

Results are paginated (12 items per page) to:
- Improve loading performance
- Reduce visual clutter
- Enable browsing through large result sets

Use the page selector dropdown to navigate.

## 🐛 Troubleshooting

### Issue: "No items in database!"

**Solution**: Run the embedding pipeline:
```bash
python -m src.core.embedding_pipeline --mode all
```

### Issue: Slow search performance

**Causes & Solutions:**
- **Large database**: Consider reducing `top_k` in search
- **GPU not detected**: Check device setting in config
- **Models not cached**: First search is slower (models loading)

### Issue: Upload fails

**Causes & Solutions:**
- **Invalid format**: Only JPG, JPEG, PNG supported
- **Corrupted file**: Re-download or try different image
- **Size too large**: Resize image before upload

### Issue: No results after filtering

**Solution**: Adjust filter criteria:
- Increase price range
- Remove category filters
- Lower similarity threshold

### Issue: Weights not updating

**Solution:**
- Refresh the page (weights persist in session state)
- Use "Reset to Default" button in sidebar
- Check browser console for errors

## 📊 Performance Tips

### For Faster Searches
1. **Use GPU**: Set `device: "cuda"` in config
2. **Reduce top_k**: Fetch fewer results (e.g., 20 instead of 50)
3. **Cache models**: Avoid restarting app (models cached in memory)

### For Better Results
1. **Upload multiple references**: Improves query representation
2. **Use high-quality images**: Clear, well-lit product photos
3. **Provide feedback**: Helps tune weights to your preferences
4. **Adjust filters**: Narrow down by price/category

## 🔐 Privacy & Data

- **No data leaves your machine**: All processing is local
- **Session state**: Cleared when browser tab closes
- **Feedback history**: Stored only in current session
- **Uploaded images**: Not saved to disk (in-memory only)

## 🛠️ Development

### File Structure
```
src/ui/
├── __init__.py
├── app.py                  # Main Streamlit app
├── state_manager.py        # Session state management
├── utils.py                # Helper functions
├── styles.py               # Custom CSS
└── components/
    ├── __init__.py
    ├── feedback_buttons.py
    ├── filters.py
    ├── item_card.py
    ├── stats_panel.py
    └── weight_display.py
```

### Adding New Features

**New Filter:**
1. Add field to `FilterSettings` in `state_manager.py`
2. Add UI control in `components/filters.py`
3. Update `apply_filters()` in `utils.py`

**New Component:**
1. Create file in `components/`
2. Implement `render_<component_name>()` function
3. Export in `components/__init__.py`
4. Import and use in `app.py`

## 📝 Keyboard Shortcuts

- **Ctrl/Cmd + R**: Refresh app
- **Ctrl/Cmd + K**: Focus search bar
- **Esc**: Close popups/modals

## 🆘 Support

For issues or questions:
1. Check this README
2. Review logs in terminal
3. Check `logs/` directory for detailed logs
4. Verify configuration in `config/config.yaml`

## 🚀 Next Steps

After launching the UI:
1. **Test with references**: Upload sample images
2. **Provide feedback**: Use 👍/👎 to tune weights
3. **Explore filters**: Try different combinations
4. **Monitor stats**: Track model performance
5. **Adjust weights**: Find optimal balance for your use case

---

**Enjoy finding your perfect fashion match!** 👗✨

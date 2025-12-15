# Vinted Scraper Module

Complete web scraping solution for Vinted fashion marketplace with Playwright automation.

## Features

- ✅ Async Playwright-based scraping (fast, non-blocking)
- ✅ Multi-strategy parsing (JSON-LD, embedded JSON, CSS selectors)
- ✅ Rate limiting with random delays
- ✅ User-agent rotation
- ✅ Exponential backoff retry logic
- ✅ Async image downloading with progress bars
- ✅ JSON storage with batch organization
- ✅ Type-safe Pydantic models
- ✅ Comprehensive error handling and logging
- ✅ CSV export support
- ✅ CLI interface

## Installation

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
playwright install chromium
```

## Configuration

Create `config/config.yaml` from `config/config.example.yaml`:

```yaml
scraper:
  base_url: "https://www.vinted.fr"
  max_pages: 10
  delay_range: [1.0, 3.0]
  timeout: 30
  headless: true
  user_agents:
    - "Mozilla/5.0 (Windows NT 10.0; Win64; x64)..."
```

## Usage

### Command-Line Interface

Basic usage:
```bash
python -m src.scraper.cli --category "robes"
```

Advanced options:
```bash
python -m src.scraper.cli \
  --category "chemises" \
  --max-pages 10 \
  --output-dir data/custom \
  --log-level DEBUG \
  --export-csv
```

Skip image downloads:
```bash
python -m src.scraper.cli --category "jeans" --no-download-images
```

### Programmatic Usage

```python
import asyncio
from pathlib import Path
from src.scraper import VintedScraper, save_batch

async def scrape_example():
    # Initialize scraper
    scraper = VintedScraper()
    
    # Scrape category
    batch = await scraper.scrape_category(
        category="robes",
        max_pages=5
    )
    
    # Download images
    output_dir = Path("data/scraped") / batch.batch_id
    await scraper.download_images(batch.items, output_dir)
    
    # Save data
    save_batch(batch, Path("data/scraped"))
    
    print(f"Scraped {batch.total_items} items")

asyncio.run(scrape_example())
```

### Loading Scraped Data

```python
from pathlib import Path
from src.scraper import load_batch, list_batches

# List all batches
output_dir = Path("data/scraped")
batches = list_batches(output_dir)
print(f"Found {len(batches)} batches")

# Load specific batch
batch = load_batch(output_dir / batches[0])
print(f"Batch: {batch.batch_id}")
print(f"Items: {batch.total_items}")

# Access items
for item in batch.items:
    print(f"{item.title} - {item.price}€")
    print(f"  Images: {len(item.image_urls)}")
```

## Data Structure

### Output Directory Layout

```
data/scraped/
└── batch_20250115_143022_123456/
    ├── batch.json           # Complete batch metadata
    ├── items.json           # All items (quick access)
    ├── items/               # Individual item files
    │   ├── 1234567.json
    │   └── 1234568.json
    └── images/              # Downloaded images
        ├── 1234567/
        │   ├── 0.jpg
        │   └── 1.jpg
        └── 1234568/
            └── 0.jpg
```

### Item Model (VintedItem)

```python
{
    "item_id": "1234567",
    "title": "Robe élégante",
    "price": 25.50,
    "currency": "EUR",
    "description": "Belle robe en excellent état...",
    "brand": "Zara",
    "size": "M",
    "condition": "Très bon état",
    "image_urls": ["https://..."],
    "url": "https://www.vinted.fr/items/1234567",
    "seller_id": "9876543",
    "category": "Robes",
    "scraped_at": "2025-01-15T14:30:22",
    "local_image_paths": ["data/scraped/.../images/1234567/0.jpg"]
}
```

## Parsing Strategies

The scraper uses multiple fallback strategies for robustness:

1. **JSON-LD Extraction** (Primary)
   - Extracts structured data from `<script type="application/ld+json">`
   - Most reliable, standardized format

2. **Embedded JSON** (Secondary)
   - Searches for JavaScript variables: `window.__INITIAL_STATE__`
   - Handles dynamic React/Vue apps

3. **CSS Selectors** (Fallback)
   - Uses BeautifulSoup with multiple selector fallbacks
   - Works when JavaScript-based methods fail

## Rate Limiting

Configured delays between requests:
- Random delay from `delay_range` (e.g., 1-3 seconds)
- Exponential backoff on retries
- User-agent rotation per context

## Error Handling

- **NavigationError**: Page failed to load (with retry)
- **ParsingError**: Failed to extract data (skips item)
- **DownloadError**: Image download failed (logs warning)

All errors are logged with context for debugging.

## Performance

- Async operations for non-blocking I/O
- Progress bars with `tqdm`
- Configurable batch sizes
- Deduplication of URLs

## Logging

Logs are saved to `logs/fashionmatch.log` with rotation.

Log levels:
- **INFO**: Scraping progress, items found
- **DEBUG**: Delays, user-agents, retries
- **WARNING**: Missing fields, skipped items
- **ERROR**: Critical failures

## Future Enhancements

- [ ] Proxy support for IP rotation
- [ ] Session persistence for large scrapes
- [ ] Distributed scraping with task queue
- [ ] Incremental updates (delta scraping)
- [ ] Price history tracking
- [ ] Seller reputation extraction

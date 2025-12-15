# FashionMatch Setup Guide

Complete installation and configuration guide for FashionMatch.

## Prerequisites

### System Requirements
- **Python**: 3.10, 3.11, or 3.12 (ChromaDB requirement)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 5GB for models and data
- **GPU**: Optional but recommended (CUDA-compatible NVIDIA GPU)
- **OS**: Windows, Linux, or macOS

### Required Software
- Python 3.10-3.12
- Git
- pip (Python package manager)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/fashionmatch.git
cd fashionmatch
```

### 2. Create Virtual Environment

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows PowerShell:**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**Windows Command Prompt:**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Playwright Browsers

Playwright is required for web scraping:

```bash
playwright install chromium
```

### 5. GPU Setup (Optional but Recommended)

If you have an NVIDIA GPU with CUDA support:

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify GPU detection:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 6. Configuration

Create your configuration file from the example:

```bash
cp config/config.example.yaml config/config.yaml
```

Edit `config/config.yaml` to customize settings:

```yaml
models:
  clip_model: "openai/clip-vit-base-patch32"  # or larger models
  dino_model: "dinov2_vits14"  # or vitb14, vitl14, vitg14
  fusion_weights:
    clip: 0.6
    dino: 0.4
  device: "auto"  # or "cuda" / "cpu"

database:
  persist_directory: "./data/chroma"
  collection_name: "fashion_items"
  batch_size: 32
  distance_metric: "cosine"

scraper:
  base_url: "https://www.vinted.fr"
  delay_range: [1.0, 3.0]
  max_retries: 3
  timeout: 30
  headless: true

data_dir: "./data"
references_dir: "./data/references"
scraped_dir: "./data/scraped"
log_level: "INFO"
```

### 7. Verify Installation

Run verification checks:

```bash
# Check Python version
python --version

# Check PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Check ChromaDB
python -c "import chromadb; print('ChromaDB OK')"

# Check Transformers
python -c "from transformers import CLIPProcessor; print('Transformers OK')"

# Check Playwright
python -c "from playwright.sync_api import sync_playwright; print('Playwright OK')"
```

All checks should complete without errors.

## Development Setup (Optional)

For contributors and developers:

### Install Test Dependencies

Uncomment test dependencies in `requirements.txt`:

```txt
# Development & Testing
# ============================================================================
pytest>=7.4.0,<8.0.0
pytest-cov>=4.1.0,<5.0.0
pytest-asyncio>=0.21.0,<1.0.0
pytest-mock>=3.11.0,<4.0.0
```

Then install:

```bash
pip install -r requirements.txt
```

### Install Jupyter (Optional)

For experimentation notebooks:

```bash
pip install jupyter matplotlib seaborn scikit-learn
```

## Troubleshooting

### Python Version Issues

**Problem**: ChromaDB requires Python 3.10-3.12

**Solution**: Install correct Python version
```bash
# Check version
python --version

# On Ubuntu/Debian
sudo apt install python3.11 python3.11-venv

# On macOS with Homebrew
brew install python@3.11

# On Windows, download from python.org
```

### ChromaDB Errors

**Problem**: `ImportError: cannot import name 'Client' from 'chromadb'`

**Solution**: Reinstall ChromaDB
```bash
pip uninstall chromadb -y
pip install chromadb>=0.4.0
```

### Playwright Installation Issues

**Problem**: `playwright._impl._api_types.Error: Executable doesn't exist`

**Solution**: Reinstall Playwright browsers
```bash
playwright install --force chromium
```

### CUDA Not Detected

**Problem**: PyTorch not using GPU

**Solutions**:
1. Verify NVIDIA drivers: `nvidia-smi`
2. Check CUDA toolkit: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version
4. Set device explicitly in config: `device: "cuda"`

### Out of Memory Errors

**Problem**: GPU out of memory during encoding

**Solutions**:
1. Reduce batch size in config: `batch_size: 16`
2. Use smaller models: `clip_model: "openai/clip-vit-base-patch32"`
3. Use CPU mode: `device: "cpu"`
4. Close other GPU applications

### Permission Errors (Linux/macOS)

**Problem**: Permission denied when creating directories

**Solution**: Create data directories manually
```bash
mkdir -p data/{references,scraped,chroma}
chmod -R 755 data/
```

### Windows Path Issues

**Problem**: Long path errors on Windows

**Solution**: Enable long paths
```powershell
# Run as Administrator
New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force
```

## Platform-Specific Notes

### Windows

- Use PowerShell (not CMD) for better experience
- Antivirus may slow down Playwright - add exclusion for project folder
- Windows Defender may block Playwright downloads

### Linux

- Ensure system packages for Playwright:
  ```bash
  sudo apt-get install libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libasound2
  ```

### macOS

- On Apple Silicon (M1/M2), use native ARM Python
- Rosetta may be needed for some dependencies
- Grant Terminal Full Disk Access in Privacy settings

## Next Steps

After successful installation:

1. **Run tests** (if installed):
   ```bash
   pytest tests/ -v
   ```

2. **Scrape some data**:
   ```bash
   python -m src.scraper.cli --category "chemises" --pages 5
   ```

3. **Process embeddings**:
   ```bash
   python -m src.core.embedding_pipeline --mode all
   ```

4. **Launch UI**:
   ```bash
   streamlit run src/ui/app.py
   ```

See [USAGE.md](USAGE.md) for detailed usage instructions.

## Getting Help

- **Issues**: Check GitHub Issues
- **Logs**: Review `logs/` directory for detailed error messages
- **Documentation**: Read [ARCHITECTURE.md](ARCHITECTURE.md) and [API.md](API.md)
- **Community**: Join discussions in GitHub Discussions

## System Diagnostics Script

Create `check_system.py` to verify your setup:

```python
#!/usr/bin/env python3
"""System diagnostics for FashionMatch."""

import sys

print("="  * 60)
print("FashionMatch System Diagnostics")
print("=" * 60)

# Python version
print(f"\nPython: {sys.version}")
assert sys.version_info >= (3, 10), "Python 3.10+ required"
assert sys.version_info < (3, 13), "Python 3.12 or lower required"
print("✓ Python version OK")

# PyTorch
try:
    import torch
    print(f"\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("✓ PyTorch OK")
except Exception as e:
    print(f"✗ PyTorch error: {e}")

# ChromaDB
try:
    import chromadb
    print(f"\n✓ ChromaDB OK")
except Exception as e:
    print(f"✗ ChromaDB error: {e}")

# Transformers
try:
    from transformers import CLIPProcessor
    print("✓ Transformers OK")
except Exception as e:
    print(f"✗ Transformers error: {e}")

# Playwright
try:
    from playwright.sync_api import sync_playwright
    print("✓ Playwright OK")
except Exception as e:
    print(f"✗ Playwright error: {e}")

# PIL
try:
    from PIL import Image
    print("✓ PIL OK")
except Exception as e:
    print(f"✗ PIL error: {e}")

# NumPy
try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} OK")
except Exception as e:
    print(f"✗ NumPy error: {e}")

print("\n" + "=" * 60)
print("Diagnostics complete!")
print("=" * 60)
```

Run with:
```bash
python check_system.py
```

#!/usr/bin/env python
"""
FashionMatch AI Demo Script

Validates that the AI encoders are working correctly by:
1. Loading CLIP and DINO models
2. Downloading a test image
3. Generating embeddings
4. Displaying results

Usage:
    python scripts/demo_ai.py
    
    # Or with a custom image URL
    python scripts/demo_ai.py --url "https://example.com/image.jpg"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def print_header(text: str) -> None:
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_status(name: str, status: str, ok: bool = True) -> None:
    """Print a status line with checkmark or cross."""
    icon = "✅" if ok else "❌"
    print(f"  {icon} {name}: {status}")


def main():
    """Run the AI demo."""
    parser = argparse.ArgumentParser(description="FashionMatch AI Demo")
    parser.add_argument(
        "--url",
        type=str,
        default="https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=400",
        help="URL of test image (default: fashion image from Unsplash)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu/mps). Default: from config",
    )
    args = parser.parse_args()

    print_header("FashionMatch AI Demo")
    print(f"\n  Test image URL: {args.url[:50]}...")
    
    # =========================================
    # Step 1: Initialize Configuration
    # =========================================
    print_header("Step 1: Configuration")
    
    try:
        from src.utils.config import get_settings
        from src.utils.logger import configure_logging
        
        configure_logging()
        settings = get_settings()
        
        print_status("Config loaded", f"v{settings.app.version}")
        print_status("Device (config)", settings.models.device)
        print_status("CLIP model", settings.models.clip.model_name)
        print_status("DINO model", settings.models.dino.model_name)
    except Exception as e:
        print_status("Configuration", f"FAILED: {e}", ok=False)
        return 1

    # =========================================
    # Step 2: Download Test Image
    # =========================================
    print_header("Step 2: Download Test Image")
    
    try:
        from src.utils.image_utils import load_image
        
        start = time.time()
        test_image = load_image(args.url)
        elapsed = time.time() - start
        
        print_status("Image downloaded", f"{test_image.size} in {elapsed:.2f}s")
        print_status("Image mode", test_image.mode)
    except Exception as e:
        print_status("Image download", f"FAILED: {e}", ok=False)
        return 1

    # =========================================
    # Step 3: Initialize CLIP Encoder
    # =========================================
    print_header("Step 3: CLIP Encoder")
    
    try:
        from src.core.encoders import CLIPEncoder
        
        clip_encoder = CLIPEncoder(device=args.device)
        print_status("CLIPEncoder created", f"device={clip_encoder.device}")
        
        start = time.time()
        clip_encoder.load_model()
        elapsed = time.time() - start
        
        print_status("CLIP Model Loaded", f"OK ({elapsed:.2f}s)")
        print_status("Embedding dim", str(clip_encoder.embedding_dim))
    except Exception as e:
        print_status("CLIP Encoder", f"FAILED: {e}", ok=False)
        return 1

    # =========================================
    # Step 4: Initialize DINO Encoder
    # =========================================
    print_header("Step 4: DINO Encoder")
    
    try:
        from src.core.encoders import DINOEncoder
        
        dino_encoder = DINOEncoder(device=args.device)
        print_status("DINOEncoder created", f"device={dino_encoder.device}")
        
        start = time.time()
        dino_encoder.load_model()
        elapsed = time.time() - start
        
        print_status("DINO Model Loaded", f"OK ({elapsed:.2f}s)")
        print_status("Embedding dim", str(dino_encoder.embedding_dim))
        print_status("Pooling strategy", dino_encoder.pooling)
    except Exception as e:
        print_status("DINO Encoder", f"FAILED: {e}", ok=False)
        return 1

    # =========================================
    # Step 5: Generate Embeddings
    # =========================================
    print_header("Step 5: Generate Embeddings")
    
    try:
        # CLIP embedding
        start = time.time()
        clip_embedding = clip_encoder.encode(test_image)
        clip_time = time.time() - start
        
        print(f"\n  📊 CLIP Embedding:")
        print(f"     Shape: {clip_embedding.shape}")
        print(f"     Dtype: {clip_embedding.dtype}")
        print(f"     Time:  {clip_time*1000:.1f}ms")
        print(f"     First 5 values: {clip_embedding[:5]}")
        
        # DINO embedding
        start = time.time()
        dino_embedding = dino_encoder.encode(test_image)
        dino_time = time.time() - start
        
        print(f"\n  📊 DINO Embedding:")
        print(f"     Shape: {dino_embedding.shape}")
        print(f"     Dtype: {dino_embedding.dtype}")
        print(f"     Time:  {dino_time*1000:.1f}ms")
        print(f"     First 5 values: {dino_embedding[:5]}")
        
    except Exception as e:
        print_status("Embedding generation", f"FAILED: {e}", ok=False)
        return 1

    # =========================================
    # Step 6: Test HybridEncoder
    # =========================================
    print_header("Step 6: Hybrid Encoder")
    
    try:
        from src.core.encoders import HybridEncoder
        
        # Create hybrid with existing encoders
        hybrid = HybridEncoder(
            clip_encoder=clip_encoder,
            dino_encoder=dino_encoder,
        )
        
        start = time.time()
        result = hybrid.encode_all(test_image)
        hybrid_time = time.time() - start
        
        print_status("HybridEncoder", f"OK ({hybrid_time*1000:.1f}ms)")
        print_status("CLIP shape", str(result.clip_embedding.shape))
        print_status("DINO shape", str(result.dino_embedding.shape))
        print_status("Combined dim", str(result.total_dim))
        
    except Exception as e:
        print_status("Hybrid Encoder", f"FAILED: {e}", ok=False)
        return 1

    # =========================================
    # Step 7: Test Similarity & Scorer
    # =========================================
    print_header("Step 7: Similarity & Scoring")
    
    try:
        import numpy as np
        from src.core.scoring import cosine_similarity, WeightedScorer
        
        # Self-similarity should be 1.0
        clip_self_sim = cosine_similarity(clip_embedding, clip_embedding)
        dino_self_sim = cosine_similarity(dino_embedding, dino_embedding)
        
        print_status("CLIP self-similarity", f"{clip_self_sim:.4f}")
        print_status("DINO self-similarity", f"{dino_self_sim:.4f}")
        
        # Test weighted scorer
        scorer = WeightedScorer()
        combined_score = scorer.compute_score(clip_self_sim, dino_self_sim)
        
        print_status("WeightedScorer", f"weights={scorer.weights.to_tuple()}")
        print_status("Combined score", f"{combined_score:.4f}")
        
    except Exception as e:
        print_status("Similarity/Scoring", f"FAILED: {e}", ok=False)
        return 1

    # =========================================
    # Summary
    # =========================================
    print_header("✅ All Tests Passed!")
    
    print(f"""
  Summary:
  ─────────────────────────────────────────
  • CLIP:   {clip_encoder.model_name} → {clip_embedding.shape}
  • DINO:   {dino_encoder.model_name} → {dino_embedding.shape}
  • Hybrid: {result.total_dim} dimensions total
  • Device: {clip_encoder.device}
  
  Your FashionMatch AI core is ready! 🚀
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())

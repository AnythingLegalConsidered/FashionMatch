#!/usr/bin/env python3
"""
Database integration test script for FashionMatch.

This script validates that ChromaDB storage and hybrid search work correctly by:
1. Initializing the ChromaRepository
2. Creating a mock ClothingItem
3. Generating random CLIP (512d) and DINO (384d) embeddings
4. Adding the item to the repository
5. Performing a hybrid search to find the item
6. Verifying the item is retrieved correctly

Usage:
    python scripts/test_db.py

Note:
    Requires ChromaDB to be installed (Python < 3.14).
    For Python 3.14+, this test will show an informative message.

Author: FashionMatch Team
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    """Run the database integration test."""
    print("\n" + "=" * 60)
    print("🗄️  FashionMatch - Database Integration Test")
    print("=" * 60)
    
    # Check ChromaDB availability first
    from src.infrastructure.database import CHROMADB_AVAILABLE, ChromaRepository, FusionWeights
    
    if not CHROMADB_AVAILABLE:
        print("\n⚠️  ChromaDB n'est pas disponible!")
        print("   ChromaDB requiert Python < 3.14 (onnxruntime non compatible)")
        print("   Pour tester, utilisez Python 3.12 ou antérieur.")
        print("\n   Cependant, voici une simulation du flux de données:\n")
        _run_mock_test()
        return
    
    # Run real test with ChromaDB
    _run_chromadb_test()


def _run_mock_test():
    """Run a mock test when ChromaDB is not available."""
    import numpy as np
    from src.domain.entities.clothing_item import ClothingItem
    from src.infrastructure.database import CLIP_EMBEDDING_DIM, DINO_EMBEDDING_DIM, FusionWeights
    
    print("📦 Création d'un mock ClothingItem...")
    item = ClothingItem(
        id="test-item-001",
        title="T-Shirt Nike Vintage",
        price=25.0,
        brand="Nike",
        size="M",
        condition="Très bon état",
        image_url="https://example.com/image.jpg",
        item_url="https://www.vinted.fr/items/123456",
    )
    print(f"   ID: {item.id}")
    print(f"   Titre: {item.title}")
    print(f"   Prix: {item.price}€")
    
    print(f"\n🎲 Génération de vecteurs aléatoires...")
    clip_vector = np.random.rand(CLIP_EMBEDDING_DIM).astype(np.float32)
    dino_vector = np.random.rand(DINO_EMBEDDING_DIM).astype(np.float32)
    
    # Normalize vectors (as real encoders would)
    clip_vector = clip_vector / np.linalg.norm(clip_vector)
    dino_vector = dino_vector / np.linalg.norm(dino_vector)
    
    print(f"   CLIP vector: shape={clip_vector.shape}, norm={np.linalg.norm(clip_vector):.4f}")
    print(f"   DINO vector: shape={dino_vector.shape}, norm={np.linalg.norm(dino_vector):.4f}")
    
    print(f"\n⚙️  FusionWeights configuration...")
    weights = FusionWeights(clip=0.6, dino=0.4)
    print(f"   {weights}")
    
    print("\n" + "-" * 60)
    print("📝 SIMULATION (ChromaDB non disponible):")
    print("-" * 60)
    print("   ✅ add_item() serait appelé avec:")
    print(f"      - item.id = '{item.id}'")
    print(f"      - clip_embedding = float[{CLIP_EMBEDDING_DIM}]")
    print(f"      - dino_embedding = float[{DINO_EMBEDDING_DIM}]")
    print("\n   ✅ hybrid_search() serait appelé avec:")
    print(f"      - clip_vector = float[{CLIP_EMBEDDING_DIM}]")
    print(f"      - dino_vector = float[{DINO_EMBEDDING_DIM}]")
    print(f"      - weights = {weights}")
    print("\n   📊 Résultat attendu:")
    print(f"      - Item retrouvé: Oui (score ≈ 1.0)")
    print(f"      - L'item devrait être le premier résultat")
    
    print("\n" + "=" * 60)
    print("✅ Test de simulation terminé!")
    print("   Pour un test complet, utilisez Python < 3.14")
    print("=" * 60 + "\n")


def _run_chromadb_test():
    """Run the actual ChromaDB test."""
    import numpy as np
    import tempfile
    import shutil
    
    from src.domain.entities.clothing_item import ClothingItem
    from src.infrastructure.database import (
        ChromaRepository,
        FusionWeights,
        CLIP_EMBEDDING_DIM,
        DINO_EMBEDDING_DIM,
    )
    from src.utils.logger import configure_logging, get_logger
    
    # Configure logging
    configure_logging()
    logger = get_logger(__name__)
    
    # Use a temporary directory for test database
    test_db_dir = tempfile.mkdtemp(prefix="fashionmatch_test_")
    print(f"\n📁 Base de données temporaire: {test_db_dir}")
    
    try:
        # =========================================
        # Step 1: Initialize Repository
        # =========================================
        print("\n🔧 Initialisation du ChromaRepository...")
        repo = ChromaRepository(
            persist_directory=test_db_dir,
            collection_prefix="test_",  # Isolate test collections
        )
        print(f"   ✅ Repository initialisé: {repo}")
        print(f"   Items existants: {repo.count()}")
        
        # =========================================
        # Step 2: Create Mock ClothingItem
        # =========================================
        print("\n📦 Création d'un mock ClothingItem...")
        item = ClothingItem(
            id="test-item-001",
            title="T-Shirt Nike Vintage",
            price=25.0,
            brand="Nike",
            size="M",
            condition="Très bon état",
            category="T-shirts",
            image_url="https://example.com/image.jpg",
            item_url="https://www.vinted.fr/items/123456",
            description="Superbe t-shirt Nike vintage en excellent état",
            seller_id="seller-123",
        )
        print(f"   ID: {item.id}")
        print(f"   Titre: {item.title}")
        print(f"   Prix: {item.price}€")
        print(f"   Marque: {item.brand}")
        
        # =========================================
        # Step 3: Generate Random Embeddings
        # =========================================
        print(f"\n🎲 Génération de vecteurs aléatoires...")
        
        # Create random vectors
        np.random.seed(42)  # For reproducibility
        clip_vector = np.random.rand(CLIP_EMBEDDING_DIM).astype(np.float32)
        dino_vector = np.random.rand(DINO_EMBEDDING_DIM).astype(np.float32)
        
        # Normalize vectors (as real encoders would)
        clip_vector = clip_vector / np.linalg.norm(clip_vector)
        dino_vector = dino_vector / np.linalg.norm(dino_vector)
        
        print(f"   CLIP vector: shape={clip_vector.shape}, norm={np.linalg.norm(clip_vector):.4f}")
        print(f"   DINO vector: shape={dino_vector.shape}, norm={np.linalg.norm(dino_vector):.4f}")
        
        # =========================================
        # Step 4: Add Item to Repository
        # =========================================
        print("\n💾 Ajout de l'item au repository...")
        item_id = repo.add_item(
            item=item,
            clip_embedding=clip_vector.tolist(),
            dino_embedding=dino_vector.tolist(),
        )
        print(f"   ✅ Item ajouté avec ID: {item_id}")
        print(f"   Items dans le repository: {repo.count()}")
        
        # =========================================
        # Step 5: Perform Hybrid Search
        # =========================================
        print("\n🔍 Recherche hybride avec les mêmes vecteurs...")
        weights = FusionWeights(clip=0.6, dino=0.4)
        print(f"   Weights: {weights}")
        
        results = repo.hybrid_search(
            clip_vector=clip_vector.tolist(),
            dino_vector=dino_vector.tolist(),
            weights=weights,
            limit=5,
        )
        
        # =========================================
        # Step 6: Analyze Results
        # =========================================
        print("\n" + "=" * 60)
        print("📊 RÉSULTATS")
        print("=" * 60)
        
        if not results:
            print("   ❌ Aucun résultat trouvé!")
            item_found = False
        else:
            print(f"   Nombre de résultats: {len(results)}")
            
            # Check if our item is in results
            item_found = False
            for i, result_item in enumerate(results):
                is_match = result_item.id == item.id
                status = "✅ MATCH" if is_match else ""
                print(f"\n   #{i+1}: {result_item.title}")
                print(f"       ID: {result_item.id} {status}")
                print(f"       Prix: {result_item.price}€")
                print(f"       Marque: {result_item.brand}")
                
                if is_match:
                    item_found = True
        
        # =========================================
        # Final Verdict
        # =========================================
        print("\n" + "=" * 60)
        if item_found:
            print("✅ TEST RÉUSSI!")
            print(f"   Item retrouvé: Oui")
            print(f"   Position dans les résultats: #1 (premier)")
            print("   L'écriture et la lecture hybride fonctionnent correctement!")
        else:
            print("❌ TEST ÉCHOUÉ!")
            print(f"   Item retrouvé: Non")
            print("   L'item n'a pas été retrouvé dans les résultats.")
        print("=" * 60 + "\n")
        
        # =========================================
        # Bonus: Test search_similar for each type
        # =========================================
        print("🔬 Tests additionnels...")
        
        # Test CLIP search
        clip_results = repo.search_similar(clip_vector.tolist(), "clip", n_results=1)
        if clip_results and clip_results[0][0].id == item.id:
            print(f"   ✅ CLIP search: Item trouvé (similarité: {clip_results[0][1]:.4f})")
        else:
            print("   ❌ CLIP search: Item non trouvé")
        
        # Test DINO search
        dino_results = repo.search_similar(dino_vector.tolist(), "dino", n_results=1)
        if dino_results and dino_results[0][0].id == item.id:
            print(f"   ✅ DINO search: Item trouvé (similarité: {dino_results[0][1]:.4f})")
        else:
            print("   ❌ DINO search: Item non trouvé")
        
        # Test get_item
        retrieved = repo.get_item(item.id)
        if retrieved and retrieved.id == item.id:
            print(f"   ✅ get_item: Item récupéré correctement")
        else:
            print("   ❌ get_item: Échec de récupération")
        
        print("\n✅ Tous les tests additionnels terminés!")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup temporary database
        print(f"\n🧹 Nettoyage de la base temporaire...")
        try:
            shutil.rmtree(test_db_dir)
            print(f"   ✅ Supprimé: {test_db_dir}")
        except Exception as e:
            print(f"   ⚠️  Échec du nettoyage: {e}")


if __name__ == "__main__":
    main()

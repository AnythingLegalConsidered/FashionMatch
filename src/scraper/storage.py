"""Data persistence utilities for scraped content.

This module handles saving and loading scraped batches with JSON serialization.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.utils import get_logger

from .models import ScrapedBatch, VintedItem
from .utils import create_batch_directory

logger = get_logger(__name__)


def save_batch(batch: ScrapedBatch, output_dir: Path) -> Path:
    """Save a scraped batch to disk with JSON files.
    
    Creates directory structure:
    - {output_dir}/{batch_id}/batch.json (full metadata)
    - {output_dir}/{batch_id}/items.json (all items list)
    - {output_dir}/{batch_id}/items/{item_id}.json (individual items)
    
    Args:
        batch: ScrapedBatch to save
        output_dir: Base output directory
        
    Returns:
        Path to batch directory
    """
    # Create batch directory
    batch_dir = create_batch_directory(output_dir, batch.batch_id)
    
    logger.info(f"Saving batch {batch.batch_id} to {batch_dir}")
    
    # Save full batch metadata
    batch_file = batch_dir / 'batch.json'
    with open(batch_file, 'w', encoding='utf-8') as f:
        json.dump(batch.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
    
    # Save items list (for quick access)
    items_file = batch_dir / 'items.json'
    items_data = [item.model_dump(mode='json') for item in batch.items]
    with open(items_file, 'w', encoding='utf-8') as f:
        json.dump(items_data, f, indent=2, ensure_ascii=False, default=str)
    
    # Save individual item files
    items_dir = batch_dir / 'items'
    for item in batch.items:
        item_file = items_dir / f"{item.item_id}.json"
        with open(item_file, 'w', encoding='utf-8') as f:
            json.dump(item.model_dump(mode='json'), f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Saved {len(batch.items)} items to {batch_dir}")
    return batch_dir


def load_batch(batch_dir: Path) -> ScrapedBatch:
    """Load a scraped batch from disk.
    
    Args:
        batch_dir: Path to batch directory
        
    Returns:
        Reconstructed ScrapedBatch instance
        
    Raises:
        FileNotFoundError: If batch.json doesn't exist
        ValueError: If batch data is invalid
    """
    batch_file = batch_dir / 'batch.json'
    
    if not batch_file.exists():
        raise FileNotFoundError(f"Batch file not found: {batch_file}")
    
    logger.debug(f"Loading batch from {batch_dir}")
    
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    # Validate and reconstruct with Pydantic
    batch = ScrapedBatch.model_validate(batch_data)
    
    logger.info(f"Loaded batch {batch.batch_id} with {len(batch.items)} items")
    return batch


def load_item(item_file: Path) -> VintedItem:
    """Load a single item from JSON file.
    
    Args:
        item_file: Path to item JSON file
        
    Returns:
        VintedItem instance
        
    Raises:
        FileNotFoundError: If item file doesn't exist
        ValueError: If item data is invalid
    """
    if not item_file.exists():
        raise FileNotFoundError(f"Item file not found: {item_file}")
    
    with open(item_file, 'r', encoding='utf-8') as f:
        item_data = json.load(f)
    
    return VintedItem.model_validate(item_data)


def list_batches(output_dir: Path) -> list[str]:
    """List all batch IDs in output directory, sorted by timestamp.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        List of batch IDs (directory names), newest first
    """
    if not output_dir.exists():
        return []
    
    # Find all directories that look like batch directories
    batch_dirs = [
        d.name for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith('batch_')
    ]
    
    # Sort by name (timestamp-based) in reverse order (newest first)
    batch_dirs.sort(reverse=True)
    
    return batch_dirs


def get_batch_stats(batch_dir: Path) -> dict:
    """Get statistics for a batch without loading all data.
    
    Args:
        batch_dir: Path to batch directory
        
    Returns:
        Dictionary with batch statistics
    """
    batch_file = batch_dir / 'batch.json'
    
    if not batch_file.exists():
        return {}
    
    with open(batch_file, 'r', encoding='utf-8') as f:
        batch_data = json.load(f)
    
    return {
        'batch_id': batch_data.get('batch_id'),
        'category': batch_data.get('category'),
        'total_items': batch_data.get('total_items'),
        'pages_scraped': batch_data.get('pages_scraped'),
        'scraped_at': batch_data.get('scraped_at'),
    }


def export_batch_to_csv(batch: ScrapedBatch, output_file: Path) -> None:
    """Export batch items to CSV format.
    
    Args:
        batch: ScrapedBatch to export
        output_file: Path to output CSV file
    """
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        if not batch.items:
            return
        
        # Get field names from first item
        fieldnames = ['item_id', 'title', 'price', 'currency', 'brand', 'size', 
                     'condition', 'url', 'seller_id', 'category', 'scraped_at']
        
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for item in batch.items:
            row = {
                'item_id': item.item_id,
                'title': item.title,
                'price': item.price,
                'currency': item.currency,
                'brand': item.brand or '',
                'size': item.size or '',
                'condition': item.condition or '',
                'url': item.url,
                'seller_id': item.seller_id or '',
                'category': item.category or '',
                'scraped_at': item.scraped_at.isoformat(),
            }
            writer.writerow(row)
    
    logger.info(f"Exported {len(batch.items)} items to {output_file}")

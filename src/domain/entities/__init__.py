# Domain Entities Package
"""
Core business entities as dataclasses.
"""

from .clothing_item import ClothingItem
from .embedding import Embedding
from .user_preference import UserPreference

__all__ = ["ClothingItem", "Embedding", "UserPreference"]

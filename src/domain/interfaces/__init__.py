# Domain Interfaces Package
"""
Abstract base classes defining contracts for infrastructure implementations.
"""

from .encoder_interface import EncoderInterface
from .repository_interface import RepositoryInterface
from .scraper_interface import ScraperInterface

__all__ = ["EncoderInterface", "RepositoryInterface", "ScraperInterface"]

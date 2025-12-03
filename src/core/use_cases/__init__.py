# Use Cases Package
"""
Application use cases (business logic).

Use cases encapsulate the business logic and orchestrate the flow
of data between domain entities, encoders, and repositories.
"""

from src.core.use_cases.add_reference import (
    AddReferenceUseCase,
    AddReferenceResult,
    ReferenceImageData,
    ClearReferencesUseCase,
    GetReferencesUseCase,
)

from src.core.use_cases.get_recommendations import (
    GetRecommendationsUseCase,
    RecommendationResult,
    RecommendationFilters,
    AddItemToReferencesUseCase,
)

__all__ = [
    # Add Reference
    "AddReferenceUseCase",
    "AddReferenceResult",
    "ReferenceImageData",
    "ClearReferencesUseCase",
    "GetReferencesUseCase",
    # Get Recommendations
    "GetRecommendationsUseCase",
    "RecommendationResult",
    "RecommendationFilters",
    "AddItemToReferencesUseCase",
]

"""Backend-owned capability adapters for bounded domain review."""

from app.review.capabilities import (
    BackendApplicabilityProvider,
    BoundReferenceUnitProvider,
    BoundReviewUnitProvider,
    BoundSelectedKnowledgeRetriever,
)

__all__ = (
    "BackendApplicabilityProvider",
    "BoundReferenceUnitProvider",
    "BoundReviewUnitProvider",
    "BoundSelectedKnowledgeRetriever",
)

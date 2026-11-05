"""Public API for provider-independent retrieval."""

from rag_core.retrieval.dense import DenseSourceRetriever
from rag_core.retrieval.errors import (
    RetrievalError,
    RetrievalIsolationError,
    RetrievalValidationError,
    UnsupportedPayloadSchemaError,
)
from rag_core.retrieval.models import (
    KnowledgeHit,
    RetrievalRequest,
    RetrievalScope,
    SourceRetrievalRequest,
)
from rag_core.retrieval.ports import (
    QueryEmbeddingProvider,
    Retriever,
    SourceRetriever,
    VectorSearch,
)
from rag_core.retrieval.service import RetrievalService

__all__ = (
    "DenseSourceRetriever",
    "KnowledgeHit",
    "QueryEmbeddingProvider",
    "RetrievalError",
    "RetrievalIsolationError",
    "RetrievalRequest",
    "RetrievalScope",
    "RetrievalService",
    "RetrievalValidationError",
    "Retriever",
    "SourceRetrievalRequest",
    "SourceRetriever",
    "UnsupportedPayloadSchemaError",
    "VectorSearch",
)

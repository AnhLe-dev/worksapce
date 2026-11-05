"""Small ports for retrieval providers and orchestration."""

from collections.abc import Sequence
from typing import Protocol

from rag_core.retrieval.models import (
    KnowledgeHit,
    RetrievalRequest,
    RetrievalScope,
    SourceRetrievalRequest,
)


class QueryEmbeddingProvider(Protocol):
    """Generate one dense query vector."""

    def embed_query(self, text: str) -> Sequence[float]:
        """Return one vector for normalized query text."""

        ...


class VectorSearch(Protocol):
    """Search one exact immutable source snapshot."""

    def search(
        self,
        *,
        query_vector: tuple[float, ...],
        scope: RetrievalScope,
        limit: int,
    ) -> tuple[KnowledgeHit, ...]:
        """Return provider-decoded hits for the requested scope."""

        ...


class SourceRetriever(Protocol):
    """Retrieve evidence from one exact source snapshot."""

    def retrieve_source(
        self,
        request: SourceRetrievalRequest,
    ) -> tuple[KnowledgeHit, ...]:
        """Return ranked and scope-validated evidence."""

        ...


class Retriever(Protocol):
    """Retrieve evidence across backend-resolved source snapshots."""

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> tuple[KnowledgeHit, ...]:
        """Return globally ranked evidence without resolving authority."""

        ...

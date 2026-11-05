"""Provider-independent dense retrieval for one source snapshot."""

from math import isfinite

from rag_core.retrieval.errors import (
    RetrievalIsolationError,
    RetrievalValidationError,
)
from rag_core.retrieval.models import (
    KnowledgeHit,
    RetrievalScope,
    SourceRetrievalRequest,
)
from rag_core.retrieval.ports import (
    QueryEmbeddingProvider,
    VectorSearch,
)


def _matches_scope(
    hit: KnowledgeHit,
    scope: RetrievalScope,
) -> bool:
    """Return whether one hit belongs to the exact requested snapshot."""

    return (
        hit.access_scope == scope.access_scope
        and hit.source_id == scope.source_id
        and hit.source_version == scope.source_version
        and hit.source_checksum_sha256
        == scope.source_checksum_sha256
        and hit.index_profile == scope.index_profile
    )


class DenseSourceRetriever:
    """Embed one query and validate vector-search results fail-closed."""

    def __init__(
        self,
        *,
        embedding_provider: QueryEmbeddingProvider,
        vector_search: VectorSearch,
        expected_dimensions: int,
    ) -> None:
        if expected_dimensions <= 0:
            raise RetrievalValidationError(
                "Expected query dimensions must be greater than zero"
            )

        self._embedding_provider = embedding_provider
        self._vector_search = vector_search
        self._expected_dimensions = expected_dimensions

    def retrieve_source(
        self,
        request: SourceRetrievalRequest,
    ) -> tuple[KnowledgeHit, ...]:
        """Retrieve, isolate, deduplicate, and deterministically rank hits."""

        raw_vector = self._embedding_provider.embed_query(
            request.query
        )

        try:
            query_vector = tuple(
                float(value)
                for value in raw_vector
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise RetrievalValidationError(
                "Embedding provider returned a non-numeric query vector"
            ) from exc

        if len(query_vector) != self._expected_dimensions:
            raise RetrievalValidationError(
                "Query vector has an unexpected dimension: "
                f"expected={self._expected_dimensions}, "
                f"actual={len(query_vector)}"
            )

        if not all(isfinite(value) for value in query_vector):
            raise RetrievalValidationError(
                "Query vector contains a non-finite value"
            )

        hits = self._vector_search.search(
            query_vector=query_vector,
            scope=request.scope,
            limit=request.limit,
        )
        unique_hits: dict[str, KnowledgeHit] = {}

        for hit in hits:
            if not _matches_scope(hit, request.scope):
                raise RetrievalIsolationError(
                    "Vector search returned evidence outside the "
                    "requested source snapshot"
                )

            previous = unique_hits.get(hit.chunk_id)

            if previous is None or hit.score > previous.score:
                unique_hits[hit.chunk_id] = hit

        ranked = sorted(
            unique_hits.values(),
            key=lambda hit: (
                -hit.score,
                hit.chunk_id,
            ),
        )

        return tuple(ranked[:request.limit])

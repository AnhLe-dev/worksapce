"""Orchestration for retrieval across authorized source snapshots."""

from rag_core.retrieval.errors import RetrievalIsolationError
from rag_core.retrieval.models import (
    KnowledgeHit,
    RetrievalRequest,
    SourceRetrievalRequest,
)
from rag_core.retrieval.ports import SourceRetriever


class RetrievalService:
    """Fan out to exact sources and merge results deterministically."""

    def __init__(
        self,
        *,
        source_retriever: SourceRetriever,
    ) -> None:
        self._source_retriever = source_retriever

    def retrieve(
        self,
        request: RetrievalRequest,
    ) -> tuple[KnowledgeHit, ...]:
        """Retrieve only from scopes already resolved by the consumer."""

        merged: dict[
            tuple[str, str, str, str, str, str],
            KnowledgeHit,
        ] = {}

        for scope in request.scopes:
            hits = self._source_retriever.retrieve_source(
                SourceRetrievalRequest(
                    query=request.query,
                    scope=scope,
                    limit=request.limit,
                )
            )

            for hit in hits:
                if (
                    hit.access_scope != scope.access_scope
                    or hit.source_id != scope.source_id
                    or hit.source_version != scope.source_version
                    or hit.source_checksum_sha256
                    != scope.source_checksum_sha256
                    or hit.index_profile != scope.index_profile
                ):
                    raise RetrievalIsolationError(
                        "Source retriever returned evidence outside "
                        "the authorized scopes"
                    )

                hit_key = (
                    hit.access_scope,
                    hit.source_id,
                    hit.source_version,
                    hit.source_checksum_sha256,
                    hit.index_profile,
                    hit.chunk_id,
                )
                previous = merged.get(hit_key)

                if previous is None or hit.score > previous.score:
                    merged[hit_key] = hit

        ranked = sorted(
            merged.values(),
            key=lambda hit: (
                -hit.score,
                hit.source_id,
                hit.source_version,
                hit.chunk_id,
            ),
        )

        return tuple(ranked[:request.limit])

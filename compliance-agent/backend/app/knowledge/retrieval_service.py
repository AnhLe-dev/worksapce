"""Retrieve only after PostgreSQL resolves authorized source snapshots."""

from rag_core.retrieval.models import (
    KnowledgeHit,
    RetrievalRequest,
    RetrievalScope,
)
from rag_core.retrieval.ports import Retriever

from app.knowledge.errors import KnowledgeIsolationError
from app.knowledge.models import (
    KnowledgeRetrievalRequest,
    KnowledgeRetrievalResult,
)
from app.knowledge.registry import KnowledgeRegistry


class KnowledgeRetrievalService:
    """Compose authority resolution with provider-neutral retrieval."""

    def __init__(
        self,
        *,
        registry: KnowledgeRegistry,
        retriever: Retriever,
    ) -> None:
        self._registry = registry
        self._retriever = retriever

    def retrieve(
        self,
        request: KnowledgeRetrievalRequest,
    ) -> KnowledgeRetrievalResult:
        resolved = self._registry.resolve(
            knowledge_base_id=request.knowledge_base_id,
            trusted_scope=request.access_scope,
        )
        hits = self._retriever.retrieve(
            RetrievalRequest(
                query=request.query,
                scopes=resolved.scopes,
                limit=request.limit,
            )
        )
        self._validate_hits(hits=hits, scopes=resolved.scopes)
        return KnowledgeRetrievalResult(
            knowledge_base_id=request.knowledge_base_id,
            snapshots=resolved.snapshots,
            scopes=resolved.scopes,
            hits=hits,
        )

    @staticmethod
    def _validate_hits(
        *,
        hits: tuple[KnowledgeHit, ...],
        scopes: tuple[RetrievalScope, ...],
    ) -> None:
        """Do not trust provider payloads after authority resolution."""

        allowed_scopes = set(scopes)
        seen_hit_keys: set[tuple[RetrievalScope, str]] = set()

        for hit in hits:
            hit_scope = RetrievalScope(
                access_scope=hit.access_scope,
                source_id=hit.source_id,
                source_version=hit.source_version,
                source_checksum_sha256=hit.source_checksum_sha256,
                index_profile=hit.index_profile,
            )

            if hit_scope not in allowed_scopes:
                raise KnowledgeIsolationError(
                    "Retriever returned a hit outside the resolved KB"
                )

            hit_key = (hit_scope, hit.chunk_id)

            if hit_key in seen_hit_keys:
                raise KnowledgeIsolationError(
                    "Retriever returned a duplicate source chunk"
                )

            seen_hit_keys.add(hit_key)

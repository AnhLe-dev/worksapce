"""Small ports for embedding and vector index implementations."""

from collections.abc import Sequence
from typing import Any, Protocol

from rag_core.indexing.models import IndexingResult


class SerializableSpan(Protocol):
    """Expose JSON-compatible provenance for index payloads."""

    def model_dump(
        self,
        *,
        mode: str = "python",
    ) -> dict[str, Any]:
        """Serialize one source span."""

        ...


class IndexableChunk(Protocol):
    """Common chunk shape accepted by the indexing core."""

    chunk_id: str
    source_id: str
    source_version: str
    source_checksum_sha256: str
    access_scope: str
    text: str
    structure_path: tuple[str, ...]
    source_spans: tuple[SerializableSpan, ...]
    page_start: int | None
    page_end: int | None


class EmbeddingProvider(Protocol):
    """Generate one dense vector for each supplied text."""

    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> Sequence[Sequence[float]]:
        """Return vectors in the same order as the input texts."""

        ...


class VectorIndex(Protocol):
    """Persist one immutable source snapshot."""

    def upsert(
        self,
        *,
        chunks: tuple[IndexableChunk, ...],
        vectors: tuple[tuple[float, ...], ...],
    ) -> IndexingResult:
        """Store chunks or reject a conflicting snapshot."""

        ...

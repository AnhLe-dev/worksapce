"""Ports implemented by generic document chunking strategies."""

from typing import Protocol

from rag_core.ingestion.models import (
    DocumentChunk,
    StructuredDocument,
)


class ChunkingStrategy(Protocol):
    """Create traceable chunks from a structured document."""

    def chunk(
        self,
        document: StructuredDocument,
    ) -> tuple[DocumentChunk, ...]:
        """Return deterministic chunks with exact source provenance."""

        ...
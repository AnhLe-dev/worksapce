"""Structure-first chunking with an explicit fixed-window fallback."""

from hashlib import sha256

from rag_core.chunking.block import BlockChunkingStrategy
from rag_core.chunking.fixed import FixedChunkingStrategy
from rag_core.ingestion.models import (
    DocumentChunk,
    StructuredDocument,
)


def _stable_id(*parts: str) -> str:
    """Create a deterministic SHA-256 chunk identifier."""

    serialized = "\x1f".join(parts)
    return sha256(serialized.encode("utf-8")).hexdigest()


class StructureAwareChunkingStrategy:
    """Keep real section boundaries and split only when required."""

    def __init__(
        self,
        *,
        max_characters: int,
        fallback_overlap_characters: int = 0,
    ) -> None:
        self._block_strategy = BlockChunkingStrategy(
            max_characters=max_characters,
        )
        self._fixed_strategy = FixedChunkingStrategy(
            max_characters=max_characters,
            overlap_characters=fallback_overlap_characters,
        )
        self._max_characters = max_characters
        self._fallback_overlap_characters = (
            fallback_overlap_characters
        )

    def chunk(
        self,
        document: StructuredDocument,
    ) -> tuple[DocumentChunk, ...]:
        """Use whole blocks per section, then fallback for oversize blocks."""

        chunks: list[DocumentChunk] = []
        source = document.source

        for section_index, section in enumerate(document.sections):
            section_document = StructuredDocument(
                source=source,
                sections=(section,),
            )

            try:
                section_chunks = self._block_strategy.chunk(
                    section_document
                )
                fallback = "block"
            except ValueError as exc:
                if "source block exceeds" not in str(exc):
                    raise

                section_chunks = self._fixed_strategy.chunk(
                    section_document
                )
                fallback = "fixed"

            for local_index, chunk in enumerate(section_chunks):
                chunk_id = _stable_id(
                    source.source_id,
                    source.source_version,
                    source.checksum_sha256,
                    "structure-aware:v1",
                    f"max:{self._max_characters}",
                    (
                        "fallback-overlap:"
                        f"{self._fallback_overlap_characters}"
                    ),
                    f"section:{section_index}",
                    f"local:{local_index}",
                    chunk.chunk_id,
                )
                chunks.append(
                    chunk.model_copy(update={
                        "chunk_id": chunk_id,
                        "metadata": {
                            **chunk.metadata,
                            "chunk_strategy": "structure_aware",
                            "chunk_strategy_version": "1",
                            "chunk_strategy_fallback": fallback,
                        },
                    })
                )

        if not chunks:
            raise ValueError(
                "Structure-aware chunking produced no document chunks"
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Structure-aware chunking produced duplicate chunk IDs"
            )

        return tuple(chunks)

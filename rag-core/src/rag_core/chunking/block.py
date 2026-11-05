"""Provenance-preserving generic block chunking."""

from hashlib import sha256

from rag_core.ingestion.models import (
    DocumentChunk,
    ExtractedBlock,
    StructuredDocument,
)


def _stable_id(*parts: str) -> str:
    """Create a deterministic SHA-256 identifier."""

    serialized = "\x1f".join(parts)
    return sha256(serialized.encode("utf-8")).hexdigest()


class BlockChunkingStrategy:
    """Group complete source blocks without weakening provenance."""

    def __init__(self, *, max_characters: int) -> None:
        if max_characters < 200:
            raise ValueError(
                "Maximum chunk characters must be at least 200"
            )

        self._max_characters = max_characters

    def _group_blocks(
        self,
        blocks: tuple[ExtractedBlock, ...],
    ) -> tuple[tuple[ExtractedBlock, ...], ...]:
        """Group whole blocks up to the configured character limit."""

        groups: list[tuple[ExtractedBlock, ...]] = []
        current: list[ExtractedBlock] = []
        current_length = 0

        for block in blocks:
            block_length = len(block.text)

            if block_length > self._max_characters:
                raise ValueError(
                    "A source block exceeds the chunk limit; "
                    "configure a specialized chunking strategy"
                )

            separator_length = 1 if current else 0
            candidate_length = (
                current_length
                + separator_length
                + block_length
            )

            if current and candidate_length > self._max_characters:
                groups.append(tuple(current))
                current = []
                current_length = 0

            if current:
                current_length += 1

            current.append(block)
            current_length += block_length

        if current:
            groups.append(tuple(current))

        return tuple(groups)

    def chunk(
        self,
        document: StructuredDocument,
    ) -> tuple[DocumentChunk, ...]:
        """Create deterministic chunks for every structured section."""

        chunks: list[DocumentChunk] = []
        source = document.source

        for section in document.sections:
            block_groups = self._group_blocks(
                section.blocks
            )
            part_count = len(block_groups)

            for part_index, blocks in enumerate(block_groups):
                structure_path = section.structure_path

                if part_count > 1:
                    structure_path = (
                        *structure_path,
                        f"part {part_index + 1} of {part_count}",
                    )

                spans = tuple(
                    block.source_span
                    for block in blocks
                )
                pages = tuple(
                    span.page_number
                    for span in spans
                    if span.page_number is not None
                )
                chunk_id = _stable_id(
                    source.source_id,
                    source.source_version,
                    source.checksum_sha256,
                    "block:v1",
                    f"max:{self._max_characters}",
                    *structure_path,
                    f"part:{part_index}",
                    "blocks:" + ",".join(
                        str(block.order_index)
                        for block in blocks
                    ),
                )

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        source_id=source.source_id,
                        source_version=source.source_version,
                        source_checksum_sha256=(
                            source.checksum_sha256
                        ),
                        document_format=(
                            source.document_format
                        ),
                        role=source.role,
                        access_scope=source.access_scope,
                        text="\n".join(
                            block.text
                            for block in blocks
                        ),
                        structure_path=structure_path,
                        source_spans=spans,
                        page_start=(
                            min(pages)
                            if pages
                            else None
                        ),
                        page_end=(
                            max(pages)
                            if pages
                            else None
                        ),
                        metadata={
                            **source.metadata,
                            "chunk_strategy": "block",
                            "chunk_strategy_version": "1",
                        },
                    )
                )

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if not chunks:
            raise ValueError(
                "Chunking strategy produced no document chunks"
            )

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Chunking strategy produced duplicate chunk IDs"
            )

        return tuple(chunks)

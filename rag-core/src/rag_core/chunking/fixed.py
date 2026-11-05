"""Fixed character windows with exact, non-fabricated provenance."""

from hashlib import sha256

from rag_core.ingestion.models import (
    DocumentChunk,
    ExtractedBlock,
    SourceSpan,
    StructuredDocument,
)


def _stable_id(*parts: str) -> str:
    """Create a deterministic SHA-256 chunk identifier."""

    serialized = "\x1f".join(parts)
    return sha256(serialized.encode("utf-8")).hexdigest()


class FixedChunkingStrategy:
    """Split individual blocks into overlapping character windows."""

    def __init__(
        self,
        *,
        max_characters: int,
        overlap_characters: int = 0,
    ) -> None:
        if max_characters < 200:
            raise ValueError(
                "Maximum chunk characters must be at least 200"
            )

        if overlap_characters < 0:
            raise ValueError(
                "Chunk overlap must not be negative"
            )

        if overlap_characters >= max_characters:
            raise ValueError(
                "Chunk overlap must be smaller than the chunk size"
            )

        self._max_characters = max_characters
        self._overlap_characters = overlap_characters

    def _windows(self, text: str) -> tuple[tuple[int, int], ...]:
        """Return deterministic half-open character ranges."""

        if len(text) <= self._max_characters:
            return ((0, len(text)),)

        step = self._max_characters - self._overlap_characters
        windows: list[tuple[int, int]] = []
        start = 0

        while start < len(text):
            end = min(start + self._max_characters, len(text))
            windows.append((start, end))

            if end == len(text):
                break

            start += step

        return tuple(windows)

    @staticmethod
    def _window_span(
        block: ExtractedBlock,
        *,
        start: int,
        end: int,
    ) -> SourceSpan:
        """Map a window to exact offsets or reject unsafe splitting."""

        span = block.source_span

        if start == 0 and end == len(block.text):
            return span

        if span.bbox is not None:
            raise ValueError(
                "Cannot split a positioned PDF block without exact "
                "sub-block geometry"
            )

        if span.char_start is None or span.char_end is None:
            raise ValueError(
                "Cannot split a block without character offsets"
            )

        return span.model_copy(update={
            "char_start": span.char_start + start,
            "char_end": span.char_start + end,
        })

    def chunk(
        self,
        document: StructuredDocument,
    ) -> tuple[DocumentChunk, ...]:
        """Create traceable windows without crossing source blocks."""

        chunks: list[DocumentChunk] = []
        source = document.source

        for section in document.sections:
            for block in section.blocks:
                windows = self._windows(block.text)
                window_count = len(windows)

                for window_index, (start, end) in enumerate(windows):
                    span = self._window_span(
                        block,
                        start=start,
                        end=end,
                    )
                    structure_path = section.structure_path

                    if window_count > 1:
                        structure_path = (
                            *structure_path,
                            (
                                f"window {window_index + 1} "
                                f"of {window_count}"
                            ),
                        )

                    chunk_id = _stable_id(
                        source.source_id,
                        source.source_version,
                        source.checksum_sha256,
                        "fixed:v1",
                        f"max:{self._max_characters}",
                        f"overlap:{self._overlap_characters}",
                        *section.structure_path,
                        f"block:{block.order_index}",
                        f"range:{start}:{end}",
                    )

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            source_id=source.source_id,
                            source_version=source.source_version,
                            source_checksum_sha256=(
                                source.checksum_sha256
                            ),
                            document_format=source.document_format,
                            role=source.role,
                            access_scope=source.access_scope,
                            text=block.text[start:end],
                            structure_path=structure_path,
                            source_spans=(span,),
                            page_start=span.page_number,
                            page_end=span.page_number,
                            metadata={
                                **source.metadata,
                                "chunk_strategy": "fixed",
                                "chunk_strategy_version": "1",
                                "chunk_max_characters": str(
                                    self._max_characters
                                ),
                                "chunk_overlap_characters": str(
                                    self._overlap_characters
                                ),
                            },
                        )
                    )

        if not chunks:
            raise ValueError(
                "Fixed chunking produced no document chunks"
            )

        chunk_ids = [chunk.chunk_id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(
                "Fixed chunking produced duplicate chunk IDs"
            )

        return tuple(chunks)

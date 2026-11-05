"""Application service for composition-based document processing."""

from __future__ import annotations

from collections.abc import Iterable

from rag_core.chunking.ports import ChunkingStrategy
from rag_core.ingestion.errors import (
    UnsupportedDocumentFormatError,
)
from rag_core.ingestion.loading import load_document_source
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentInput,
    DocumentProcessingResult,
)
from rag_core.ingestion.ports import (
    DocumentExtractor,
    StructureParser,
)


class DocumentProcessor:
    """Coordinate source loading, extraction, parsing, and chunking."""

    def __init__(
        self,
        *,
        extractors: Iterable[DocumentExtractor],
        parser: StructureParser,
        chunking_strategy: ChunkingStrategy,
    ) -> None:
        extractors_by_format: dict[
            DocumentFormat,
            DocumentExtractor,
        ] = {}

        for extractor in extractors:
            document_format = extractor.document_format

            if document_format in extractors_by_format:
                raise ValueError(
                    "Duplicate extractor for document format: "
                    f"{document_format.value}"
                )

            extractors_by_format[document_format] = extractor

        if not extractors_by_format:
            raise ValueError(
                "Document processor requires at least one extractor"
            )

        self._extractors = extractors_by_format
        self._parser = parser
        self._chunking_strategy = chunking_strategy

    def process(
        self,
        document_input: DocumentInput,
    ) -> DocumentProcessingResult:
        """Run one deterministic in-memory document pipeline."""

        source = load_document_source(
            document_input
        )
        extractor = self._extractors.get(
            source.document_format
        )

        if extractor is None:
            raise UnsupportedDocumentFormatError(
                "No extractor is configured for document format: "
                f"{source.document_format.value}"
            )

        extracted = extractor.extract(
            source,
            document_input.path.resolve(),
        )
        structured = self._parser.parse(
            extracted
        )

        if not structured.sections:
            raise ValueError(
                "Structure parser produced no sections"
            )

        chunks = self._chunking_strategy.chunk(
            structured
        )

        if not chunks:
            raise ValueError(
                "Chunking strategy produced no document chunks"
            )

        return DocumentProcessingResult(
            source=source,
            extracted=extracted,
            structured=structured,
            chunks=chunks,
        )

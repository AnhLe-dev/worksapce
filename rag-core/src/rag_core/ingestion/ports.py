"""Small strategies used by the generic document processor."""

from pathlib import Path
from typing import Protocol

from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentSource,
    ExtractedDocument,
    StructuredDocument,
)


class DocumentExtractor(Protocol):
    """Extract one physical document format."""

    @property
    def document_format(self) -> DocumentFormat:
        """Return the single format handled by this extractor."""

        ...

    def extract(
        self,
        source: DocumentSource,
        file_path: Path,
    ) -> ExtractedDocument:
        """Extract traceable blocks from a validated source."""

        ...


class StructureParser(Protocol):
    """Convert extracted blocks into a logical structure."""

    def parse(
        self,
        document: ExtractedDocument,
    ) -> StructuredDocument:
        """Return an explicit structure without losing provenance."""

        ...

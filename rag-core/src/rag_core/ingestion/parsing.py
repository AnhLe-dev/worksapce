"""Safe fallback parsing without domain-specific structure inference."""

from rag_core.ingestion.models import (
    ExtractedDocument,
    StructuredDocument,
    StructuredSection,
)


class FlatStructureParser:
    """Keep all blocks under one filename-based structure path."""

    def parse(
        self,
        document: ExtractedDocument,
    ) -> StructuredDocument:
        """Return one flat section without guessing headings or clauses."""

        return StructuredDocument(
            source=document.source,
            sections=(
                StructuredSection(
                    structure_path=(
                        document.source.filename,
                    ),
                    blocks=document.blocks,
                ),
            ),
        )

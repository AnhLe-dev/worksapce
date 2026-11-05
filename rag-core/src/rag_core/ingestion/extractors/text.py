"""UTF-8 plain-text extraction with exact character provenance."""

import re
from pathlib import Path

from rag_core.ingestion.errors import (
    EmptyDocumentError,
    InvalidTextEncodingError,
)
from rag_core.ingestion.loading import validate_source_file
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentSource,
    ExtractedBlock,
    ExtractedDocument,
    SourceSpan,
)

_PARAGRAPH_SEPARATOR = re.compile(
    r"(?:\r?\n[ \t]*){2,}"
)


class TextDocumentExtractor:
    """Extract non-empty paragraphs from one UTF-8 text file."""

    @property
    def document_format(self) -> DocumentFormat:
        """Return the format handled by this adapter."""

        return DocumentFormat.TXT

    @staticmethod
    def _append_block(
        *,
        blocks: list[ExtractedBlock],
        content: str,
        segment_start: int,
        segment_end: int,
    ) -> None:
        """Append one trimmed segment with adjusted source offsets."""

        segment = content[segment_start:segment_end]
        trimmed = segment.strip()

        if not trimmed:
            return

        leading_length = len(segment) - len(segment.lstrip())
        trailing_length = len(segment) - len(segment.rstrip())
        char_start = segment_start + leading_length
        char_end = segment_end - trailing_length

        blocks.append(
            ExtractedBlock(
                order_index=len(blocks),
                text=content[char_start:char_end],
                source_span=SourceSpan(
                    block_index=len(blocks),
                    char_start=char_start,
                    char_end=char_end,
                ),
            )
        )

    def extract(
        self,
        source: DocumentSource,
        file_path: Path,
    ) -> ExtractedDocument:
        """Decode UTF-8 and preserve paragraph offsets."""

        if source.document_format is not DocumentFormat.TXT:
            raise ValueError(
                "Text extractor requires a TXT document source"
            )

        resolved_path = validate_source_file(
            source,
            file_path,
        )
        raw_content = resolved_path.read_bytes()

        try:
            content = raw_content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise InvalidTextEncodingError(
                "TXT document must contain valid UTF-8 text"
            ) from error

        if "\x00" in content:
            raise InvalidTextEncodingError(
                "TXT document must not contain NUL characters"
            )

        blocks: list[ExtractedBlock] = []
        segment_start = 0

        for separator in _PARAGRAPH_SEPARATOR.finditer(content):
            self._append_block(
                blocks=blocks,
                content=content,
                segment_start=segment_start,
                segment_end=separator.start(),
            )
            segment_start = separator.end()

        self._append_block(
            blocks=blocks,
            content=content,
            segment_start=segment_start,
            segment_end=len(content),
        )

        if not blocks:
            raise EmptyDocumentError(
                "TXT document does not contain extractable text"
            )

        return ExtractedDocument(
            source=source,
            blocks=tuple(blocks),
        )

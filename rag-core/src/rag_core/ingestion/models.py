"""Framework-independent contracts for generic document processing."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


class DocumentFormat(StrEnum):
    """Identify a supported physical document format."""

    PDF = "pdf"
    TXT = "txt"


class DocumentRole(StrEnum):
    """Describe how a document participates in an application."""

    KNOWLEDGE = "knowledge"
    CONTRACT = "contract"


class _FrozenModel(BaseModel):
    """Reject unknown fields and mutation in core value objects."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


class DocumentInput(_FrozenModel):
    """Describe one local document entering the generic pipeline."""

    path: Path
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    access_scope: str = Field(min_length=1)
    role: DocumentRole
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator(
        "source_id",
        "source_version",
        "access_scope",
        mode="before",
    )
    @classmethod
    def normalize_identity(cls, value: object) -> object:
        """Strip trust-boundary identity strings."""

        if isinstance(value, str):
            return value.strip()

        return value


class DocumentSource(_FrozenModel):
    """Record the validated identity and content hash of a document."""

    source_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    access_scope: str = Field(min_length=1)
    role: DocumentRole
    document_format: DocumentFormat
    metadata: dict[str, str] = Field(default_factory=dict)


class SourceSpan(_FrozenModel):
    """Locate extracted text without inventing unavailable coordinates."""

    block_index: int = Field(ge=0)
    page_number: int | None = Field(default=None, ge=1)
    bbox: tuple[float, float, float, float] | None = None
    page_width: float | None = Field(default=None, gt=0)
    page_height: float | None = Field(default=None, gt=0)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_location(self) -> SourceSpan:
        """Require coordinate and character-offset fields as groups."""

        page_values = (
            self.page_number,
            self.bbox,
            self.page_width,
            self.page_height,
        )

        if any(value is not None for value in page_values) and not all(
            value is not None
            for value in page_values
        ):
            raise ValueError(
                "Page number, bounding box, width, and height "
                "must be provided together"
            )

        if self.bbox is not None:
            x0, y0, x1, y1 = self.bbox

            if x1 <= x0 or y1 <= y0:
                raise ValueError(
                    "Source span bounding box must have "
                    "positive width and height"
                )

        has_start = self.char_start is not None
        has_end = self.char_end is not None

        if has_start != has_end:
            raise ValueError(
                "Character offsets must be provided together"
            )

        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError(
                "Character end must be greater than start"
            )

        return self


class PageGeometry(_FrozenModel):
    """Preserve physical page dimensions, including empty pages."""

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)


class ExtractedBlock(_FrozenModel):
    """Represent one ordered text block and its exact source location."""

    order_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    source_span: SourceSpan


class ExtractedDocument(_FrozenModel):
    """Contain format-neutral blocks produced by an extractor."""

    source: DocumentSource
    blocks: tuple[ExtractedBlock, ...] = Field(min_length=1)
    page_geometries: tuple[PageGeometry, ...] = ()

    @model_validator(mode="after")
    def validate_extracted_document(self) -> ExtractedDocument:
        """Validate ordering and page references without fabricating data."""

        order_indexes = [block.order_index for block in self.blocks]

        if order_indexes != list(range(len(self.blocks))):
            raise ValueError(
                "Extracted block order indexes must be contiguous"
            )

        page_numbers = [
            page.page_number
            for page in self.page_geometries
        ]

        if len(page_numbers) != len(set(page_numbers)):
            raise ValueError(
                "Page geometries must have unique page numbers"
            )

        geometries = {
            page.page_number: page
            for page in self.page_geometries
        }

        for block in self.blocks:
            span = block.source_span

            if span.page_number is None:
                continue

            geometry = geometries.get(span.page_number)

            if geometry is None:
                raise ValueError(
                    "Extracted block refers to an unknown page"
                )

            if (
                span.page_width != geometry.width
                or span.page_height != geometry.height
            ):
                raise ValueError(
                    "Extracted block page dimensions do not match"
                )

        return self


class StructuredSection(_FrozenModel):
    """Group extracted blocks under an explicit logical path."""

    structure_path: tuple[str, ...] = Field(min_length=1)
    blocks: tuple[ExtractedBlock, ...] = Field(min_length=1)


class StructuredDocument(_FrozenModel):
    """Represent parser output independently from a domain schema."""

    source: DocumentSource
    sections: tuple[StructuredSection, ...] = Field(min_length=1)


class DocumentChunk(_FrozenModel):
    """Represent one generic, traceable document chunk."""

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_version: str = Field(min_length=1)
    source_checksum_sha256: str = Field(
        min_length=64,
        max_length=64,
        pattern=r"^[0-9a-f]{64}$",
    )
    document_format: DocumentFormat
    role: DocumentRole
    access_scope: str = Field(min_length=1)
    text: str = Field(min_length=1)
    structure_path: tuple[str, ...] = Field(min_length=1)
    source_spans: tuple[SourceSpan, ...] = Field(min_length=1)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_page_range(self) -> DocumentChunk:
        """Derive page requirements solely from real source spans."""

        span_pages = [span.page_number for span in self.source_spans]
        pages_available = all(page is not None for page in span_pages)
        pages_absent = all(page is None for page in span_pages)

        if not (pages_available or pages_absent):
            raise ValueError(
                "Chunk source spans cannot mix paged and unpaged text"
            )

        if pages_absent:
            if self.page_start is not None or self.page_end is not None:
                raise ValueError(
                    "Unpaged chunks cannot declare a page range"
                )

            return self

        if self.page_start is None or self.page_end is None:
            raise ValueError(
                "Paged chunks require a page range"
            )

        concrete_pages = [
            page
            for page in span_pages
            if page is not None
        ]

        if (
            self.page_start != min(concrete_pages)
            or self.page_end != max(concrete_pages)
        ):
            raise ValueError(
                "Chunk page range must match its source spans"
            )

        return self


class DocumentProcessingResult(_FrozenModel):
    """Contain every deterministic artifact from one processing run."""

    source: DocumentSource
    extracted: ExtractedDocument
    structured: StructuredDocument
    chunks: tuple[DocumentChunk, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_lineage(self) -> DocumentProcessingResult:
        """Require every artifact to belong to the same source."""

        if (
            self.extracted.source != self.source
            or self.structured.source != self.source
        ):
            raise ValueError(
                "Processing artifacts must share one document source"
            )

        for chunk in self.chunks:
            if (
                chunk.source_id != self.source.source_id
                or chunk.source_version
                != self.source.source_version
                or chunk.source_checksum_sha256
                != self.source.checksum_sha256
            ):
                raise ValueError(
                    "Document chunk lineage does not match its source"
                )

        return self


class PageTextBlock(BaseModel):
    """Represent a positioned text block extracted from a PDF page."""

    block_index: int = Field(ge=0)
    text: str = Field(min_length=1)
    bbox: tuple[float, float, float, float]

    @model_validator(mode="after")
    def validate_bounding_box(self) -> PageTextBlock:
        """Ensure the bounding box has a positive width and height."""

        x0, y0, x1, y1 = self.bbox

        if x1 <= x0 or y1 <= y0:
            raise ValueError(
                "Bounding box must have a positive width and height"
            )

        return self


class ExtractedPage(BaseModel):
    """Represent text blocks extracted from one PDF page."""

    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: tuple[PageTextBlock, ...]

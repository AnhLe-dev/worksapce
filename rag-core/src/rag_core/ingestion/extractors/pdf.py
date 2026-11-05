"""PyMuPDF adapter for layout-aware PDF extraction."""

from pathlib import Path

import pymupdf

from rag_core.ingestion.errors import (
    EncryptedDocumentError,
    InvalidPdfError,
    LayoutUnsupportedError,
    OcrRequiredError,
)
from rag_core.ingestion.loading import (
    validate_pdf_path,
    validate_source_file,
)
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentSource,
    ExtractedBlock,
    ExtractedDocument,
    ExtractedPage,
    PageGeometry,
    PageTextBlock,
    SourceSpan,
)


class PdfDocumentExtractor:
    """Extract positioned PDF text while preserving empty pages."""

    @property
    def document_format(self) -> DocumentFormat:
        """Return the format handled by this adapter."""

        return DocumentFormat.PDF

    @staticmethod
    def _extract_page(
        page: pymupdf.Page,
        page_number: int,
    ) -> ExtractedPage:
        """Extract the exact legacy layout view of one PDF page."""

        text_blocks: list[PageTextBlock] = []

        for raw_block in page.get_text("blocks", sort=True):
            (
                x0,
                y0,
                x1,
                y1,
                text,
                block_number,
                block_type,
            ) = raw_block[:7]

            if block_type != 0:
                continue

            stripped_text = str(text).strip()

            if not stripped_text:
                continue

            text_blocks.append(
                PageTextBlock(
                    block_index=int(block_number),
                    text=stripped_text,
                    bbox=(
                        float(x0),
                        float(y0),
                        float(x1),
                        float(y1),
                    ),
                )
            )

        return ExtractedPage(
            page_number=page_number,
            width=float(page.rect.width),
            height=float(page.rect.height),
            blocks=tuple(text_blocks),
        )

    @staticmethod
    def _looks_like_multiple_columns(
        page: ExtractedPage,
    ) -> bool:
        """Conservatively detect side-by-side text columns.

        False positives are preferable to returning evidence with an
        unverified reading order.
        """

        if len(page.blocks) < 4:
            return False

        midpoint = page.width / 2
        left_blocks = [
            block
            for block in page.blocks
            if block.bbox[2] <= midpoint
        ]
        right_blocks = [
            block
            for block in page.blocks
            if block.bbox[0] >= midpoint
        ]

        if len(left_blocks) < 2 or len(right_blocks) < 2:
            return False

        overlapping_pairs = 0

        for left in left_blocks:
            left_y0 = left.bbox[1]
            left_y1 = left.bbox[3]

            for right in right_blocks:
                right_y0 = right.bbox[1]
                right_y1 = right.bbox[3]
                overlap = min(left_y1, right_y1) - max(
                    left_y0,
                    right_y0,
                )

                if overlap > 0:
                    overlapping_pairs += 1
                    break

        return overlapping_pairs >= 2

    def extract_pages(
        self,
        file_path: str | Path,
    ) -> tuple[ExtractedPage, ...]:
        """Return the compatibility page view used by existing parsers."""

        resolved_path = validate_pdf_path(
            Path(file_path)
        )

        try:
            with pymupdf.open(str(resolved_path)) as document:
                if document.needs_pass:
                    raise EncryptedDocumentError(
                        "Encrypted PDF documents are not supported"
                    )

                pages = tuple(
                    self._extract_page(
                        page,
                        page_index + 1,
                    )
                    for page_index, page in enumerate(document)
                )
        except pymupdf.FileDataError as error:
            raise InvalidPdfError(
                f"PDF structure is invalid: {resolved_path}"
            ) from error

        if not pages:
            raise InvalidPdfError(
                "PDF document does not contain any pages"
            )

        if not any(page.blocks for page in pages):
            raise OcrRequiredError(
                "PDF document contains no extractable text; "
                "OCR is required"
            )
        unsupported_pages = [
            page.page_number
            for page in pages
            if self._looks_like_multiple_columns(page)
        ]

        if unsupported_pages:
            raise LayoutUnsupportedError(
                "Multi-column PDF layout is not supported for pages: "
                + ", ".join(
                    str(page_number)
                    for page_number in unsupported_pages
                )
            )
        return pages

    def extract(
        self,
        source: DocumentSource,
        file_path: Path,
    ) -> ExtractedDocument:
        """Convert the PDF layout view to generic extracted blocks."""

        if source.document_format is not DocumentFormat.PDF:
            raise ValueError(
                "PDF extractor requires a PDF document source"
            )

        validate_source_file(source, file_path)
        pages = self.extract_pages(file_path)
        blocks: list[ExtractedBlock] = []

        for page in pages:
            for block in page.blocks:
                blocks.append(
                    ExtractedBlock(
                        order_index=len(blocks),
                        text=block.text,
                        source_span=SourceSpan(
                            page_number=page.page_number,
                            block_index=block.block_index,
                            bbox=block.bbox,
                            page_width=page.width,
                            page_height=page.height,
                        ),
                    )
                )

        return ExtractedDocument(
            source=source,
            blocks=tuple(blocks),
            page_geometries=tuple(
                PageGeometry(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                )
                for page in pages
            ),
        )

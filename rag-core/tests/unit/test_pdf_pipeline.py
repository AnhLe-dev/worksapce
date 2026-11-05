"""Offline PDF extraction tests for supported and rejected layouts."""

from pathlib import Path

import pymupdf
import pytest

from rag_core.chunking import BlockChunkingStrategy
from rag_core.ingestion.errors import (
    InvalidFileSignatureError,
    LayoutUnsupportedError,
    OcrRequiredError,
)
from rag_core.ingestion.extractors.pdf import PdfDocumentExtractor
from rag_core.ingestion.models import (
    DocumentInput,
    DocumentRole,
    SourceSpan,
)
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor


def save_pdf(
    path: Path,
    *,
    positioned_text: tuple[tuple[float, float, str], ...] = (),
) -> None:
    """Create one licensed test-only PDF in a temporary directory."""

    document = pymupdf.open()
    page = document.new_page(width=600, height=800)

    for x, y, text in positioned_text:
        page.insert_text((x, y), text)

    document.save(path)
    document.close()


def make_processor() -> DocumentProcessor:
    """Build the declared single-column PDF pipeline."""

    return DocumentProcessor(
        extractors=(PdfDocumentExtractor(),),
        parser=FlatStructureParser(),
        chunking_strategy=BlockChunkingStrategy(
            max_characters=500,
        ),
    )


def make_input(path: Path) -> DocumentInput:
    """Build one generic PDF input."""

    return DocumentInput(
        path=path,
        source_id="pdf-policy",
        source_version="1",
        access_scope="tenant-a",
        role=DocumentRole.KNOWLEDGE,
    )


def test_single_column_pdf_preserves_page_geometry(tmp_path: Path) -> None:
    path = tmp_path / "policy.pdf"
    save_pdf(
        path,
        positioned_text=(
            (72, 72, "Refund requests require a receipt."),
            (72, 110, "Requests must be filed within thirty days."),
        ),
    )

    result = make_processor().process(make_input(path))

    assert len(result.extracted.page_geometries) == 1
    assert result.extracted.page_geometries[0].page_number == 1
    assert result.chunks

    for span in result.chunks[0].source_spans:
        assert span.page_number == 1
        assert span.bbox is not None
        assert span.page_width == 600
        assert span.page_height == 800
        assert span.char_start is None
        assert span.char_end is None

    assert result.chunks[0].page_start == 1
    assert result.chunks[0].page_end == 1


def test_blank_pdf_requires_ocr(tmp_path: Path) -> None:
    path = tmp_path / "scan.pdf"
    save_pdf(path)

    with pytest.raises(OcrRequiredError) as error:
        make_processor().process(make_input(path))

    assert error.value.code.value == "ocr_required"


def test_pdf_with_invalid_signature_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"not-a-pdf")

    with pytest.raises(InvalidFileSignatureError) as error:
        make_processor().process(make_input(path))

    assert error.value.code.value == "invalid_file_signature"


def test_multi_column_pdf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "columns.pdf"
    save_pdf(
        path,
        positioned_text=(
            (50, 100, "Left column first row"),
            (350, 110, "Right column first row"),
            (50, 150, "Left column second row"),
            (350, 160, "Right column second row"),
        ),
    )

    with pytest.raises(LayoutUnsupportedError) as error:
        make_processor().process(make_input(path))

    assert error.value.code.value == "layout_unsupported"


def test_source_span_rejects_partial_page_geometry() -> None:
    with pytest.raises(ValueError, match="must be provided together"):
        SourceSpan(
            block_index=0,
            page_number=1,
            bbox=(10.0, 20.0, 30.0, 40.0),
        )

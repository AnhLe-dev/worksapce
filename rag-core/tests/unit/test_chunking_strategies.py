"""Tests for fixed and structure-aware chunking provenance."""

from pathlib import Path

import pytest

from rag_core.chunking import (
    BlockChunkingStrategy,
    FixedChunkingStrategy,
    StructureAwareChunkingStrategy,
)
from rag_core.ingestion.extractors.text import TextDocumentExtractor
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentInput,
    DocumentRole,
    DocumentSource,
    ExtractedBlock,
    SourceSpan,
    StructuredDocument,
    StructuredSection,
)
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor


def make_processor(strategy) -> DocumentProcessor:
    """Build a TXT processor with one supplied chunking strategy."""

    return DocumentProcessor(
        extractors=(TextDocumentExtractor(),),
        parser=FlatStructureParser(),
        chunking_strategy=strategy,
    )


def make_input(path: Path) -> DocumentInput:
    """Build one generic knowledge-document input."""

    return DocumentInput(
        path=path,
        source_id="policy",
        source_version="1",
        access_scope="tenant-a",
        role=DocumentRole.KNOWLEDGE,
    )


def test_fixed_windows_preserve_exact_offsets_and_overlap(
    tmp_path: Path,
) -> None:
    content = "A" * 450
    path = tmp_path / "policy.txt"
    path.write_text(content, encoding="utf-8")
    processor = make_processor(
        FixedChunkingStrategy(
            max_characters=200,
            overlap_characters=20,
        )
    )

    first = processor.process(make_input(path))
    second = processor.process(make_input(path))

    assert [len(chunk.text) for chunk in first.chunks] == [
        200,
        200,
        90,
    ]
    reconstructed = first.chunks[0].text + "".join(
        chunk.text[20:]
        for chunk in first.chunks[1:]
    )
    assert reconstructed == content
    assert [
        chunk.chunk_id
        for chunk in first.chunks
    ] == [
        chunk.chunk_id
        for chunk in second.chunks
    ]

    for chunk in first.chunks:
        span = chunk.source_spans[0]
        assert span.char_start is not None
        assert span.char_end is not None
        assert content[span.char_start:span.char_end] == chunk.text
        assert chunk.metadata["chunk_strategy"] == "fixed"
        assert chunk.metadata["chunk_strategy_version"] == "1"


def test_structure_aware_uses_block_strategy_for_short_sections(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.txt"
    path.write_text(
        "First requirement.\n\nSecond requirement.",
        encoding="utf-8",
    )
    processor = make_processor(
        StructureAwareChunkingStrategy(
            max_characters=200,
            fallback_overlap_characters=20,
        )
    )

    result = processor.process(make_input(path))

    assert len(result.chunks) == 1
    assert result.chunks[0].text == (
        "First requirement.\nSecond requirement."
    )
    assert result.chunks[0].metadata["chunk_strategy"] == (
        "structure_aware"
    )
    assert result.chunks[0].metadata[
        "chunk_strategy_fallback"
    ] == "block"


def test_structure_aware_falls_back_for_oversize_txt_block(
    tmp_path: Path,
) -> None:
    content = "B" * 450
    path = tmp_path / "policy.txt"
    path.write_text(content, encoding="utf-8")
    processor = make_processor(
        StructureAwareChunkingStrategy(
            max_characters=200,
            fallback_overlap_characters=20,
        )
    )

    result = processor.process(make_input(path))

    assert len(result.chunks) == 3
    assert all(
        chunk.metadata["chunk_strategy"] == "structure_aware"
        for chunk in result.chunks
    )
    assert all(
        chunk.metadata["chunk_strategy_fallback"] == "fixed"
        for chunk in result.chunks
    )


def test_fixed_strategy_rejects_unsafe_pdf_sub_block_geometry() -> None:
    text = "C" * 250
    source = DocumentSource(
        source_id="positioned-policy",
        filename="policy.pdf",
        source_version="1",
        checksum_sha256="a" * 64,
        access_scope="tenant-a",
        role=DocumentRole.KNOWLEDGE,
        document_format=DocumentFormat.PDF,
    )
    document = StructuredDocument(
        source=source,
        sections=(
            StructuredSection(
                structure_path=("section-1",),
                blocks=(
                    ExtractedBlock(
                        order_index=0,
                        text=text,
                        source_span=SourceSpan(
                            block_index=0,
                            page_number=1,
                            bbox=(10.0, 20.0, 300.0, 80.0),
                            page_width=600.0,
                            page_height=800.0,
                        ),
                    ),
                ),
            ),
        ),
    )
    strategy = FixedChunkingStrategy(
        max_characters=200,
        overlap_characters=20,
    )

    with pytest.raises(ValueError, match="sub-block geometry"):
        strategy.chunk(document)


def test_block_strategy_records_its_profile_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("One complete block.", encoding="utf-8")
    result = make_processor(
        BlockChunkingStrategy(max_characters=200)
    ).process(make_input(path))

    assert result.chunks[0].metadata["chunk_strategy"] == "block"
    assert result.chunks[0].metadata["chunk_strategy_version"] == "1"

    different_profile = make_processor(
        BlockChunkingStrategy(max_characters=300)
    ).process(make_input(path))

    assert different_profile.chunks[0].chunk_id != (
        result.chunks[0].chunk_id
    )

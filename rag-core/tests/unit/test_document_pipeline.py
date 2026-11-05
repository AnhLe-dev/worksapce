"""Unit tests for TXT processing and composable document strategies."""

import tempfile
import unittest
from pathlib import Path

from rag_core.chunking.block import BlockChunkingStrategy
from rag_core.ingestion.errors import (
    UnsupportedDocumentFormatError,
)
from rag_core.ingestion.extractors.text import (
    TextDocumentExtractor,
)
from rag_core.ingestion.loading import load_document_source
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentInput,
    DocumentRole,
    StructuredDocument,
)
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor


class RecordingTextExtractor:
    """Custom extractor proving composition through the public port."""

    def __init__(self) -> None:
        self.called = False
        self._delegate = TextDocumentExtractor()

    @property
    def document_format(self) -> DocumentFormat:
        return DocumentFormat.TXT

    def extract(self, source, file_path):
        self.called = True
        return self._delegate.extract(source, file_path)


class RecordingParser:
    """Custom parser delegating to the safe flat fallback."""

    def __init__(self) -> None:
        self.called = False
        self._delegate = FlatStructureParser()

    def parse(self, document):
        self.called = True
        return self._delegate.parse(document)


class RecordingChunker:
    """Custom chunker delegating to whole-block chunking."""

    def __init__(self) -> None:
        self.called = False
        self._delegate = BlockChunkingStrategy(
            max_characters=200
        )

    def chunk(self, document):
        self.called = True
        return self._delegate.chunk(document)


class PdfOnlyExtractor:
    """Dependency-light fake used to test missing format selection."""

    @property
    def document_format(self) -> DocumentFormat:
        return DocumentFormat.PDF

    def extract(self, source, file_path):
        raise AssertionError("PDF fake must not be called")


class EmptyParser:
    """Return deliberately invalid output to test the service guard."""

    def parse(self, document):
        return StructuredDocument.model_construct(
            source=document.source,
            sections=(),
        )


class DocumentPipelineTests(unittest.TestCase):
    """Protect TXT provenance, failures, and strategy replacement."""

    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.root = Path(self.temporary_directory.name)

    def _input(self, path: Path) -> DocumentInput:
        return DocumentInput(
            path=path,
            source_id="internal-policy",
            source_version="2026-10",
            access_scope="tenant-a",
            role=DocumentRole.KNOWLEDGE,
            metadata={"owner": "legal"},
        )

    @staticmethod
    def _processor(
        *,
        extractor=None,
        parser=None,
        chunker=None,
    ) -> DocumentProcessor:
        return DocumentProcessor(
            extractors=(
                extractor or TextDocumentExtractor(),
            ),
            parser=parser or FlatStructureParser(),
            chunking_strategy=(
                chunker
                or BlockChunkingStrategy(
                    max_characters=200
                )
            ),
        )

    def test_txt_pipeline_preserves_offsets_without_fake_pages(self) -> None:
        content = (
            "  First policy paragraph.  \n\n"
            "Second line one.\nSecond line two.  \n"
        )
        path = self.root / "policy.txt"
        path.write_text(content, encoding="utf-8")

        result = self._processor().process(
            self._input(path)
        )

        self.assertEqual(
            result.source.document_format,
            DocumentFormat.TXT,
        )
        self.assertEqual(len(result.extracted.blocks), 2)
        self.assertEqual(result.extracted.page_geometries, ())
        self.assertEqual(len(result.structured.sections), 1)
        self.assertEqual(len(result.chunks), 1)
        self.assertEqual(result.chunks[0].metadata["owner"], "legal")
        self.assertIsNone(result.chunks[0].page_start)
        self.assertIsNone(result.chunks[0].page_end)

        for block in result.extracted.blocks:
            span = block.source_span
            self.assertIsNone(span.page_number)
            self.assertIsNone(span.bbox)
            self.assertEqual(
                content[span.char_start:span.char_end],
                block.text,
            )

    def test_unicode_txt_preserves_exact_offsets(self) -> None:
        content = "Chính sách hoàn tiền.\n\nCần hóa đơn hợp lệ."
        path = self.root / "unicode-policy.txt"
        path.write_text(content, encoding="utf-8")

        result = self._processor().process(self._input(path))

        self.assertEqual(len(result.extracted.blocks), 2)

        for block in result.extracted.blocks:
            span = block.source_span
            self.assertIsNotNone(span.char_start)
            self.assertIsNotNone(span.char_end)
            self.assertEqual(
                content[span.char_start:span.char_end],
                block.text,
            )

    def test_missing_file_is_rejected(self) -> None:
        path = self.root / "missing.txt"

        with self.assertRaises(FileNotFoundError):
            self._processor().process(self._input(path))

    def test_chunk_ids_are_deterministic(self) -> None:
        path = self.root / "policy.txt"
        path.write_text("First paragraph.\n\nSecond paragraph.")
        processor = self._processor()
        document_input = self._input(path)

        first = processor.process(document_input)
        second = processor.process(document_input)

        self.assertEqual(
            [chunk.chunk_id for chunk in first.chunks],
            [chunk.chunk_id for chunk in second.chunks],
        )

    def test_extractor_rejects_content_changed_after_source_load(self) -> None:
        path = self.root / "policy.txt"
        document_input = self._input(path)
        path.write_text("Original policy")
        source = load_document_source(document_input)
        path.write_text("Changed policy")

        with self.assertRaisesRegex(
            ValueError,
            "checksum does not match",
        ):
            TextDocumentExtractor().extract(
                source,
                path,
            )

    def test_custom_strategies_run_without_processor_changes(self) -> None:
        path = self.root / "policy.txt"
        path.write_text("Custom strategy proof.")
        extractor = RecordingTextExtractor()
        parser = RecordingParser()
        chunker = RecordingChunker()

        result = self._processor(
            extractor=extractor,
            parser=parser,
            chunker=chunker,
        ).process(self._input(path))

        self.assertTrue(extractor.called)
        self.assertTrue(parser.called)
        self.assertTrue(chunker.called)
        self.assertEqual(len(result.chunks), 1)

    def test_rejects_duplicate_or_missing_extractors(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate extractor",
        ):
            DocumentProcessor(
                extractors=(
                    TextDocumentExtractor(),
                    TextDocumentExtractor(),
                ),
                parser=FlatStructureParser(),
                chunking_strategy=BlockChunkingStrategy(
                    max_characters=200
                ),
            )

        path = self.root / "policy.txt"
        path.write_text("Policy")
        processor = DocumentProcessor(
            extractors=(PdfOnlyExtractor(),),
            parser=FlatStructureParser(),
            chunking_strategy=BlockChunkingStrategy(
                max_characters=200
            ),
        )

        with self.assertRaisesRegex(
            UnsupportedDocumentFormatError,
            "No extractor is configured",
        ):
            processor.process(self._input(path))

    def test_rejects_empty_structure_and_oversized_block(self) -> None:
        path = self.root / "policy.txt"
        path.write_text("x" * 201)

        with self.assertRaisesRegex(
            ValueError,
            "produced no sections",
        ):
            self._processor(
                parser=EmptyParser(),
            ).process(self._input(path))

        with self.assertRaisesRegex(
            ValueError,
            "exceeds the chunk limit",
        ):
            self._processor().process(
                self._input(path)
            )

    def test_rejects_invalid_txt_and_unsupported_formats(self) -> None:
        invalid_utf8 = self.root / "invalid.txt"
        invalid_utf8.write_bytes(b"\xff\xfe")
        nul_text = self.root / "nul.txt"
        nul_text.write_bytes(b"policy\x00text")
        whitespace = self.root / "whitespace.txt"
        whitespace.write_text(" \n\t\n ")
        empty = self.root / "empty.txt"
        empty.write_bytes(b"")
        unsupported = self.root / "policy.docx"
        unsupported.write_bytes(b"not-a-docx")
        disguised_pdf = self.root / "disguised.txt"
        disguised_pdf.write_bytes(b"%PDF-not-text")

        cases = (
            (invalid_utf8, "valid UTF-8"),
            (nul_text, "NUL"),
            (whitespace, "extractable text"),
            (empty, "empty"),
            (unsupported, "Unsupported document format"),
            (disguised_pdf, "does not match"),
        )

        for path, message in cases:
            with self.subTest(path=path), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                self._processor().process(
                    self._input(path)
                )


if __name__ == "__main__":
    unittest.main()

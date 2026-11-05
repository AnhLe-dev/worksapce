"""Offline Phase 1 smoke test across every generic core stage."""

from pathlib import Path

import pymupdf

import rag_core
from rag_core.chunking import StructureAwareChunkingStrategy
from rag_core.generation import GroundedRagService
from rag_core.indexing import (
    DocumentIndexingService,
    IndexingResult,
    IndexingStatus,
)
from rag_core.ingestion.extractors.pdf import PdfDocumentExtractor
from rag_core.ingestion.extractors.text import TextDocumentExtractor
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentInput,
    DocumentRole,
)
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor
from rag_core.retrieval import (
    DenseSourceRetriever,
    KnowledgeHit,
    RetrievalRequest,
    RetrievalScope,
    RetrievalService,
)

INDEX_PROFILE = "fake:dimensions=3:chunk=structure_aware:v1"


class FakeEmbeddingProvider:
    """Provide deterministic document and query vectors offline."""

    def embed_documents(self, texts):
        return tuple(
            (1.0, float(index), 0.5)
            for index, _ in enumerate(texts)
        )

    def embed_query(self, text):
        return (1.0, 0.0, 0.5)


class InMemoryVectorStore:
    """Implement indexing and search ports for one smoke run."""

    def __init__(self) -> None:
        self.chunks = ()
        self.vectors = ()

    def upsert(self, *, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors
        first = chunks[0]

        return IndexingResult(
            collection_name="memory",
            source_id=first.source_id,
            source_version=first.source_version,
            source_checksum_sha256=first.source_checksum_sha256,
            access_scope=first.access_scope,
            status=IndexingStatus.INDEXED,
            chunk_count=len(chunks),
            uploaded_count=len(chunks),
            stored_count=len(chunks),
            vector_dimensions=len(vectors[0]),
        )

    def search(self, *, query_vector, scope, limit):
        assert query_vector == (1.0, 0.0, 0.5)
        matching = tuple(
            chunk
            for chunk in self.chunks
            if (
                chunk.access_scope == scope.access_scope
                and chunk.source_id == scope.source_id
                and chunk.source_version == scope.source_version
                and chunk.source_checksum_sha256
                == scope.source_checksum_sha256
            )
        )

        return tuple(
            KnowledgeHit(
                chunk_id=chunk.chunk_id,
                source_id=chunk.source_id,
                source_version=chunk.source_version,
                source_checksum_sha256=(
                    chunk.source_checksum_sha256
                ),
                access_scope=chunk.access_scope,
                index_profile=scope.index_profile,
                text=chunk.text,
                structure_path=chunk.structure_path,
                source_spans=chunk.source_spans,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                score=1.0 - (index * 0.01),
                document_format=chunk.document_format,
                role=chunk.role,
                metadata=dict(chunk.metadata),
            )
            for index, chunk in enumerate(matching[:limit])
        )


class FakeAnswerGenerator:
    """Return one source-labelled answer without a live model."""

    def __init__(self) -> None:
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return "A receipt is required for a refund [S1]."


def test_phase1_txt_to_grounded_answer_smoke(tmp_path: Path) -> None:
    content = (
        "Refund requests require a receipt.\n\n"
        "Requests must be submitted within thirty days."
    )
    path = tmp_path / "refund-policy.txt"
    path.write_text(content, encoding="utf-8")
    processor = DocumentProcessor(
        extractors=(TextDocumentExtractor(),),
        parser=FlatStructureParser(),
        chunking_strategy=StructureAwareChunkingStrategy(
            max_characters=300,
            fallback_overlap_characters=30,
        ),
    )

    processing = processor.process(
        DocumentInput(
            path=path,
            source_id="refund-policy",
            source_version="2026-01",
            access_scope="tenant-a",
            role=DocumentRole.KNOWLEDGE,
        )
    )

    assert processing.source.document_format is DocumentFormat.TXT
    assert processing.chunks
    assert processing.chunks[0].metadata["chunk_strategy"] == (
        "structure_aware"
    )

    embeddings = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    indexing = DocumentIndexingService(
        embedding_provider=embeddings,
        vector_index=vector_store,
        expected_dimensions=3,
    )
    indexing_result = indexing.index(processing.chunks)

    assert indexing_result.status is IndexingStatus.INDEXED
    assert indexing_result.stored_count == len(processing.chunks)

    scope = RetrievalScope(
        access_scope=processing.source.access_scope,
        source_id=processing.source.source_id,
        source_version=processing.source.source_version,
        source_checksum_sha256=processing.source.checksum_sha256,
        index_profile=INDEX_PROFILE,
    )
    source_retriever = DenseSourceRetriever(
        embedding_provider=embeddings,
        vector_search=vector_store,
        expected_dimensions=3,
    )
    retrieval = RetrievalService(
        source_retriever=source_retriever,
    )
    generator = FakeAnswerGenerator()
    rag = GroundedRagService(
        retriever=retrieval,
        answer_generator=generator,
    )
    execution = rag.execute(
        RetrievalRequest(
            query="What is required for a refund?",
            scopes=(scope,),
            limit=3,
        )
    )

    assert execution.result.answer == (
        "A receipt is required for a refund [S1]."
    )
    assert execution.result.citations[0].source_id == "refund-policy"
    assert execution.result.citations[0].source_checksum_sha256 == (
        processing.source.checksum_sha256
    )
    citation = execution.result.citations[0]
    cited_source_text = "\n".join(
        content[span.char_start:span.char_end]
        for span in citation.source_spans
        if span.char_start is not None and span.char_end is not None
    )
    assert cited_source_text == citation.text
    assert len(generator.prompts) == 1
    assert execution.trace.rendered_context is not None
    assert rag_core.__version__


def test_phase1_pdf_to_grounded_answer_smoke(tmp_path: Path) -> None:
    path = tmp_path / "refund-policy.pdf"
    document = pymupdf.open()
    page = document.new_page(width=600, height=800)
    page.insert_text(
        (72, 72),
        "Refund requests require a receipt.",
    )
    document.save(path)
    document.close()
    processor = DocumentProcessor(
        extractors=(PdfDocumentExtractor(),),
        parser=FlatStructureParser(),
        chunking_strategy=StructureAwareChunkingStrategy(
            max_characters=300,
            fallback_overlap_characters=30,
        ),
    )
    processing = processor.process(
        DocumentInput(
            path=path,
            source_id="refund-pdf",
            source_version="2026-01",
            access_scope="tenant-a",
            role=DocumentRole.KNOWLEDGE,
        )
    )
    embeddings = FakeEmbeddingProvider()
    vector_store = InMemoryVectorStore()
    indexing_result = DocumentIndexingService(
        embedding_provider=embeddings,
        vector_index=vector_store,
        expected_dimensions=3,
    ).index(processing.chunks)

    assert indexing_result.status is IndexingStatus.INDEXED

    scope = RetrievalScope(
        access_scope=processing.source.access_scope,
        source_id=processing.source.source_id,
        source_version=processing.source.source_version,
        source_checksum_sha256=processing.source.checksum_sha256,
        index_profile=INDEX_PROFILE,
    )
    retrieval = RetrievalService(
        source_retriever=DenseSourceRetriever(
            embedding_provider=embeddings,
            vector_search=vector_store,
            expected_dimensions=3,
        ),
    )
    execution = GroundedRagService(
        retriever=retrieval,
        answer_generator=FakeAnswerGenerator(),
    ).execute(
        RetrievalRequest(
            query="What is required for a refund?",
            scopes=(scope,),
            limit=3,
        )
    )

    citation = execution.result.citations[0]
    assert citation.page_start == 1
    assert citation.page_end == 1
    assert citation.source_spans[0].bbox is not None
    assert citation.source_spans[0].page_width == 600
    assert citation.source_spans[0].page_height == 800

"""Unit tests for provider-independent indexing invariants."""

import math

import pytest

from rag_core.indexing import (
    DocumentIndexingService,
    IndexingResult,
    IndexingStatus,
    IndexingValidationError,
    SourceVersionConflictError,
    build_index_payload,
)
from rag_core.ingestion.models import (
    DocumentChunk,
    DocumentFormat,
    DocumentRole,
    SourceSpan,
)

CHECKSUM = "a" * 64


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    source_id: str = "policy",
    source_version: str = "1",
    checksum: str = CHECKSUM,
    access_scope: str = "tenant-a",
    metadata: dict[str, str] | None = None,
) -> DocumentChunk:
    """Build one generic TXT chunk with real character provenance."""

    text = f"Text for {chunk_id}"

    return DocumentChunk(
        chunk_id=chunk_id,
        source_id=source_id,
        source_version=source_version,
        source_checksum_sha256=checksum,
        document_format=DocumentFormat.TXT,
        role=DocumentRole.KNOWLEDGE,
        access_scope=access_scope,
        text=text,
        structure_path=("policy.txt", chunk_id),
        source_spans=(
            SourceSpan(
                block_index=0,
                char_start=0,
                char_end=len(text),
            ),
        ),
        metadata=metadata or {"owner": "legal"},
    )


class FakeEmbeddingProvider:
    """Record embedding input and return configured vectors."""

    def __init__(self, vectors) -> None:
        self.vectors = vectors
        self.texts = None
        self.call_count = 0

    def embed_documents(self, texts):
        self.call_count += 1
        self.texts = texts
        return self.vectors


class FakeVectorIndex:
    """Record the validated snapshot received from the service."""

    def __init__(self) -> None:
        self.chunks = None
        self.vectors = None

    def upsert(self, *, chunks, vectors):
        self.chunks = chunks
        self.vectors = vectors
        first = chunks[0]

        return IndexingResult(
            collection_name="knowledge",
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


class ConflictingVectorIndex:
    """Simulate an immutable snapshot conflict in an adapter."""

    def upsert(self, *, chunks, vectors):
        raise SourceVersionConflictError("checksum conflict")


def make_service(vectors, *, vector_index=None, dimensions=3):
    provider = FakeEmbeddingProvider(vectors)
    index = vector_index or FakeVectorIndex()

    service = DocumentIndexingService(
        embedding_provider=provider,
        vector_index=index,
        expected_dimensions=dimensions,
    )

    return service, provider, index


def test_embeds_in_order_and_delegates_validated_vectors() -> None:
    chunks = (
        make_chunk(chunk_id="chunk-1"),
        make_chunk(chunk_id="chunk-2"),
    )

    service, provider, index = make_service(
        ([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
    )

    result = service.index(chunks)

    assert result.status is IndexingStatus.INDEXED

    assert provider.texts == (
        "policy.txt > chunk-1\nText for chunk-1",
        "policy.txt > chunk-2\nText for chunk-2",
    )

    assert index.chunks == chunks

    assert index.vectors == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )


@pytest.mark.parametrize(
    "chunks",
    [
        (),
        (make_chunk(), make_chunk()),
        (
            make_chunk(),
            make_chunk(
                chunk_id="chunk-2",
                source_id="other",
            ),
        ),
        (
            make_chunk(),
            make_chunk(
                chunk_id="chunk-2",
                source_version="2",
            ),
        ),
        (
            make_chunk(),
            make_chunk(
                chunk_id="chunk-2",
                checksum="b" * 64,
            ),
        ),
        (
            make_chunk(),
            make_chunk(
                chunk_id="chunk-2",
                access_scope="tenant-b",
            ),
        ),
    ],
)
def test_rejects_invalid_snapshot_before_embedding(chunks) -> None:
    service, provider, _ = make_service(())

    with pytest.raises(IndexingValidationError):
        service.index(chunks)

    assert provider.call_count == 0


@pytest.mark.parametrize(
    "vectors",
    [
        (),
        ([0.1, 0.2],),
        ([0.1, math.inf, 0.3],),
        ([0.1, math.nan, 0.3],),
        ([0.1, "not-a-number", 0.3],),
    ],
)
def test_rejects_invalid_embedding_results(vectors) -> None:
    service, _, index = make_service(vectors)

    with pytest.raises(IndexingValidationError):
        service.index((make_chunk(),))

    assert index.chunks is None


def test_payload_is_generic_and_preserves_provenance() -> None:
    chunk = make_chunk(
        metadata={"article_number": "5"}
    )

    payload = build_index_payload(chunk)

    assert payload["chunk_id"] == chunk.chunk_id
    assert payload["source_checksum_sha256"] == CHECKSUM
    assert payload["document_format"] == "txt"
    assert payload["role"] == "knowledge"
    assert payload["metadata"] == {
        "article_number": "5"
    }
    assert payload["source_spans"][0]["char_start"] == 0
    assert "article_number" not in payload
    assert "knowledge_base_id" not in payload


def test_payload_rejects_backend_owned_membership() -> None:
    chunk = make_chunk(
        metadata={"knowledge_base_id": "kb-1"}
    )

    with pytest.raises(
        IndexingValidationError,
        match="backend-owned",
    ):
        build_index_payload(chunk)


def test_adapter_conflict_remains_a_typed_failure() -> None:
    service, _, _ = make_service(
        ([0.1, 0.2, 0.3],),
        vector_index=ConflictingVectorIndex(),
    )

    with pytest.raises(SourceVersionConflictError):
        service.index((make_chunk(),))
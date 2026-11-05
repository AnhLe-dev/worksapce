"""Offline tests for optional OpenAI and Qdrant adapters."""

from types import SimpleNamespace

import pytest
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    VectorParams,
)

from rag_core.adapters.openai_embeddings import (
    OpenAIEmbeddingProvider,
)
from rag_core.adapters.qdrant import (
    QdrantVectorIndex,
    ensure_qdrant_collection,
)
from rag_core.indexing import (
    IndexingStatus,
    IndexingValidationError,
    IndexSnapshotConflictError,
    IndexSnapshotStatus,
    IndexVerificationError,
    SourceVersionConflictError,
)
from rag_core.ingestion.models import (
    DocumentChunk,
    DocumentFormat,
    DocumentRole,
    SourceSpan,
)

CHECKSUM = "a" * 64

INDEX_PROFILE = (
    "openai:text-embedding-3-small:"
    "dimensions=3:chunk=block:v1"
)


def make_chunk(
    *,
    chunk_id: str = "chunk-1",
    checksum: str = CHECKSUM,
) -> DocumentChunk:
    """Build one generic chunk for adapter tests."""

    text = f"Text for {chunk_id}"

    return DocumentChunk(
        chunk_id=chunk_id,
        source_id="policy",
        source_version="1",
        source_checksum_sha256=checksum,
        document_format=DocumentFormat.TXT,
        role=DocumentRole.KNOWLEDGE,
        access_scope="tenant-a",
        text=text,
        structure_path=(
            "policy.txt",
            chunk_id,
        ),
        source_spans=(
            SourceSpan(
                block_index=0,
                char_start=0,
                char_end=len(text),
            ),
        ),
        metadata={
            "owner": "legal",
        },
    )


class FakeEmbeddingsResource:
    """Return configured OpenAI-compatible response items."""

    def __init__(self, data) -> None:
        self.data = data
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        return SimpleNamespace(
            data=self.data,
        )


class FakeOpenAIClient:
    """Expose only the SDK resource used by the adapter."""

    def __init__(self, data) -> None:
        self.embeddings = FakeEmbeddingsResource(data)


class FakeQdrantClient:
    """Emulate only Qdrant operations used by the adapter."""

    def __init__(
        self,
        payloads=(),
        *,
        collection_exists=True,
        vector_dimensions=3,
        distance=Distance.COSINE,
    ) -> None:
        self.payloads = list(payloads)
        self.exists = collection_exists

        self.vector_config = VectorParams(
            size=vector_dimensions,
            distance=distance,
        )

        self.payload_schema = {}

        self.create_collection_calls = []
        self.create_index_calls = []
        self.scroll_calls = []
        self.upsert_calls = []
        self.count_calls = []

    def collection_exists(self, **kwargs):
        return self.exists

    def create_collection(self, **kwargs):
        self.create_collection_calls.append(kwargs)
        self.exists = True
        self.vector_config = kwargs["vectors_config"]

    def get_collection(self, collection_name):
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=self.vector_config,
                ),
            ),
            payload_schema=self.payload_schema,
        )

    def create_payload_index(self, **kwargs):
        self.create_index_calls.append(kwargs)

        self.payload_schema[
            kwargs["field_name"]
        ] = SimpleNamespace(
            data_type=kwargs["field_schema"],
        )

    def scroll(self, **kwargs):
        self.scroll_calls.append(kwargs)

        records = [
            SimpleNamespace(payload=payload)
            for payload in self.payloads
        ]

        return records, None

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)

        for point in kwargs["points"]:
            self.payloads.append(point.payload)

    def count(self, **kwargs):
        self.count_calls.append(kwargs)

        return SimpleNamespace(
            count=len(self.payloads),
        )


def make_qdrant_adapter(
    client,
    *,
    index_profile: str = INDEX_PROFILE,
) -> QdrantVectorIndex:
    """Build a Qdrant adapter with deterministic test settings."""

    return QdrantVectorIndex(
        client=client,
        collection_name="knowledge",
        vector_dimensions=3,
        index_profile=index_profile,
        batch_size=1,
    )


def test_openai_adapter_uses_official_shape_and_restores_order() -> None:
    client = FakeOpenAIClient((
        SimpleNamespace(
            index=1,
            embedding=[0.4, 0.5, 0.6],
        ),
        SimpleNamespace(
            index=0,
            embedding=[0.1, 0.2, 0.3],
        ),
    ))

    adapter = OpenAIEmbeddingProvider(
        client=client,
        model="text-embedding-3-small",
        dimensions=3,
    )

    vectors = adapter.embed_documents((
        "first",
        "second",
    ))

    assert vectors == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )

    assert client.embeddings.calls == [{
        "model": "text-embedding-3-small",
        "input": [
            "first",
            "second",
        ],
        "encoding_format": "float",
        "dimensions": 3,
    }]

    assert adapter.index_profile == (
        "openai:text-embedding-3-small:"
        "dimensions=3"
    )


def test_openai_adapter_returns_empty_without_provider_call() -> None:
    client = FakeOpenAIClient(())

    adapter = OpenAIEmbeddingProvider(
        client=client,
        model="text-embedding-3-small",
        dimensions=3,
    )

    assert adapter.embed_documents(()) == ()
    assert client.embeddings.calls == []


def test_openai_adapter_rejects_empty_input_text() -> None:
    client = FakeOpenAIClient(())

    adapter = OpenAIEmbeddingProvider(
        client=client,
        model="text-embedding-3-small",
        dimensions=3,
    )

    with pytest.raises(
        IndexingValidationError,
        match="empty text",
    ):
        adapter.embed_documents((
            "valid",
            "   ",
        ))

    assert client.embeddings.calls == []


def test_openai_adapter_rejects_invalid_response_indexes() -> None:
    client = FakeOpenAIClient((
        SimpleNamespace(
            index=0,
            embedding=[0.1, 0.2, 0.3],
        ),
        SimpleNamespace(
            index=2,
            embedding=[0.4, 0.5, 0.6],
        ),
    ))

    adapter = OpenAIEmbeddingProvider(
        client=client,
        model="text-embedding-3-small",
        dimensions=3,
    )

    with pytest.raises(
        IndexingValidationError,
        match="indexes",
    ):
        adapter.embed_documents((
            "first",
            "second",
        ))


def test_collection_setup_creates_cosine_collection_and_indexes() -> None:
    client = FakeQdrantClient(
        collection_exists=False,
    )

    ensure_qdrant_collection(
        client,
        collection_name="knowledge",
        vector_dimensions=3,
    )

    assert len(
        client.create_collection_calls
    ) == 1

    vector_config = client.create_collection_calls[
        0
    ]["vectors_config"]

    assert vector_config.size == 3
    assert vector_config.distance == Distance.COSINE

    assert [
        call["field_name"]
        for call in client.create_index_calls
    ] == [
        "access_scope",
        "source_id",
        "source_version",
        "source_checksum_sha256",
        "index_profile",
    ]

    assert all(
        call["field_schema"]
        == PayloadSchemaType.KEYWORD
        for call in client.create_index_calls
    )


def test_collection_setup_rejects_wrong_dimensions() -> None:
    client = FakeQdrantClient(
        vector_dimensions=4,
    )

    with pytest.raises(
        IndexVerificationError,
        match="vector size",
    ):
        ensure_qdrant_collection(
            client,
            collection_name="knowledge",
            vector_dimensions=3,
        )


def test_qdrant_indexes_missing_points_with_exact_identity() -> None:
    client = FakeQdrantClient()

    chunks = (
        make_chunk(chunk_id="chunk-1"),
        make_chunk(chunk_id="chunk-2"),
    )

    result = make_qdrant_adapter(
        client
    ).upsert(
        chunks=chunks,
        vectors=(
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
        ),
    )

    assert result.status is IndexingStatus.INDEXED
    assert result.uploaded_count == 2
    assert len(client.upsert_calls) == 2

    assert {
        payload["index_profile"]
        for payload in client.payloads
    } == {
        INDEX_PROFILE,
    }

    assert (
        "knowledge_base_id"
        not in client.payloads[0]
    )

    scroll_conditions = client.scroll_calls[
        0
    ]["scroll_filter"].must

    assert [
        condition.key
        for condition in scroll_conditions
    ] == [
        "access_scope",
        "source_id",
        "source_version",
        "index_profile",
    ]

    count_conditions = client.count_calls[
        0
    ]["count_filter"].must

    assert [
        condition.key
        for condition in count_conditions
    ] == [
        "access_scope",
        "source_id",
        "source_version",
        "index_profile",
        "source_checksum_sha256",
    ]


def test_qdrant_exact_snapshot_is_idempotent() -> None:
    client = FakeQdrantClient(({
        "chunk_id": "chunk-1",
        "source_checksum_sha256": CHECKSUM,
    },))

    result = make_qdrant_adapter(
        client
    ).upsert(
        chunks=(
            make_chunk(),
        ),
        vectors=(
            (0.1, 0.2, 0.3),
        ),
    )

    assert result.status is IndexingStatus.UNCHANGED
    assert result.uploaded_count == 0
    assert client.upsert_calls == []
    assert client.count_calls == []


def test_qdrant_partial_snapshot_uploads_only_missing_chunks() -> None:
    client = FakeQdrantClient(({
        "chunk_id": "chunk-1",
        "source_checksum_sha256": CHECKSUM,
    },))

    chunks = (
        make_chunk(chunk_id="chunk-1"),
        make_chunk(chunk_id="chunk-2"),
    )

    result = make_qdrant_adapter(
        client
    ).upsert(
        chunks=chunks,
        vectors=(
            (0.1, 0.2, 0.3),
            (0.4, 0.5, 0.6),
        ),
    )

    assert result.uploaded_count == 1
    assert len(client.upsert_calls) == 1

    point = client.upsert_calls[
        0
    ]["points"][0]

    assert point.payload["chunk_id"] == "chunk-2"


@pytest.mark.parametrize(
    (
        "payload",
        "error_type",
    ),
    [
        (
            {
                "chunk_id": "chunk-1",
                "source_checksum_sha256": (
                    "b" * 64
                ),
            },
            SourceVersionConflictError,
        ),
        (
            {
                "chunk_id": "stale-chunk",
                "source_checksum_sha256": (
                    CHECKSUM
                ),
            },
            IndexSnapshotConflictError,
        ),
    ],
)
def test_qdrant_rejects_immutable_snapshot_conflicts(
    payload,
    error_type,
) -> None:
    client = FakeQdrantClient((
        payload,
    ))

    with pytest.raises(error_type):
        make_qdrant_adapter(
            client
        ).upsert(
            chunks=(
                make_chunk(),
            ),
            vectors=(
                (0.1, 0.2, 0.3),
            ),
        )

    assert client.upsert_calls == []


def test_qdrant_inspection_is_read_only() -> None:
    client = FakeQdrantClient(({
        "chunk_id": "chunk-1",
        "source_checksum_sha256": CHECKSUM,
    },))

    chunks = (
        make_chunk(chunk_id="chunk-1"),
        make_chunk(chunk_id="chunk-2"),
    )

    result = make_qdrant_adapter(
        client
    ).inspect_snapshot(chunks)

    assert (
        result.status
        is IndexSnapshotStatus.PARTIAL
    )

    assert result.missing_chunk_ids == (
        "chunk-2",
    )

    assert client.upsert_calls == []
    assert client.count_calls == []


def test_point_identity_changes_with_index_profile() -> None:
    first_client = FakeQdrantClient()
    second_client = FakeQdrantClient()
    chunk = make_chunk()

    make_qdrant_adapter(
        first_client,
        index_profile="profile-a",
    ).upsert(
        chunks=(chunk,),
        vectors=((0.1, 0.2, 0.3),),
    )

    make_qdrant_adapter(
        second_client,
        index_profile="profile-b",
    ).upsert(
        chunks=(chunk,),
        vectors=((0.1, 0.2, 0.3),),
    )

    first_id = first_client.upsert_calls[
        0
    ]["points"][0].id

    second_id = second_client.upsert_calls[
        0
    ]["points"][0].id

    assert first_id != second_id

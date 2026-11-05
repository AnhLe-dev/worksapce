"""Offline tests for exact-scope dense retrieval."""

import math
from types import SimpleNamespace

import pytest

from rag_core.adapters.qdrant import QdrantVectorSearch
from rag_core.ingestion.models import (
    DocumentFormat,
    DocumentRole,
    SourceSpan,
)
from rag_core.retrieval import (
    DenseSourceRetriever,
    KnowledgeHit,
    RetrievalIsolationError,
    RetrievalRequest,
    RetrievalScope,
    RetrievalService,
    RetrievalValidationError,
    SourceRetrievalRequest,
    UnsupportedPayloadSchemaError,
)

CHECKSUM_A = "a" * 64
CHECKSUM_B = "b" * 64
INDEX_PROFILE = "openai:model:dimensions=3:chunk=block:v1"


def make_scope(
    *,
    source_id: str = "policy",
    source_version: str = "1",
    checksum: str = CHECKSUM_A,
    access_scope: str = "tenant-a",
    index_profile: str = INDEX_PROFILE,
) -> RetrievalScope:
    """Build one exact authorized source scope."""

    return RetrievalScope(
        access_scope=access_scope,
        source_id=source_id,
        source_version=source_version,
        source_checksum_sha256=checksum,
        index_profile=index_profile,
    )


def make_hit(
    *,
    chunk_id: str = "chunk-1",
    score: float = 0.8,
    source_id: str = "policy",
    source_version: str = "1",
    checksum: str = CHECKSUM_A,
    access_scope: str = "tenant-a",
    index_profile: str = INDEX_PROFILE,
) -> KnowledgeHit:
    """Build one generic unpaged retrieval result."""

    text = f"Evidence for {chunk_id}"

    return KnowledgeHit(
        chunk_id=chunk_id,
        source_id=source_id,
        source_version=source_version,
        source_checksum_sha256=checksum,
        access_scope=access_scope,
        index_profile=index_profile,
        text=text,
        structure_path=("policy.txt", chunk_id),
        source_spans=(
            SourceSpan(
                block_index=0,
                char_start=0,
                char_end=len(text),
            ),
        ),
        score=score,
        document_format=DocumentFormat.TXT,
        role=DocumentRole.KNOWLEDGE,
        metadata={"owner": "legal"},
    )


def payload_from_hit(hit: KnowledgeHit) -> dict[str, object]:
    """Build a schema-v2 Qdrant payload from one hit."""

    return {
        "schema_version": 2,
        "chunk_id": hit.chunk_id,
        "source_id": hit.source_id,
        "source_version": hit.source_version,
        "source_checksum_sha256": hit.source_checksum_sha256,
        "access_scope": hit.access_scope,
        "index_profile": hit.index_profile,
        "text": hit.text,
        "structure_path": list(hit.structure_path),
        "source_spans": [
            span.model_dump(mode="json")
            for span in hit.source_spans
        ],
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "document_format": hit.document_format.value,
        "role": hit.role.value,
        "metadata": dict(hit.metadata),
    }


class FakeQueryEmbeddingProvider:
    """Return one configured query vector and record input."""

    def __init__(self, vector) -> None:
        self.vector = vector
        self.queries = []

    def embed_query(self, text: str):
        self.queries.append(text)
        return self.vector


class FakeVectorSearch:
    """Return configured hits and record exact search arguments."""

    def __init__(self, hits=()) -> None:
        self.hits = hits
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.hits


class FakeSourceRetriever:
    """Return hits configured for each source ID."""

    def __init__(self, hits_by_source) -> None:
        self.hits_by_source = hits_by_source
        self.requests = []

    def retrieve_source(self, request):
        self.requests.append(request)
        return self.hits_by_source.get(
            request.scope.source_id,
            (),
        )


class FakeQdrantQueryClient:
    """Return configured Qdrant-compatible points."""

    def __init__(self, points=()) -> None:
        self.points = points
        self.calls = []

    def query_points(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(points=self.points)


def make_dense_retriever(
    *,
    vector=(0.1, 0.2, 0.3),
    hits=(),
):
    """Build a dense retriever and expose its fake dependencies."""

    embedding_provider = FakeQueryEmbeddingProvider(vector)
    vector_search = FakeVectorSearch(hits)
    retriever = DenseSourceRetriever(
        embedding_provider=embedding_provider,
        vector_search=vector_search,
        expected_dimensions=3,
    )

    return retriever, embedding_provider, vector_search


def test_dense_retrieval_validates_vector_and_ranks_deterministically() -> None:
    scope = make_scope()
    lower = make_hit(chunk_id="chunk-b", score=0.7)
    higher = make_hit(chunk_id="chunk-a", score=0.9)
    duplicate = make_hit(chunk_id="chunk-a", score=0.8)
    retriever, embedding_provider, vector_search = make_dense_retriever(
        hits=(lower, duplicate, higher),
    )
    request = SourceRetrievalRequest(
        query="  What is required?  ",
        scope=scope,
        limit=5,
    )

    results = retriever.retrieve_source(request)

    assert embedding_provider.queries == ["What is required?"]
    assert vector_search.calls == [{
        "query_vector": (0.1, 0.2, 0.3),
        "scope": scope,
        "limit": 5,
    }]
    assert [hit.chunk_id for hit in results] == [
        "chunk-a",
        "chunk-b",
    ]
    assert results[0].score == 0.9


def test_dense_retrieval_accepts_empty_hits() -> None:
    retriever, _, _ = make_dense_retriever()

    results = retriever.retrieve_source(
        SourceRetrievalRequest(
            query="Unknown topic",
            scope=make_scope(),
            limit=3,
        )
    )

    assert results == ()


@pytest.mark.parametrize(
    "vector",
    [
        (),
        (0.1, 0.2),
        (0.1, math.inf, 0.3),
        (0.1, math.nan, 0.3),
        (0.1, "not-a-number", 0.3),
    ],
)
def test_dense_retrieval_rejects_invalid_query_vectors(vector) -> None:
    retriever, _, vector_search = make_dense_retriever(
        vector=vector,
    )

    with pytest.raises(RetrievalValidationError):
        retriever.retrieve_source(
            SourceRetrievalRequest(
                query="Question",
                scope=make_scope(),
                limit=3,
            )
        )

    assert vector_search.calls == []


@pytest.mark.parametrize(
    "outside_hit",
    [
        make_hit(access_scope="tenant-b"),
        make_hit(source_id="other-policy"),
        make_hit(source_version="2"),
        make_hit(checksum=CHECKSUM_B),
        make_hit(index_profile="different-profile"),
    ],
)
def test_dense_retrieval_rejects_hits_outside_exact_scope(
    outside_hit,
) -> None:
    retriever, _, _ = make_dense_retriever(
        hits=(outside_hit,),
    )

    with pytest.raises(RetrievalIsolationError):
        retriever.retrieve_source(
            SourceRetrievalRequest(
                query="Question",
                scope=make_scope(),
                limit=3,
            )
        )


def test_retrieval_service_merges_sources_with_global_ranking() -> None:
    policy_scope = make_scope(source_id="policy")
    handbook_scope = make_scope(
        source_id="handbook",
        checksum=CHECKSUM_B,
    )
    source_retriever = FakeSourceRetriever({
        "policy": (
            make_hit(
                chunk_id="policy-1",
                source_id="policy",
                score=0.7,
            ),
        ),
        "handbook": (
            make_hit(
                chunk_id="handbook-1",
                source_id="handbook",
                checksum=CHECKSUM_B,
                score=0.9,
            ),
            make_hit(
                chunk_id="handbook-2",
                source_id="handbook",
                checksum=CHECKSUM_B,
                score=0.6,
            ),
        ),
    })
    service = RetrievalService(
        source_retriever=source_retriever,
    )

    results = service.retrieve(
        RetrievalRequest(
            query="Question",
            scopes=(policy_scope, handbook_scope),
            limit=2,
        )
    )

    assert [hit.chunk_id for hit in results] == [
        "handbook-1",
        "policy-1",
    ]
    assert len(source_retriever.requests) == 2
    assert all(
        request.limit == 2
        for request in source_retriever.requests
    )


def test_retrieval_service_rejects_source_retriever_scope_escape() -> None:
    source_retriever = FakeSourceRetriever({
        "policy": (
            make_hit(checksum=CHECKSUM_B),
        ),
    })
    service = RetrievalService(
        source_retriever=source_retriever,
    )

    with pytest.raises(RetrievalIsolationError):
        service.retrieve(
            RetrievalRequest(
                query="Question",
                scopes=(make_scope(),),
                limit=3,
            )
        )


def test_qdrant_search_uses_exact_five_field_filter() -> None:
    expected_hit = make_hit()
    point = SimpleNamespace(
        id="point-1",
        payload=payload_from_hit(expected_hit),
        score=expected_hit.score,
    )
    client = FakeQdrantQueryClient((point,))
    search = QdrantVectorSearch(
        client=client,
        collection_name="knowledge",
        vector_dimensions=3,
    )

    results = search.search(
        query_vector=(0.1, 0.2, 0.3),
        scope=make_scope(),
        limit=5,
    )

    assert results == (expected_hit,)
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["query"] == [0.1, 0.2, 0.3]
    assert call["with_payload"] is True
    assert call["with_vectors"] is False
    assert [
        (condition.key, condition.match.value)
        for condition in call["query_filter"].must
    ] == [
        ("access_scope", "tenant-a"),
        ("source_id", "policy"),
        ("source_version", "1"),
        ("source_checksum_sha256", CHECKSUM_A),
        ("index_profile", INDEX_PROFILE),
    ]


@pytest.mark.parametrize(
    "schema_version",
    [None, 1, 999, 2.0, "2", True],
)
def test_qdrant_search_rejects_unknown_payload_schema(
    schema_version,
) -> None:
    payload = payload_from_hit(make_hit())
    payload["schema_version"] = schema_version
    point = SimpleNamespace(
        id="point-1",
        payload=payload,
        score=0.8,
    )
    search = QdrantVectorSearch(
        client=FakeQdrantQueryClient((point,)),
        collection_name="knowledge",
        vector_dimensions=3,
    )

    with pytest.raises(UnsupportedPayloadSchemaError):
        search.search(
            query_vector=(0.1, 0.2, 0.3),
            scope=make_scope(),
            limit=5,
        )


def test_qdrant_search_rejects_missing_payload() -> None:
    point = SimpleNamespace(
        id="point-1",
        payload=None,
        score=0.8,
    )
    search = QdrantVectorSearch(
        client=FakeQdrantQueryClient((point,)),
        collection_name="knowledge",
        vector_dimensions=3,
    )

    with pytest.raises(RetrievalValidationError, match="no payload"):
        search.search(
            query_vector=(0.1, 0.2, 0.3),
            scope=make_scope(),
            limit=5,
        )

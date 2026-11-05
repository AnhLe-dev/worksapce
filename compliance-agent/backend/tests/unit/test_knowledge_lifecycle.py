"""Offline tests for PostgreSQL-owned knowledge lifecycle semantics."""

from pathlib import Path
from uuid import uuid4

import pytest
from rag_core.chunking import BlockChunkingStrategy
from rag_core.indexing import (
    DocumentIndexingService,
    IndexingResult,
    IndexingStatus,
)
from rag_core.ingestion.extractors.text import TextDocumentExtractor
from rag_core.ingestion.parsing import FlatStructureParser
from rag_core.ingestion.service import DocumentProcessor
from rag_core.retrieval.models import KnowledgeHit
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.processing_service import ContractLifecycleService
from app.contracts.repository import ContractRepository
from app.documents.ingestion_service import StoredObjectIngestionService
from app.documents.models import StoredObject
from app.documents.repository import StoredObjectRepository
from app.knowledge.db_models import KnowledgeBaseMembershipRecord
from app.knowledge.errors import (
    KnowledgeBaseNotFoundError,
    KnowledgeIsolationError,
    KnowledgeReferenceNotFoundError,
)
from app.knowledge.indexing_service import KnowledgeIndexingService
from app.knowledge.ingestion_service import KnowledgeIngestionService
from app.knowledge.models import (
    KnowledgeRetrievalRequest,
    MembershipStatus,
    SnapshotStatus,
)
from app.knowledge.reference_service import KnowledgeReferenceService
from app.knowledge.registry import KnowledgeRegistry
from app.knowledge.repository import KnowledgeRepository
from app.knowledge.retrieval_service import KnowledgeRetrievalService
from tests.fakes import FakeObjectStorage

INDEX_PROFILE = "fake:dimensions=3:block:v1"


class FakeEmbeddingProvider:
    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, float(index), 0.5) for index, _ in enumerate(texts))


class FakeVectorIndex:
    def __init__(self) -> None:
        self.call_count = 0

    def upsert(self, *, chunks, vectors) -> IndexingResult:
        self.call_count += 1
        return IndexingResult(
            collection_name="knowledge_test",
            source_id=chunks[0].source_id,
            source_version=chunks[0].source_version,
            source_checksum_sha256=chunks[0].source_checksum_sha256,
            access_scope=chunks[0].access_scope,
            status=IndexingStatus.INDEXED,
            chunk_count=len(chunks),
            uploaded_count=len(chunks),
            stored_count=len(chunks),
            vector_dimensions=len(vectors[0]),
        )


class RecordingRetriever:
    def __init__(self) -> None:
        self.requests = []

    def retrieve(self, request):
        self.requests.append(request)
        return ()


class EscapingRetriever:
    def retrieve(self, request):
        scope = request.scopes[0]
        return (
            KnowledgeHit(
                chunk_id="escaped-chunk",
                source_id=scope.source_id,
                source_version=scope.source_version,
                source_checksum_sha256=scope.source_checksum_sha256,
                access_scope="system-b",
                index_profile=scope.index_profile,
                text="This hit escaped the selected knowledge base.",
                structure_path=("root",),
                source_spans=(
                    {
                        "block_index": 0,
                        "char_start": 0,
                        "char_end": 4,
                    },
                ),
                score=0.9,
            ),
        )


def make_processor() -> DocumentProcessor:
    return DocumentProcessor(
        extractors=(TextDocumentExtractor(),),
        parser=FlatStructureParser(),
        chunking_strategy=BlockChunkingStrategy(max_characters=500),
    )


def create_stored_object(
    session: Session,
    tmp_path: Path,
) -> tuple[Path, StoredObject]:
    path = tmp_path / "policy.txt"
    path.write_text(
        "Refunds require a receipt.\n\nRequests expire after thirty days.",
        encoding="utf-8",
    )
    stored_object = StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
        storage=FakeObjectStorage(),
        max_upload_bytes=4096,
    ).ingest(path).stored_object
    return path, stored_object


def register_knowledge(
    session: Session,
    tmp_path: Path,
):
    path, stored_object = create_stored_object(session, tmp_path)
    repository = KnowledgeRepository(session)
    service = KnowledgeIngestionService(
        repository=repository,
        processor=make_processor(),
    )
    knowledge_base = service.create_base(
        name="Refund policies",
        trusted_scope="system-a",
    )
    registration = service.register_document(
        knowledge_base_id=knowledge_base.knowledge_base_id,
        stored_object=stored_object,
        source_id="refund-policy",
        source_version="1",
        filename=path.name,
        index_profile=INDEX_PROFILE,
        trusted_scope="system-a",
    )
    return (
        path,
        stored_object,
        repository,
        service,
        knowledge_base,
        registration,
    )


def test_contract_and_knowledge_share_blob_not_domain_identity(
    session: Session,
    tmp_path: Path,
) -> None:
    (
        _,
        stored_object,
        _,
        _,
        _,
        registration,
    ) = register_knowledge(session, tmp_path)
    contract, _ = ContractLifecycleService(
        ContractRepository(session)
    ).create(
        stored_object=stored_object,
        filename="contract.txt",
        trusted_scope="system-a",
        source_version="1",
    )

    assert contract.stored_object_id == registration.document.stored_object_id
    assert contract.contract_id != registration.document.knowledge_document_id
    assert contract.role == "contract"
    assert registration.document.role == "knowledge"


def test_database_rejects_knowledge_role_change(
    session: Session,
    tmp_path: Path,
) -> None:
    _, _, repository, _, _, _registration = register_knowledge(
        session,
        tmp_path,
    )
    record = repository.get_document_record_by_source(
        source_id="refund-policy",
        source_version="1",
        access_scope="system-a",
    )
    assert record is not None
    record.role = "contract"

    with pytest.raises(IntegrityError):
        session.flush()


def test_database_rejects_cross_scope_membership(
    session: Session,
    tmp_path: Path,
) -> None:
    *_, knowledge_base, registration = register_knowledge(session, tmp_path)
    session.add(
        KnowledgeBaseMembershipRecord(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            snapshot_id=registration.snapshot.snapshot_id,
            access_scope="system-b",
            membership_version=2,
            status=MembershipStatus.PENDING,
        )
    )

    with pytest.raises(IntegrityError):
        session.flush()


def test_two_bases_reuse_one_snapshot_and_one_vector_write(
    session: Session,
    tmp_path: Path,
) -> None:
    (
        path,
        stored_object,
        repository,
        ingestion,
        first_base,
        registration,
    ) = register_knowledge(session, tmp_path)
    second_base = ingestion.create_base(
        name="Customer support policies",
        trusted_scope="system-a",
    )
    second_registration = ingestion.register_document(
        knowledge_base_id=second_base.knowledge_base_id,
        stored_object=stored_object,
        source_id="refund-policy",
        source_version="1",
        filename=path.name,
        index_profile=INDEX_PROFILE,
        trusted_scope="system-a",
    )
    vector_index = FakeVectorIndex()
    indexing = KnowledgeIndexingService(
        repository=repository,
        ingestion=ingestion,
        indexer=DocumentIndexingService(
            embedding_provider=FakeEmbeddingProvider(),
            vector_index=vector_index,
            expected_dimensions=3,
        ),
    )
    first_outcome = indexing.index(
        snapshot_id=registration.snapshot.snapshot_id,
        trusted_scope="system-a",
        file_path=path,
    )
    second_outcome = indexing.index(
        snapshot_id=second_registration.snapshot.snapshot_id,
        trusted_scope="system-a",
        file_path=path,
    )

    assert first_base.knowledge_base_id != second_base.knowledge_base_id
    assert registration.snapshot.snapshot_id == (
        second_registration.snapshot.snapshot_id
    )
    assert second_registration.reused_snapshot is True
    assert first_outcome.reused_existing is False
    assert second_outcome.reused_existing is True
    assert vector_index.call_count == 1
    assert first_outcome.snapshot.status is SnapshotStatus.READY


def test_registry_resolves_exact_ready_snapshot_scope(
    session: Session,
    tmp_path: Path,
) -> None:
    (
        path,
        _,
        repository,
        ingestion,
        knowledge_base,
        registration,
    ) = register_knowledge(session, tmp_path)
    KnowledgeIndexingService(
        repository=repository,
        ingestion=ingestion,
        indexer=DocumentIndexingService(
            embedding_provider=FakeEmbeddingProvider(),
            vector_index=FakeVectorIndex(),
            expected_dimensions=3,
        ),
    ).index(
        snapshot_id=registration.snapshot.snapshot_id,
        trusted_scope="system-a",
        file_path=path,
    )
    resolved = KnowledgeRegistry(repository).resolve(
        knowledge_base_id=knowledge_base.knowledge_base_id,
        trusted_scope="system-a",
    )

    assert len(resolved.scopes) == 1
    assert len(resolved.snapshots) == 1
    assert resolved.snapshots[0].snapshot_id == (
        registration.snapshot.snapshot_id
    )
    assert resolved.scopes[0].source_id == "refund-policy"
    assert resolved.scopes[0].source_checksum_sha256 == (
        registration.snapshot.source_checksum_sha256
    )
    assert resolved.scopes[0].index_profile == INDEX_PROFILE

    with pytest.raises(KnowledgeBaseNotFoundError):
        KnowledgeRegistry(repository).resolve(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            trusted_scope="system-b",
        )


def test_reference_service_is_scope_isolated(
    session: Session,
    tmp_path: Path,
) -> None:
    (
        path,
        _,
        repository,
        ingestion,
        _,
        registration,
    ) = register_knowledge(session, tmp_path)
    KnowledgeIndexingService(
        repository=repository,
        ingestion=ingestion,
        indexer=DocumentIndexingService(
            embedding_provider=FakeEmbeddingProvider(),
            vector_index=FakeVectorIndex(),
            expected_dimensions=3,
        ),
    ).index(
        snapshot_id=registration.snapshot.snapshot_id,
        trusted_scope="system-a",
        file_path=path,
    )
    prepared = ingestion.prepare_snapshot(
        snapshot_id=registration.snapshot.snapshot_id,
        trusted_scope="system-a",
        file_path=path,
    )
    reference_id = prepared.reference_units[0].reference_unit_id
    service = KnowledgeReferenceService(repository)

    assert service.get_reference(
        reference_unit_id=reference_id,
        authorized_snapshot_ids=(registration.snapshot.snapshot_id,),
        trusted_scope="system-a",
    ).text == "Refunds require a receipt."

    with pytest.raises(KnowledgeReferenceNotFoundError):
        service.get_reference(
            reference_unit_id=reference_id,
            authorized_snapshot_ids=(registration.snapshot.snapshot_id,),
            trusted_scope="system-b",
        )

    with pytest.raises(KnowledgeReferenceNotFoundError):
        service.get_reference(
            reference_unit_id=reference_id,
            authorized_snapshot_ids=(uuid4(),),
            trusted_scope="system-a",
        )

    references = service.list_references_for_evidence(
        snapshot_id=registration.snapshot.snapshot_id,
        evidence_spans=prepared.reference_units[0].source_spans,
        authorized_snapshot_ids=(registration.snapshot.snapshot_id,),
        trusted_scope="system-a",
    )
    assert tuple(unit.reference_unit_id for unit in references) == (
        reference_id,
    )


def test_retrieval_resolves_authority_before_zero_hit_search(
    session: Session,
    tmp_path: Path,
) -> None:
    (
        path,
        _,
        repository,
        ingestion,
        knowledge_base,
        registration,
    ) = register_knowledge(session, tmp_path)
    KnowledgeIndexingService(
        repository=repository,
        ingestion=ingestion,
        indexer=DocumentIndexingService(
            embedding_provider=FakeEmbeddingProvider(),
            vector_index=FakeVectorIndex(),
            expected_dimensions=3,
        ),
    ).index(
        snapshot_id=registration.snapshot.snapshot_id,
        trusted_scope="system-a",
        file_path=path,
    )
    retriever = RecordingRetriever()
    result = KnowledgeRetrievalService(
        registry=KnowledgeRegistry(repository),
        retriever=retriever,
    ).retrieve(
        KnowledgeRetrievalRequest(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            access_scope="system-a",
            query="What is the refund deadline?",
            limit=5,
        )
    )

    assert result.hits == ()
    assert result.snapshots[0].snapshot_id == (
        registration.snapshot.snapshot_id
    )
    assert len(retriever.requests) == 1
    assert retriever.requests[0].scopes == result.scopes
    assert result.scopes[0].source_checksum_sha256 == (
        registration.snapshot.source_checksum_sha256
    )


def test_retrieval_rejects_hit_outside_resolved_knowledge_base(
    session: Session,
    tmp_path: Path,
) -> None:
    (
        path,
        _,
        repository,
        ingestion,
        knowledge_base,
        registration,
    ) = register_knowledge(session, tmp_path)
    KnowledgeIndexingService(
        repository=repository,
        ingestion=ingestion,
        indexer=DocumentIndexingService(
            embedding_provider=FakeEmbeddingProvider(),
            vector_index=FakeVectorIndex(),
            expected_dimensions=3,
        ),
    ).index(
        snapshot_id=registration.snapshot.snapshot_id,
        trusted_scope="system-a",
        file_path=path,
    )
    service = KnowledgeRetrievalService(
        registry=KnowledgeRegistry(repository),
        retriever=EscapingRetriever(),
    )

    with pytest.raises(
        KnowledgeIsolationError,
        match="outside the resolved KB",
    ):
        service.retrieve(
            KnowledgeRetrievalRequest(
                knowledge_base_id=knowledge_base.knowledge_base_id,
                access_scope="system-a",
                query="What is the refund deadline?",
                limit=5,
            )
        )

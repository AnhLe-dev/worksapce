"""Offline E2E tests for committed claims and generic worker handlers."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from rag_core.indexing import (
    DocumentIndexingService,
    IndexingResult,
    IndexingStatus,
)
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.pool import StaticPool

from app.contracts.models import ContractStatus
from app.contracts.processing_service import ContractLifecycleService
from app.contracts.repository import ContractRepository, to_contract
from app.core.config import Settings
from app.db.base import Base
from app.db.session import create_session_factory, session_scope
from app.documents.ingestion_service import StoredObjectIngestionService
from app.documents.repository import StoredObjectRepository
from app.jobs.models import JobStatus
from app.jobs.reconciler import JobReconciler
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobService
from app.knowledge.db_models import KnowledgeReferenceUnitRecord
from app.knowledge.index_profile import build_index_profile
from app.knowledge.ingestion_service import KnowledgeIngestionService
from app.knowledge.models import MembershipStatus, SnapshotStatus
from app.knowledge.repository import KnowledgeRepository, to_snapshot
from app.worker.handlers import DomainFailureRecorder
from app.worker.runtime import WorkerRuntime, create_document_processor
from tests.fakes import FakeObjectStorage


class FakeEmbeddingProvider:
    def embed_documents(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple((1.0, float(index), 0.5) for index, _ in enumerate(texts))


class FakeVectorIndex:
    def upsert(self, *, chunks, vectors) -> IndexingResult:
        return IndexingResult(
            collection_name="worker_test",
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


def create_runtime():
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def enable_foreign_keys(dbapi_connection, connection_record) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    settings = Settings(
        _env_file=None,
        database_url="sqlite+pysqlite://",
        system_access_scope="worker-test",
        job_lease_seconds=30,
        job_max_attempts=2,
        embedding_dimensions=3,
        max_chunk_characters=500,
    )
    session_factory = create_session_factory(engine)
    storage = FakeObjectStorage()
    indexer = DocumentIndexingService(
        embedding_provider=FakeEmbeddingProvider(),
        vector_index=FakeVectorIndex(),
        expected_dimensions=3,
    )
    runtime = WorkerRuntime(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        storage=storage,
        processor=create_document_processor(settings),
        indexer_factory=lambda: indexer,
    )
    return runtime, session_factory, storage, settings


def create_stored_object(
    path: Path,
    session,
    storage: FakeObjectStorage,
):
    return StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
        storage=storage,
        max_upload_bytes=4096,
    ).ingest(path).stored_object


def create_job_service(session, settings: Settings) -> JobService:
    return JobService(
        jobs=JobRepository(session),
        outbox=OutboxRepository(session),
        lease_seconds=settings.job_lease_seconds,
        max_attempts=settings.job_max_attempts,
    )


def test_contract_job_downloads_processes_and_commits(
    tmp_path: Path,
) -> None:
    runtime, session_factory, storage, settings = create_runtime()
    source = tmp_path / "contract.txt"
    source.write_text(
        "Delivery requires a receipt.\n\nPayment is due in thirty days.",
        encoding="utf-8",
    )

    with session_scope(session_factory) as session:
        stored = create_stored_object(source, session, storage)
        contract, _ = ContractLifecycleService(
            ContractRepository(session)
        ).create(
            stored_object=stored,
            filename=source.name,
            trusted_scope=settings.system_access_scope,
            source_version="1",
        )
        job = create_job_service(
            session,
            settings,
        ).create_contract_processing(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            expected_revision=0,
            idempotency_key="contract-e2e",
        ).job

    result = runtime.execute(job.job_id)
    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.result_payload == {
        "contract_id": str(contract.contract_id),
        "processing_revision": 1,
        "review_unit_count": 2,
    }

    with session_scope(session_factory) as session:
        repository = ContractRepository(session)
        record = repository.get_record(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
        )
        assert record is not None
        assert to_contract(record).status is ContractStatus.READY
        assert len(
            repository.list_review_units(
                contract_id=contract.contract_id,
                access_scope=settings.system_access_scope,
                processing_revision=1,
            )
        ) == 2

    runtime.close()


def test_knowledge_job_indexes_and_activates_membership(
    tmp_path: Path,
) -> None:
    runtime, session_factory, storage, settings = create_runtime()
    source = tmp_path / "policy.txt"
    source.write_text(
        "Refunds require a receipt.\n\nRequests expire after thirty days.",
        encoding="utf-8",
    )

    with session_scope(session_factory) as session:
        stored = create_stored_object(source, session, storage)
        repository = KnowledgeRepository(session)
        ingestion = KnowledgeIngestionService(repository=repository)
        knowledge_base = ingestion.create_base(
            name="Refund policy",
            trusted_scope=settings.system_access_scope,
        )
        registration = ingestion.register_document(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            stored_object=stored,
            source_id="refund-policy",
            source_version="1",
            filename=source.name,
            index_profile=build_index_profile(settings),
            trusted_scope=settings.system_access_scope,
        )
        job = create_job_service(
            session,
            settings,
        ).create_knowledge_indexing(
            snapshot_id=registration.snapshot.snapshot_id,
            access_scope=settings.system_access_scope,
            idempotency_key="knowledge-e2e",
        ).job

    result = runtime.execute(job.job_id)
    assert result is not None
    assert result.status is JobStatus.SUCCEEDED
    assert result.result_payload is not None
    assert result.result_payload["chunk_count"] == 1

    with session_scope(session_factory) as session:
        repository = KnowledgeRepository(session)
        record = repository.get_snapshot_record(
            snapshot_id=registration.snapshot.snapshot_id,
            access_scope=settings.system_access_scope,
        )
        assert record is not None
        assert to_snapshot(record).status is SnapshotStatus.READY
        memberships = repository.list_memberships(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            access_scope=settings.system_access_scope,
        )
        assert memberships[0][0].status is MembershipStatus.READY
        reference_count = session.scalar(
            select(func.count())
            .select_from(KnowledgeReferenceUnitRecord)
            .where(
                KnowledgeReferenceUnitRecord.snapshot_id
                == registration.snapshot.snapshot_id
            )
        )
        assert reference_count == 2

    runtime.close()


def test_review_job_returns_safe_domain_outcome_without_policy(
    tmp_path: Path,
) -> None:
    runtime, session_factory, storage, settings = create_runtime()
    contract_path = tmp_path / "contract.txt"
    contract_path.write_text(
        "Refund requests must include a receipt.",
        encoding="utf-8",
    )
    policy_path = tmp_path / "policy.txt"
    policy_path.write_text(
        "A receipt is required for every refund request.",
        encoding="utf-8",
    )

    with session_scope(session_factory) as session:
        contract_stored = create_stored_object(
            contract_path,
            session,
            storage,
        )
        contract, _ = ContractLifecycleService(
            ContractRepository(session)
        ).create(
            stored_object=contract_stored,
            filename=contract_path.name,
            trusted_scope=settings.system_access_scope,
            source_version="1",
        )
        processing_job = create_job_service(
            session,
            settings,
        ).create_contract_processing(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            expected_revision=0,
            idempotency_key="review-contract-processing",
        ).job

    processed = runtime.execute(processing_job.job_id)
    assert processed is not None
    assert processed.status is JobStatus.SUCCEEDED

    with session_scope(session_factory) as session:
        policy_stored = create_stored_object(policy_path, session, storage)
        knowledge = KnowledgeIngestionService(
            repository=KnowledgeRepository(session)
        )
        knowledge_base = knowledge.create_base(
            name="Policy without applicability proof",
            trusted_scope=settings.system_access_scope,
        )
        registration = knowledge.register_document(
            knowledge_base_id=knowledge_base.knowledge_base_id,
            stored_object=policy_stored,
            source_id="refund-policy",
            source_version="1",
            filename=policy_path.name,
            index_profile=build_index_profile(settings),
            trusted_scope=settings.system_access_scope,
        )
        indexing_job = create_job_service(
            session,
            settings,
        ).create_knowledge_indexing(
            snapshot_id=registration.snapshot.snapshot_id,
            access_scope=settings.system_access_scope,
            idempotency_key="review-policy-indexing",
        ).job

    indexed = runtime.execute(indexing_job.job_id)
    assert indexed is not None
    assert indexed.status is JobStatus.SUCCEEDED

    with session_scope(session_factory) as session:
        units = ContractRepository(session).list_review_units(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            processing_revision=1,
        )
        review_job = create_job_service(
            session,
            settings,
        ).create_contract_review(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            knowledge_base_id=knowledge_base.knowledge_base_id,
            expected_processing_revision=1,
            review_unit_ids=tuple(
                unit.review_unit_id for unit in units
            ),
            question="Does the selected policy apply?",
            idempotency_key="review-safe-outcome",
        ).job

    reviewed = runtime.execute(review_job.job_id)
    assert reviewed is not None
    assert reviewed.status is JobStatus.SUCCEEDED
    assert reviewed.result_payload is not None
    assert (
        reviewed.result_payload["outcome"]
        == "APPLICABILITY_UNDETERMINED"
    )

    with session_scope(session_factory) as session:
        contract_record = ContractRepository(session).get_record(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
        )
        assert contract_record is not None
        assert contract_record.status is ContractStatus.READY

    runtime.close()


def test_permanent_worker_failure_updates_job_and_contract(
    tmp_path: Path,
) -> None:
    runtime, session_factory, storage, settings = create_runtime()
    source = tmp_path / "contract.txt"
    source.write_text("One valid paragraph.", encoding="utf-8")

    with session_scope(session_factory) as session:
        stored = create_stored_object(source, session, storage)
        contract, _ = ContractLifecycleService(
            ContractRepository(session)
        ).create(
            stored_object=stored,
            filename=source.name,
            trusted_scope=settings.system_access_scope,
            source_version="1",
        )
        job = create_job_service(
            session,
            settings,
        ).create_contract_processing(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            expected_revision=99,
            idempotency_key="invalid-revision",
        ).job

    result = runtime.execute(job.job_id)
    assert result is not None
    assert result.status is JobStatus.FAILED

    with session_scope(session_factory) as session:
        record = ContractRepository(session).get_record(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
        )
        assert record is not None
        assert record.status is ContractStatus.FAILED
        assert record.error_code == "contract_processing_invalid"

    runtime.close()


def test_expired_final_attempt_fails_job_and_contract(
    tmp_path: Path,
) -> None:
    runtime, session_factory, storage, settings = create_runtime()
    source = tmp_path / "contract.txt"
    source.write_text("One valid paragraph.", encoding="utf-8")
    now = datetime(2026, 11, 5, 9, 0, tzinfo=UTC)

    with session_scope(session_factory) as session:
        stored = create_stored_object(source, session, storage)
        contract, _ = ContractLifecycleService(
            ContractRepository(session)
        ).create(
            stored_object=stored,
            filename=source.name,
            trusted_scope=settings.system_access_scope,
            source_version="1",
        )
        service = JobService(
            jobs=JobRepository(session),
            outbox=OutboxRepository(session),
            lease_seconds=30,
            max_attempts=1,
        )
        job = service.create_contract_processing(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
            expected_revision=0,
            idempotency_key="expired-final-attempt",
            now=now,
        ).job
        assert service.claim(job.job_id, now=now) is not None

    with session_scope(session_factory) as session:
        service = JobService(
            jobs=JobRepository(session),
            outbox=OutboxRepository(session),
            lease_seconds=30,
            max_attempts=1,
        )
        report = JobReconciler(
            jobs=service,
            outbox=OutboxRepository(session),
            failure_handler=DomainFailureRecorder(session),
        ).reconcile(now=now + timedelta(seconds=31))

    assert report.failed_jobs == 1

    with session_scope(session_factory) as session:
        record = ContractRepository(session).get_record(
            contract_id=contract.contract_id,
            access_scope=settings.system_access_scope,
        )
        assert record is not None
        assert record.status is ContractStatus.FAILED
        assert record.error_code == "job_attempts_exhausted"

    runtime.close()

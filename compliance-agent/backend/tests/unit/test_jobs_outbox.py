"""Offline crash, duplicate, lease, and outbox recovery tests."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.contracts.processing_service import ContractLifecycleService
from app.contracts.repository import ContractRepository
from app.documents.ingestion_service import StoredObjectIngestionService
from app.documents.repository import StoredObjectRepository
from app.jobs.db_models import OutboxEventRecord, TechnicalJobRecord
from app.jobs.models import JobKind, JobStage, JobStatus
from app.jobs.outbox_dispatcher import (
    MessagePublishError,
    OutboxDispatcher,
)
from app.jobs.outbox_models import OutboxStatus
from app.jobs.reconciler import JobReconciler
from app.jobs.repository import JobRepository, OutboxRepository
from app.jobs.service import JobAttemptOwnershipError, JobService
from app.knowledge.db_models import (
    KnowledgeDocumentRecord,
    KnowledgeSourceSnapshotRecord,
)
from app.knowledge.models import (
    KnowledgeDocumentStatus,
    SnapshotStatus,
)
from app.worker.dispatcher import WorkerDispatcher
from tests.fakes import FakeObjectStorage

NOW = datetime(2026, 11, 5, 8, 0, tzinfo=UTC)


class RecordingPublisher:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.messages: list[
            tuple[str, dict[str, object], str]
        ] = []

    def publish(
        self,
        *,
        topic: str,
        payload: dict[str, object],
        message_id: str,
    ) -> None:
        if self.fail:
            raise MessagePublishError("broker unavailable")

        self.messages.append((topic, payload, message_id))


class CountingHandler:
    def __init__(self) -> None:
        self.call_count = 0

    def __call__(self, claim) -> dict[str, object]:
        self.call_count += 1
        return {"handled_job_id": str(claim.job_id)}


def make_job_service(
    session: Session,
    *,
    max_attempts: int = 3,
    lease_seconds: int = 30,
) -> tuple[JobService, OutboxRepository]:
    outbox = OutboxRepository(session)
    service = JobService(
        jobs=JobRepository(session),
        outbox=outbox,
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
    )
    return service, outbox


def create_contract_target(
    session: Session,
    tmp_path: Path,
):
    path = tmp_path / "contract.txt"
    path.write_text("Generic contract terms.", encoding="utf-8")
    stored_object = StoredObjectIngestionService(
        repository=StoredObjectRepository(session),
        storage=FakeObjectStorage(),
        max_upload_bytes=4096,
    ).ingest(path).stored_object
    contract, _ = ContractLifecycleService(
        ContractRepository(session)
    ).create(
        stored_object=stored_object,
        filename=path.name,
        trusted_scope="system-a",
        source_version="1",
    )
    return stored_object, contract


def create_contract_job(
    session: Session,
    tmp_path: Path,
    *,
    max_attempts: int = 3,
) -> tuple[JobService, OutboxRepository, object]:
    _, contract = create_contract_target(session, tmp_path)
    service, outbox = make_job_service(
        session,
        max_attempts=max_attempts,
    )
    created = service.create_contract_processing(
        contract_id=contract.contract_id,
        access_scope="system-a",
        expected_revision=0,
        idempotency_key="contract-upload-1",
        now=NOW,
    )
    return service, outbox, created.job


def table_count(session: Session, model: type) -> int:
    return int(
        session.scalar(select(func.count()).select_from(model)) or 0
    )


def test_job_and_outbox_rollback_together(
    session: Session,
    tmp_path: Path,
) -> None:
    _, contract = create_contract_target(session, tmp_path)
    service, _ = make_job_service(session)
    savepoint = session.begin_nested()
    service.create_contract_processing(
        contract_id=contract.contract_id,
        access_scope="system-a",
        expected_revision=0,
        idempotency_key="rollback-me",
        now=NOW,
    )
    savepoint.rollback()

    assert table_count(session, TechnicalJobRecord) == 0
    assert table_count(session, OutboxEventRecord) == 0


def test_idempotent_create_has_one_job_and_one_event(
    session: Session,
    tmp_path: Path,
) -> None:
    service, _, job = create_contract_job(session, tmp_path)
    duplicate = service.create_contract_processing(
        contract_id=job.contract_id,
        access_scope="system-a",
        expected_revision=0,
        idempotency_key="contract-upload-1",
        now=NOW,
    )

    assert duplicate.created is False
    assert duplicate.job.job_id == job.job_id
    assert table_count(session, TechnicalJobRecord) == 1
    assert table_count(session, OutboxEventRecord) == 1


def test_publish_failure_keeps_pending_event(
    session: Session,
    tmp_path: Path,
) -> None:
    _, outbox, job = create_contract_job(session, tmp_path)
    report = OutboxDispatcher(
        repository=outbox,
        publisher=RecordingPublisher(fail=True),
        relay_id="relay-1",
        lease_seconds=30,
        batch_size=10,
    ).dispatch(now=NOW)
    event = session.scalar(select(OutboxEventRecord))

    assert report.failed == 1
    assert event is not None
    assert event.status is OutboxStatus.PENDING
    assert event.publish_attempts == 1
    assert event.last_error == "MessagePublishError"
    assert job.status is JobStatus.QUEUED


def test_published_message_contains_only_job_id(
    session: Session,
    tmp_path: Path,
) -> None:
    _, outbox, job = create_contract_job(session, tmp_path)
    publisher = RecordingPublisher()
    report = OutboxDispatcher(
        repository=outbox,
        publisher=publisher,
        relay_id="relay-1",
        lease_seconds=30,
        batch_size=10,
    ).dispatch(now=NOW)

    assert report.published == 1
    assert publisher.messages[0][1] == {"job_id": str(job.job_id)}


def test_stale_attempt_cannot_write_after_reclaim(
    session: Session,
    tmp_path: Path,
) -> None:
    service, _, job = create_contract_job(session, tmp_path)
    first = service.claim(job.job_id, now=NOW)
    assert first is not None
    second = service.claim(
        job.job_id,
        now=NOW + timedelta(seconds=31),
    )
    assert second is not None

    with pytest.raises(JobAttemptOwnershipError):
        service.succeed(
            job_id=job.job_id,
            attempt_id=first.attempt_id,
            result_payload={"wrong": True},
            now=NOW + timedelta(seconds=32),
        )

    completed = service.succeed(
        job_id=job.job_id,
        attempt_id=second.attempt_id,
        result_payload={"ok": True},
        now=NOW + timedelta(seconds=32),
    )
    assert completed.status is JobStatus.SUCCEEDED


def test_duplicate_delivery_executes_handler_once(
    session: Session,
    tmp_path: Path,
) -> None:
    service, _, job = create_contract_job(session, tmp_path)
    handler = CountingHandler()
    dispatcher = WorkerDispatcher(
        jobs=service,
        handlers={
            JobKind.CONTRACT_PROCESSING: handler,
            JobKind.CONTRACT_REVIEW: handler,
            JobKind.KNOWLEDGE_INDEXING: handler,
        },
    )

    first = dispatcher.execute(job.job_id, now=NOW)
    duplicate = dispatcher.execute(
        job.job_id,
        now=NOW + timedelta(seconds=1),
    )

    assert first is not None
    assert first.status is JobStatus.SUCCEEDED
    assert duplicate is None
    assert handler.call_count == 1


def test_reconciler_retries_expired_lease(
    session: Session,
    tmp_path: Path,
) -> None:
    service, outbox, job = create_contract_job(session, tmp_path)
    publisher = RecordingPublisher()
    OutboxDispatcher(
        repository=outbox,
        publisher=publisher,
        relay_id="relay-1",
        lease_seconds=30,
        batch_size=10,
    ).dispatch(now=NOW)
    assert service.claim(job.job_id, now=NOW) is not None

    report = JobReconciler(jobs=service, outbox=outbox).reconcile(
        now=NOW + timedelta(seconds=31)
    )
    current = service.get_job(job.job_id, "system-a")
    events = outbox.list_unpublished_for_job(job.job_id)

    assert report.retried_jobs == 1
    assert current.status is JobStatus.RETRY
    assert current.dispatch_generation == 1
    assert len(events) == 1
    assert events[0].deduplication_key.endswith(":dispatch:1")


def test_reconciler_fails_after_attempt_exhaustion(
    session: Session,
    tmp_path: Path,
) -> None:
    service, outbox, job = create_contract_job(
        session,
        tmp_path,
        max_attempts=1,
    )
    assert service.claim(job.job_id, now=NOW) is not None
    report = JobReconciler(jobs=service, outbox=outbox).reconcile(
        now=NOW + timedelta(seconds=31)
    )
    current = service.get_job(job.job_id, "system-a")

    assert report.failed_jobs == 1
    assert current.status is JobStatus.FAILED
    assert current.error_code == "job_attempts_exhausted"


def test_knowledge_job_uses_snapshot_target_only(
    session: Session,
    tmp_path: Path,
) -> None:
    stored_object, _ = create_contract_target(session, tmp_path)
    document = KnowledgeDocumentRecord(
        role="knowledge",
        stored_object_id=stored_object.stored_object_id,
        source_id="policy",
        source_version="1",
        filename="contract.txt",
        access_scope="system-a",
        status=KnowledgeDocumentStatus.UPLOADED,
    )
    session.add(document)
    session.flush()
    snapshot = KnowledgeSourceSnapshotRecord(
        knowledge_document_id=document.knowledge_document_id,
        access_scope="system-a",
        source_id="policy",
        source_version="1",
        source_checksum_sha256=stored_object.checksum_sha256,
        index_profile="fake:3:block:v1",
        status=SnapshotStatus.PENDING,
    )
    session.add(snapshot)
    session.flush()
    service, _ = make_job_service(session)
    result = service.create_knowledge_indexing(
        snapshot_id=snapshot.snapshot_id,
        access_scope="system-a",
        idempotency_key="knowledge-upload-1",
        now=NOW,
    )

    assert result.job.kind is JobKind.KNOWLEDGE_INDEXING
    assert result.job.snapshot_id == snapshot.snapshot_id
    assert result.job.contract_id is None


def test_review_job_persists_intent_and_dispatches_only_job_id(
    session: Session,
    tmp_path: Path,
) -> None:
    _, contract = create_contract_target(session, tmp_path)
    service, outbox = make_job_service(session)
    unit_id = uuid4()
    knowledge_base_id = uuid4()
    result = service.create_contract_review(
        contract_id=contract.contract_id,
        access_scope="system-a",
        knowledge_base_id=knowledge_base_id,
        expected_processing_revision=1,
        review_unit_ids=(unit_id,),
        question="Does this clause follow the selected policy?",
        idempotency_key="review-1",
        now=NOW,
    )
    duplicate = service.create_contract_review(
        contract_id=contract.contract_id,
        access_scope="system-a",
        knowledge_base_id=knowledge_base_id,
        expected_processing_revision=1,
        review_unit_ids=(unit_id,),
        question="Does this clause follow the selected policy?",
        idempotency_key="review-1",
        now=NOW,
    )
    publisher = RecordingPublisher()
    OutboxDispatcher(
        repository=outbox,
        publisher=publisher,
        relay_id="relay-1",
        lease_seconds=30,
        batch_size=10,
    ).dispatch(now=NOW)

    assert result.job.kind is JobKind.CONTRACT_REVIEW
    assert duplicate.created is False
    assert duplicate.job.job_id == result.job.job_id
    assert table_count(session, TechnicalJobRecord) == 1
    assert table_count(session, OutboxEventRecord) == 1
    assert result.job.contract_id == contract.contract_id
    assert result.job.snapshot_id is None
    assert result.job.request_payload == {
        "knowledge_base_id": str(knowledge_base_id),
        "expected_processing_revision": 1,
        "review_unit_ids": [str(unit_id)],
        "question": "Does this clause follow the selected policy?",
    }
    assert publisher.messages[0][1] == {
        "job_id": str(result.job.job_id)
    }


def test_database_rejects_cross_scope_job_target(
    session: Session,
    tmp_path: Path,
) -> None:
    _, contract = create_contract_target(session, tmp_path)
    session.add(
        TechnicalJobRecord(
            kind=JobKind.CONTRACT_PROCESSING,
            status=JobStatus.QUEUED,
            stage=JobStage.QUEUED,
            access_scope="system-b",
            contract_id=contract.contract_id,
            snapshot_id=None,
            request_payload={"expected_revision": 0},
            idempotency_key="cross-scope-job",
            dispatch_generation=0,
            attempt_count=0,
            max_attempts=3,
        )
    )

    with pytest.raises(IntegrityError):
        session.flush()
